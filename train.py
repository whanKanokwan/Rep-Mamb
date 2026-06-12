"""
Training script for Rep-Mamba (RSISR on AID dataset).

Paper settings (§IV-B):
  • batch_size  = 16
  • lr_patch    = 64 × 64
  • optimiser   = Adam (β1=0.9, β2=0.999)
  • lr_init     = 5 × 10⁻⁴
  • lr_decay    = 0.5 every 250 epochs
  • total_epochs= 1 000
  • loss        = L1

Usage:
  python train.py --data_root /path/to/AID --scale 4
  python train.py --data_root /path/to/AID --scale 2 --epochs 1000
"""

import argparse
import os
import time
import math
import logging
import gdown
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

from model import RepMamba
from dataset import get_dataloaders

DATASET_DIR = "./data/AID"
ZIP_FILE = "./data/AID.ZIP"

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger('rep_mamba')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    ch = logging.StreamHandler()
    fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ──────────────────────────────────────────────────────────────
# Metric helpers (operate on single uint8 / float images)
# ──────────────────────────────────────────────────────────────

def tensor_to_y(t: torch.Tensor) -> np.ndarray:
    """Convert a (3, H, W) float-[0,1] tensor to Y-channel uint8."""
    rgb = t.clamp(0, 1).permute(1, 2, 0).cpu().numpy()   # H W 3
    rgb = (rgb * 255.0).round().astype(np.uint8)
    # BT.601 RGB → Y
    y = (rgb[:, :, 0] * 65.481 +
         rgb[:, :, 1] * 128.553 +
         rgb[:, :, 2] * 24.966 +
         16.5).clip(0, 255).astype(np.uint8)
    return y


def compute_metrics(sr: torch.Tensor, hr: torch.Tensor) -> tuple[float, float]:
    """PSNR and SSIM on Y channel (as in paper §IV-C)."""
    sr_y = tensor_to_y(sr)
    hr_y = tensor_to_y(hr)
    psnr = calc_psnr(hr_y, sr_y, data_range=255)
    ssim = calc_ssim(hr_y, sr_y, data_range=255)
    return psnr, ssim


# ──────────────────────────────────────────────────────────────
# Evaluation loop
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    psnrs, ssims = [], []
    for lr, hr, _ in loader:
        lr = lr.to(device)
        hr = hr.to(device)
        sr = model(lr).clamp(0, 1)
        for i in range(sr.size(0)):
            p, s = compute_metrics(sr[i], hr[i])
            psnrs.append(p)
            ssims.append(s)
    return float(np.mean(psnrs)), float(np.mean(ssims))


# ──────────────────────────────────────────────────────────────
# Learning-rate scheduler (step ×0.5 every 250 epochs)
# ──────────────────────────────────────────────────────────────

def build_scheduler(optimizer: optim.Optimizer,
                    step_epochs: int = 250,
                    gamma: float = 0.5) -> optim.lr_scheduler.StepLR:
    return optim.lr_scheduler.StepLR(optimizer, step_size=step_epochs, gamma=gamma)


# ──────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────

def train(args):
    # ---- output dirs ----
    out_dir = Path(args.out_dir) / f'x{args.scale}'
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)
    log = setup_logger(str(out_dir / 'train.log'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # ---- data ----
    train_loader, test_loader, n_tr, n_te = get_dataloaders(
        args.data_root,
        scale=args.scale,
        lr_patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    log.info(f"Train={n_tr}  Test={n_te}  scale=×{args.scale}")

    # ---- model ----
    model = RepMamba(
        scale=args.scale,
        n_feat=args.n_feat,
        n_blocks=args.n_blocks,
        group_sizes=tuple(args.group_sizes),
        d_state=args.d_state,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"Parameters: {n_params:.3f} M")

    # ---- optimiser & scheduler ----
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, betas=(0.9, 0.999))
    scheduler = build_scheduler(optimizer,
                                step_epochs=args.lr_step,
                                gamma=args.lr_gamma)
    criterion = nn.L1Loss()
    scaler = GradScaler(enabled=args.amp)

    # ---- optional resume ----
    start_epoch = 1
    best_psnr = 0.0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr = ckpt.get('best_psnr', 0.0)
        log.info(f"Resumed from epoch {ckpt['epoch']}  best_psnr={best_psnr:.2f}")

    # ── Training ──────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for lr_img, hr_img in train_loader:
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(enabled=args.amp):
                sr = model(lr_img)
                loss = criterion(sr, hr_img)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        # ── Evaluation every eval_freq epochs ──
        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            psnr, ssim = evaluate(model, test_loader, device)
            lr_now = optimizer.param_groups[0]['lr']
            log.info(
                f"Epoch [{epoch:4d}/{args.epochs}]  "
                f"loss={avg_loss:.4f}  "
                f"PSNR={psnr:.2f} dB  SSIM={ssim:.4f}  "
                f"lr={lr_now:.2e}  time={elapsed:.1f}s"
            )
            # Save best checkpoint
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_psnr': best_psnr,
                }, ckpt_dir / 'best.pth')
                log.info(f"  ▶ Saved best model  PSNR={best_psnr:.2f} dB")
        else:
            lr_now = optimizer.param_groups[0]['lr']
            log.info(
                f"Epoch [{epoch:4d}/{args.epochs}]  "
                f"loss={avg_loss:.4f}  lr={lr_now:.2e}  time={elapsed:.1f}s"
            )

        # Save periodic checkpoint
        if epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_psnr': best_psnr,
            }, ckpt_dir / f'epoch_{epoch:04d}.pth')

    # ── Save final model ──
    torch.save(model.state_dict(), out_dir / 'final_model.pth')
    log.info(f"Training complete. Best PSNR={best_psnr:.2f} dB")


# ──────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser('Rep-Mamba Training')

    # Data
    p.add_argument('--data_root', type=str, required=True,
                   help='Path to AID dataset root (contains 30 class folders)')
    p.add_argument('--out_dir', type=str, default='./results',
                   help='Directory for checkpoints and logs')
    p.add_argument('--num_workers', type=int, default=4)

    # SR settings
    p.add_argument('--scale', type=int, default=4, choices=[2, 3, 4])
    p.add_argument('--patch_size', type=int, default=64,
                   help='LR patch size (HR = patch_size × scale)')

    # Model (paper defaults)
    p.add_argument('--n_feat', type=int, default=64)
    p.add_argument('--n_blocks', type=int, default=6)
    p.add_argument('--group_sizes', type=int, nargs='+', default=[8, 8, 16, 32])
    p.add_argument('--d_state', type=int, default=16)

    # Training (paper defaults)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--lr_step', type=int, default=250,
                   help='Decay lr by lr_gamma every this many epochs')
    p.add_argument('--lr_gamma', type=float, default=0.5)

    # Misc
    p.add_argument('--amp', action='store_true', default=True,
                   help='Use automatic mixed precision (faster on RTX GPUs)')
    p.add_argument('--resume', type=str, default=None,
                   help='Path to checkpoint to resume from')
    p.add_argument('--eval_freq', type=int, default=10,
                   help='Evaluate on test set every N epochs')
    p.add_argument('--save_freq', type=int, default=100,
                   help='Save checkpoint every N epochs')

    return p.parse_args()

	
def donwload_dataset():
	if os.path.exists(DATASET_DIR):
		print("Dataset already exists.")
		return
		
	os.makedirs("./data",exist_ok=True)
	
	file_id = "1safmD7VOs8pwOjPiMt_hbetR2z7I1eIo"
	url = f"https://drive.google.com/uc?id={file_id}"
	
	print("Downloading AID dataset..")
	 gdown.download(url, ZIP_FILE, quiet=False)
	 
	print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall("./data")
		
	print("Dataset ready.")
	

if __name__ == '__main__':
    args = parse_args()
	donwload_dataset()
    train(args)
