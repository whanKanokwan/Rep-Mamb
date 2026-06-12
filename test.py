"""
Evaluation / inference script for Rep-Mamba.

Usage:
  # Evaluate a trained model on AID test set
  python test.py --data_root /path/to/AID --ckpt results/x4/best.pth --scale 4

  # Run on a single image (saves the SR result)
  python test.py --img /path/to/lr.png --ckpt results/x4/best.pth --scale 4

Paper metrics (§IV-C):
  • PSNR and SSIM computed on the Y channel of YCbCr colour space.
  • Averaged over all test images.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

from model import RepMamba
from dataset import get_dataloaders


# ──────────────────────────────────────────────────────────────
# Metric helpers (Y channel, as in paper §IV-C)
# ──────────────────────────────────────────────────────────────

def tensor_to_y(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) float [0,1] tensor → Y-channel uint8 array."""
    rgb = t.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    rgb = (rgb * 255.0).round().astype(np.float32)
    y = (rgb[:, :, 0] * 65.481 +
         rgb[:, :, 1] * 128.553 +
         rgb[:, :, 2] * 24.966 + 16.5).clip(0, 255).astype(np.uint8)
    return y


def compute_metrics(sr: torch.Tensor, hr: torch.Tensor):
    sr_y = tensor_to_y(sr)
    hr_y = tensor_to_y(hr)
    psnr = calc_psnr(hr_y, sr_y, data_range=255)
    ssim = calc_ssim(hr_y, sr_y, data_range=255)
    return psnr, ssim


# ──────────────────────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, args, device: torch.device,
               deploy: bool = True) -> nn.Module:
    model = RepMamba(
        scale=args.scale,
        n_feat=args.n_feat,
        n_blocks=args.n_blocks,
        group_sizes=tuple(args.group_sizes),
        d_state=args.d_state,
        deploy=False,           # load in train mode first
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    # Support both plain state_dict and wrapped checkpoints
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=True)

    if deploy:
        model.reparameterize()   # fuse RepConv branches for faster inference
        print("[Info] RepConv branches reparameterized (inference mode).")

    model.eval()
    return model


# ──────────────────────────────────────────────────────────────
# Evaluation on AID test set
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_dataset(model: nn.Module, loader, device: torch.device,
                     save_dir: str = None):
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    psnrs, ssims, times = [], [], []
    for i, (lr, hr, path) in enumerate(loader):
        lr = lr.to(device)
        hr = hr.to(device)

        t0 = time.time()
        sr = model(lr).clamp(0, 1)
        elapsed = (time.time() - t0) * 1000      # ms

        for b in range(sr.size(0)):
            p, s = compute_metrics(sr[b], hr[b])
            psnrs.append(p)
            ssims.append(s)
            times.append(elapsed)

            if save_dir:
                stem = Path(path[b]).stem
                sr_img = TF.to_pil_image(sr[b].cpu())
                sr_img.save(Path(save_dir) / f'{stem}_SR.png')

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(loader)}]  "
                  f"PSNR={np.mean(psnrs):.2f}  SSIM={np.mean(ssims):.4f}")

    mean_psnr = float(np.mean(psnrs))
    mean_ssim = float(np.mean(ssims))
    mean_time = float(np.mean(times))
    print(f"\n{'='*55}")
    print(f"  Scale ×{args.scale}  |  {len(psnrs)} images")
    print(f"  PSNR  : {mean_psnr:.4f} dB")
    print(f"  SSIM  : {mean_ssim:.4f}")
    print(f"  Avg inference time : {mean_time:.1f} ms")
    print(f"{'='*55}\n")
    return mean_psnr, mean_ssim


# ──────────────────────────────────────────────────────────────
# Single image SR
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def sr_single_image(model: nn.Module, img_path: str,
                    device: torch.device,
                    out_path: str = None) -> str:
    lr_img = Image.open(img_path).convert('RGB')
    lr_t = TF.to_tensor(lr_img).unsqueeze(0).to(device)

    t0 = time.time()
    sr_t = model(lr_t).clamp(0, 1).squeeze(0).cpu()
    elapsed = (time.time() - t0) * 1000

    sr_pil = TF.to_pil_image(sr_t)
    if out_path is None:
        stem = Path(img_path).stem
        out_path = str(Path(img_path).parent / f'{stem}_x{args.scale}_SR.png')
    sr_pil.save(out_path)
    print(f"SR image saved: {out_path}  ({elapsed:.1f} ms)")
    return out_path


# ──────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser('Rep-Mamba Evaluation')
    p.add_argument('--ckpt', type=str, required=True,
                   help='Path to trained checkpoint (.pth)')
    p.add_argument('--scale', type=int, default=4, choices=[2, 3, 4])

    # Dataset (for full evaluation)
    p.add_argument('--data_root', type=str, default=None,
                   help='AID root directory (for full test-set evaluation)')
    p.add_argument('--num_workers', type=int, default=4)

    # Single image (optional)
    p.add_argument('--img', type=str, default=None,
                   help='Path to a single LR image for SR inference')
    p.add_argument('--out_img', type=str, default=None,
                   help='Path to save the super-resolved output')

    # Save SR results
    p.add_argument('--save_dir', type=str, default=None,
                   help='Directory to save SR images during dataset evaluation')

    # Model architecture (must match the trained model)
    p.add_argument('--n_feat', type=int, default=64)
    p.add_argument('--n_blocks', type=int, default=6)
    p.add_argument('--group_sizes', type=int, nargs='+', default=[8, 8, 16, 32])
    p.add_argument('--d_state', type=int, default=16)

    # Misc
    p.add_argument('--no_deploy', action='store_true',
                   help='Keep RepConv in training mode (no reparameterization)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(args.ckpt, args, device, deploy=not args.no_deploy)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.3f} M")

    if args.img is not None:
        sr_single_image(model, args.img, device, args.out_img)

    if args.data_root is not None:
        _, test_loader, _, _ = get_dataloaders(
            args.data_root,
            scale=args.scale,
            batch_size=1,
            num_workers=args.num_workers,
        )
        evaluate_dataset(model, test_loader, device, save_dir=args.save_dir)
