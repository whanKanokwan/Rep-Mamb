# Rep-Mamba

PyTorch implementation of **Rep-Mamba: Re-Parameterization in Vision Mamba for Lightweight Remote Sensing Image Super-Resolution** (IEEE TGRS 2025).

---

## Architecture

```
I_LR  →  Conv (shallow features F0)
          └── LPFM × 6 (each followed by Conv + residual skip)  →  F_DF
          F0 + F_DF  →  PixelShuffle × scale  →  I_SR
```

| Module | Role |
|---|---|
| **RepConv** | 3×3 + 1×1 + identity branches during training; single 3×3 at inference |
| **SS2D** | 4-directional VMamba selective scan (pure-PyTorch; optional mamba-ssm CUDA kernel) |
| **RMB** | Dual-branch block: global (SS2D on downsampled feature) + local (RepConv) |
| **CSSP** | 4 progressive RMB branches with channels (8→16→32→64) via cat-and-process |
| **ConvFFN** | Linear → DW-Conv → Linear feed-forward |
| **LPFM** | LN → CSSP + skip, then LN → ConvFFN + skip |

Paper model specs: **6 LPFM blocks**, **64 channels**, **0.785 M params**, **10.92 GFLOPs** at ×4.

---

## AID Dataset

Download: https://captain-whu.github.io/DiRS/

Expected folder layout:
```
<data_root>/
    Airport/
    BareLand/
    BaseballField/
    ... (30 categories)
```

Paper split: **100 train / 30 test** images per category (3 000 / 900 total).

---

## Installation

```bash
pip install torch torchvision einops scikit-image Pillow

# Optional: much faster selective scan on CUDA GPUs
pip install causal_conv1d mamba-ssm
```

---

## Training

```bash
# ×4 super-resolution (paper default)
python train.py --data_root /path/to/AID --scale 4

# ×2 or ×3
python train.py --data_root /path/to/AID --scale 2
python train.py --data_root /path/to/AID --scale 3
```

Key hyper-parameters (matching the paper §IV-B):

| Parameter | Value |
|---|---|
| Batch size | 16 |
| LR patch | 64 × 64 |
| Optimiser | Adam (β₁=0.9, β₂=0.999) |
| Learning rate | 5 × 10⁻⁴ |
| LR decay | ×0.5 every 250 epochs |
| Total epochs | 1 000 |
| Loss | L1 |

Checkpoints and logs are saved to `./results/x<scale>/`.

---

## Evaluation

```bash
# Full AID test set (PSNR/SSIM on Y channel)
python test.py --data_root /path/to/AID --ckpt results/x4/best.pth --scale 4

# Single image inference
python test.py --img /path/to/lr.png --ckpt results/x4/best.pth --scale 4
```

---

## Target Results on AID (from Table I)

| Scale | PSNR (dB) | SSIM |
|---|---|---|
| ×2 | 35.58 | 0.9398 |
| ×3 | 29.96 | 0.8140 |
| ×4 | 29.37 | 0.7901 |

---

## Quick model check

```bash
python model.py   # prints param count and runs a forward pass
```

---

## Citation

```bibtex
@article{jiang2025repmamba,
  title={Rep-Mamba: Re-Parameterization in Vision Mamba for Lightweight Remote Sensing Image Super-Resolution},
  author={Jiang, Kui and Yang, Mengru and Xiao, Yi and Wu, Jianbo and Wang, Guangcheng and Feng, Xiaocheng and Jiang, Junjun},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={63},
  year={2025},
  doi={10.1109/TGRS.2025.3597745}
}
```
