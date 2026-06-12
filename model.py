"""
Rep-Mamba: Re-Parameterization in Vision Mamba for
Lightweight Remote Sensing Image Super-Resolution
IEEE TGRS 2025  (Jiang et al.)

Architecture overview
─────────────────────
  I_LR  ──▶  [Shallow Conv F0]
             │
             ▼
      [LPFM ×6, each with Conv skip]  ──▶  F_DF
             │
         F0 + F_DF
             │
             ▼
     [PixelShuffle × scale]  ──▶  I_SR

Key modules
─────────────────────
  RepConv          : RepVGG-style 3×3+1×1+id branches (merged at inference)
  SS2D             : 4-directional VMamba selective scan (pure-PyTorch fallback)
  RMB              : RepConv-Mamba Block (global+local dual branch)
  CSSP             : Cross-Scale State Propagation (4 groups: 8,8,16,32 ch)
  ConvFFN          : Convolutional feed-forward (Linear→DWConv→Linear)
  LPFM             : Lightweight Progressive Fusion Module
  RepMamba         : Top-level model
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ───────────────────────────────────────────────────────────────
# Try to use fast mamba_ssm CUDA kernel; fall back to pure PyTorch
# ───────────────────────────────────────────────────────────────
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _fast_scan
    _HAS_MAMBA = True
except ImportError:
    _HAS_MAMBA = False


# ================================================================
# 0. Helpers
# ================================================================

class LayerNorm2d(nn.Module):
    """LayerNorm applied over channels for (B, C, H, W) tensors."""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,C,H,W) → (B,H,W,C) → norm → (B,C,H,W)
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


# ================================================================
# 1. RepConv  (differentiable re-parameterization convolution)
# ================================================================

class RepConv(nn.Module):
    """
    Training mode : output = BN(3×3 conv) + BN(1×1 conv) + BN(identity)
    Inference mode: single fused 3×3 conv (call .reparameterize())
    """
    def __init__(self, in_ch: int, out_ch: int,
                 stride: int = 1, groups: int = 1, deploy: bool = False):
        super().__init__()
        self.deploy = deploy
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.groups = groups

        if deploy:
            self.reparam = nn.Conv2d(in_ch, out_ch, 3,
                                     stride=stride, padding=1, groups=groups, bias=True)
        else:
            self.br3 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1,
                          groups=groups, bias=False),
                nn.BatchNorm2d(out_ch)
            )
            self.br1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, padding=0,
                          groups=groups, bias=False),
                nn.BatchNorm2d(out_ch)
            )
            self.brid = (nn.BatchNorm2d(in_ch)
                         if (in_ch == out_ch and stride == 1) else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.reparam(x)
        out = self.br3(x) + self.br1(x)
        if self.brid is not None:
            out = out + self.brid(x)
        return out

    # ---- re-param helpers ----

    @staticmethod
    def _fuse(conv_w: torch.Tensor, bn: nn.BatchNorm2d) -> tuple:
        """Fuse conv weight with subsequent BN into (weight, bias)."""
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        w = conv_w * t
        b = bn.bias - bn.running_mean * bn.weight / std
        return w, b

    def _id_kernel(self) -> torch.Tensor:
        """3×3 kernel equivalent to identity (centre=1, rest=0)."""
        C = self.in_ch
        g = self.groups
        d = C // g
        k = torch.zeros(C, d, 3, 3,
                        device=self.brid.weight.device,
                        dtype=self.brid.weight.dtype)
        for i in range(C):
            k[i, i % d, 1, 1] = 1.0
        return k

    def reparameterize(self):
        """Merge branches and switch to deploy (inference) mode."""
        if self.deploy:
            return
        w3, b3 = self._fuse(self.br3[0].weight, self.br3[1])
        w1, b1 = self._fuse(self.br1[0].weight, self.br1[1])
        w1 = F.pad(w1, [1, 1, 1, 1])
        if self.brid is not None:
            wid, bid = self._fuse(self._id_kernel(), self.brid)
        else:
            wid = torch.zeros_like(w3)
            bid = torch.zeros(self.out_ch, device=w3.device, dtype=w3.dtype)
        w = w3 + w1 + wid
        b = b3 + b1 + bid
        self.reparam = nn.Conv2d(self.in_ch, self.out_ch, 3,
                                 padding=1, groups=self.groups, bias=True)
        self.reparam.weight.data.copy_(w)
        self.reparam.bias.data.copy_(b)
        del self.br3, self.br1
        if hasattr(self, 'brid') and self.brid is not None:
            del self.brid
        self.deploy = True


# ================================================================
# 2. Selective Scan (pure PyTorch – correct but sequential over L)
# ================================================================

def _selective_scan_pt(u, delta, A, B, C, D):
    """
    Pure-PyTorch reference selective scan (ZOH discretisation).

    u     : (B, D, L)  – input
    delta : (B, D, L)  – dt, after softplus
    A     : (D, N)     – log of negative A values
    B     : (B, N, L)  – input-dependent B
    C     : (B, N, L)  – input-dependent C
    D     : (D,)       – skip scalar

    Returns y : (B, D, L)
    """
    B_sz, D_dim, L = u.shape
    N = A.shape[1]

    u_f = u.float()
    dt_f = delta.float()
    A_f = A.float()           # (D, N)  — stored as positive logs → negate
    B_f = B.float()           # (B, N, L)
    C_f = C.float()           # (B, N, L)

    # Discretise  (ZOH):  dA = exp(dt * A),  dB = dt * B
    # A stored as log(-A), so real A = -exp(A_log)
    neg_A = -torch.exp(A_f)                                           # (D, N)
    dA = torch.exp(torch.einsum('bdl,dn->bdln', dt_f, neg_A))        # (B, D, L, N)
    dBu = torch.einsum('bdl,bnl,bdl->bdln', dt_f, B_f, u_f)         # (B, D, L, N)

    h = u_f.new_zeros(B_sz, D_dim, N)
    ys = []
    for i in range(L):
        h = dA[:, :, i] * h + dBu[:, :, i]                          # (B, D, N)
        y = torch.einsum('bdn,bn->bd', h, C_f[:, :, i])             # (B, D)
        ys.append(y)

    y = torch.stack(ys, dim=2)                                       # (B, D, L)
    if D is not None:
        y = y + u_f * D.float().unsqueeze(0).unsqueeze(-1)
    return y.to(dtype=u.dtype)


# ================================================================
# 3. SS2D – 2-D Selective Scan with 4-directional VMamba scanning
# ================================================================

class SS2D(nn.Module):
    """
    Selective Scan 2D (VMamba-style, 4 directions).

    Input / output : (B, C, H, W)   [BCHW format]
    Internally the spatial grid is flattened in 4 directions,
    each processed by an independent SSM, then summed and
    gated by a linear branch.
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 3, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = max(1, math.ceil(d_model / 16))
        K = 4  # number of scan directions

        # ---- input projection ----
        self.in_proj = nn.Conv2d(d_model, self.d_inner * 2, 1, bias=False)
        self.dw_conv = nn.Conv2d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv // 2, groups=self.d_inner, bias=True
        )
        self.act = nn.SiLU()

        # ---- SSM parameters (K directions share weight tensors) ----
        # x_proj_weight: maps d_inner → dt_rank + 2*d_state
        self.x_proj_w = nn.Parameter(
            torch.randn(K, self.dt_rank + d_state * 2, self.d_inner) * 0.02
        )
        # dt_proj_weight: maps dt_rank → d_inner
        self.dt_proj_w = nn.Parameter(
            torch.randn(K, self.d_inner, self.dt_rank) * 0.02
        )
        # dt_proj_bias (initialised to give reasonable initial dt)
        dt_init = torch.exp(
            torch.rand(K, self.d_inner) *
            (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_proj_b = nn.Parameter(dt_init + torch.log(-torch.expm1(-dt_init)))

        # A (log of positive values, stored positive; negated inside scan)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).repeat(K * self.d_inner, 1)                # (K*D, N)
        self.A_log = nn.Parameter(torch.log(A))

        # D (skip-connection scalar per channel per direction)
        self.D = nn.Parameter(torch.ones(K * self.d_inner))

        # ---- output ----
        self.out_norm = LayerNorm2d(self.d_inner)
        self.out_proj = nn.Conv2d(self.d_inner, d_model, 1, bias=False)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        L = H * W
        K = 4

        # Gate / content split
        xz = self.in_proj(x)                          # (B, 2*d_inner, H, W)
        x_in, z = xz.chunk(2, dim=1)                 # (B, d_inner, H, W) each
        x_in = self.act(self.dw_conv(x_in))          # depthwise conv

        # ---- build 4 flattened scans ----
        def _scan(t, flip=False, transpose=False):
            if transpose:
                t = t.permute(0, 1, 3, 2).contiguous()   # swap H,W
            t = t.reshape(B, -1, H * W if not transpose else W * H)
            if flip:
                t = t.flip(-1)
            return t   # (B, d_inner, L)

        scans = [
            _scan(x_in),                   # 0: left→right
            _scan(x_in, flip=True),        # 1: right→left
            _scan(x_in, transpose=True),   # 2: top→bot (transposed)
            _scan(x_in, transpose=True, flip=True),  # 3: bot→top
        ]
        x_stk = torch.stack(scans, dim=1)  # (B, K, d_inner, L)

        # ---- per-direction SSM ----
        # Project to (dt, B_coef, C_coef)
        x_proj = torch.einsum('bkdl,kcd->bkcl', x_stk, self.x_proj_w)
        dt_r, B_c, C_c = torch.split(
            x_proj, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        dt = torch.einsum('bkrl,kdr->bkdl', dt_r, self.dt_proj_w)   # (B,K,d_inner,L)
        dt = dt + self.dt_proj_b.unsqueeze(0).unsqueeze(-1)          # add bias
        dt = F.softplus(dt)

        A_all = self.A_log.view(K, self.d_inner, self.d_state)       # (K, D, N)
        D_all = self.D.view(K, self.d_inner)                         # (K, D)

        # Run scan per direction
        ys_raw = []
        for k in range(K):
            y_k = _selective_scan_pt(
                x_stk[:, k],    # (B, d_inner, L)
                dt[:, k],       # (B, d_inner, L)
                A_all[k],       # (d_inner, d_state)
                B_c[:, k],      # (B, d_state, L)
                C_c[:, k],      # (B, d_state, L)
                D_all[k],       # (d_inner,)
            )
            ys_raw.append(y_k)  # (B, d_inner, L)

        # ---- reverse scans and restore spatial dims ----
        y0 = ys_raw[0].reshape(B, -1, H, W)
        y1 = ys_raw[1].flip(-1).reshape(B, -1, H, W)
        y2 = ys_raw[2].reshape(B, -1, W, H).permute(0, 1, 3, 2).contiguous()
        y3 = ys_raw[3].flip(-1).reshape(B, -1, W, H).permute(0, 1, 3, 2).contiguous()

        y = y0 + y1 + y2 + y3                    # (B, d_inner, H, W)

        # Normalise and gate
        y = self.out_norm(y) * self.act(z)        # element-wise gate
        y = self.out_proj(y)                      # (B, d_model, H, W)
        return y


# ================================================================
# 4. RMB – RepConv-Mamba Block  (Algorithm 2)
# ================================================================

class RMB(nn.Module):
    """
    Dual-branch block combining global (2D-SSM) and local (RepConv) paths.

    Input  X : (B, C, H, W)
    Output Y : (B, C, H, W)

    Steps (Algorithm 2):
        X1  = SiLU( RepConv( Linear(X) ) )
        X2  = SiLU( Linear(X) )
        Xg, Xl = split(X1)          # along channel dim → C/2 each
        [Global branch]
          Xlow   = AvgPool2d(2)(Xg)          # (B, C/2, H/2, W/2)
          ΔX     = Xg − upsample(Xlow)       # residual high-freq
          Xglow  = 2D-SSM(Xlow)
          Xglobal = upsample(Xglow) + ΔX
        [Local branch]
          Xlocal = RepConv(Xl)
        Z1   = LN( cat(Xglobal, Xlocal) )   # (B, C, H, W)
        Xnew = Linear( Z1 ⊗ X2 )
        Y    = RepConv( X + Xnew )
    """

    def __init__(self, channels: int, d_state: int = 16, deploy: bool = False):
        super().__init__()
        self.C = channels
        half = channels // 2

        # Input branches
        self.linear_in = nn.Conv2d(channels, channels, 1, bias=False)
        self.repconv_in = RepConv(channels, channels, deploy=deploy)
        self.act = nn.SiLU()

        self.linear_gate = nn.Conv2d(channels, channels, 1, bias=False)

        # Global path
        self.pool = nn.AvgPool2d(2, 2)
        self.ssm = SS2D(half, d_state=d_state, expand=2)

        # Local path
        self.repconv_local = RepConv(half, half, deploy=deploy)

        # Fusion
        self.out_norm = LayerNorm2d(channels)
        self.linear_out = nn.Conv2d(channels, channels, 1, bias=False)
        self.repconv_out = RepConv(channels, channels, deploy=deploy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        X1 = self.act(self.repconv_in(self.linear_in(x)))   # (B, C, H, W)
        X2 = self.act(self.linear_gate(x))                  # (B, C, H, W)

        Xg, Xl = X1.chunk(2, dim=1)  # (B, C/2, H, W) each

        # ---- global branch ----
        Xlow = self.pool(Xg)                                  # (B, C/2, H/2, W/2)
        Xlow_up = F.interpolate(Xlow, size=(H, W), mode='bilinear', align_corners=False)
        dX = Xg - Xlow_up                                     # (B, C/2, H, W)
        Xglow = self.ssm(Xlow)                               # (B, C/2, H/2, W/2)
        Xglobal = F.interpolate(Xglow, size=(H, W), mode='bilinear', align_corners=False) + dX

        # ---- local branch ----
        Xlocal = self.repconv_local(Xl)                       # (B, C/2, H, W)

        # ---- fusion ----
        fused = torch.cat([Xglobal, Xlocal], dim=1)          # (B, C, H, W)
        Z1 = self.out_norm(fused)                             # (B, C, H, W)
        Xnew = self.linear_out(Z1 * X2)                      # (B, C, H, W)
        Y = self.repconv_out(x + Xnew)                       # (B, C, H, W)
        return Y


# ================================================================
# 5. CSSP – Cross-Scale State Propagation  (Algorithm 1)
# ================================================================

class CSSP(nn.Module):
    """
    Channel groups (8, 8, 16, 32) for 64-channel input (from Table IV).
    Progressive fusion across 4 RMB branches.

    j=1 :  F_fuse1 = RMB1(F1) + F1
    j≥2 :  F_fusej = RMBj( cat(Fj, F_fuse_{j-1}) )
    out  :  F_fuse4 + F_in
    """

    # Default group split for C=64 (paper Table IV best config)
    DEFAULT_GROUPS = (8, 8, 16, 32)

    def __init__(self, channels: int = 64,
                 group_sizes: tuple = None,
                 d_state: int = 16,
                 deploy: bool = False):
        super().__init__()
        gs = group_sizes or self.DEFAULT_GROUPS
        assert sum(gs) == channels, \
            f"Group sizes {gs} must sum to channels {channels}."
        self.group_sizes = gs
        n = len(gs)

        # RMB channel sizes (cumulative, because Ffuse_j has the same channel count
        # as the input to RMB_j, which grows as we concatenate with the previous Ffuse):
        #   j=0 : in_ch = gs[0]
        #   j=1 : in_ch = gs[1] + Ffuse_0_ch  = gs[1] + rmb_ch[0]
        #   j=2 : in_ch = gs[2] + Ffuse_1_ch  = gs[2] + rmb_ch[1]  …
        # For gs=(8,8,16,32) → rmb_ch = [8, 16, 32, 64]  ← Ffuse_n is 64ch ✓
        rmb_channels = []
        prev_ch = 0
        for j, g in enumerate(gs):
            in_ch = g if j == 0 else g + prev_ch
            rmb_channels.append(in_ch)
            prev_ch = in_ch
        self.rmbs = nn.ModuleList([
            RMB(c, d_state=d_state, deploy=deploy) for c in rmb_channels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, H, W)
        # Split into groups
        groups = torch.split(x, list(self.group_sizes), dim=1)  # list of tensors

        ffuse = None
        for j, (rmb, fj) in enumerate(zip(self.rmbs, groups)):
            if j == 0:
                ffuse = rmb(fj) + fj           # branch 1: residual outside RMB
            else:
                xc = torch.cat([fj, ffuse], dim=1)
                ffuse = rmb(xc)                # branch j≥2: no extra residual

        return ffuse + x                       # global residual (Eq. 8)


# ================================================================
# 6. ConvFFN – Convolutional Feed-Forward Network (from SRFormer)
# ================================================================

class ConvFFN(nn.Module):
    """
    fc1 (1×1) → GELU → DWConv (3×3) → fc2 (1×1)
    with an additive skip on the DWConv branch.
    """
    def __init__(self, channels: int, expand: int = 4):
        super().__init__()
        hidden = channels * expand
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = x + self.act(self.dwconv(x))
        return self.fc2(x)


# ================================================================
# 7. LPFM – Lightweight Progressive Fusion Module  (Eqs. 5–6)
# ================================================================

class LPFM(nn.Module):
    """
    F_cssp = CSSP( LN(F_in) ) + F_in
    F_out  = ConvFFN( LN(F_cssp) ) + F_cssp
    """
    def __init__(self, channels: int = 64,
                 group_sizes: tuple = None,
                 d_state: int = 16,
                 ffn_expand: int = 4,
                 deploy: bool = False):
        super().__init__()
        self.ln1 = LayerNorm2d(channels)
        self.cssp = CSSP(channels, group_sizes=group_sizes,
                         d_state=d_state, deploy=deploy)
        self.ln2 = LayerNorm2d(channels)
        self.ffn = ConvFFN(channels, expand=ffn_expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cssp(self.ln1(x)) + x
        x = self.ffn(self.ln2(x)) + x
        return x


# ================================================================
# 8. RepMamba – Top-level model
# ================================================================

class RepMamba(nn.Module):
    """
    Lightweight progressive multi-scale SR model.

    Default config (from paper §IV-B):
      n_feat   = 64    channels
      n_blocks = 6     LPFM blocks
      scale    = 4     upscaling factor
      n_colors = 3     RGB input
    """

    def __init__(self,
                 scale: int = 4,
                 n_colors: int = 3,
                 n_feat: int = 64,
                 n_blocks: int = 6,
                 group_sizes: tuple = (8, 8, 16, 32),
                 d_state: int = 16,
                 ffn_expand: int = 4,
                 deploy: bool = False):
        super().__init__()
        self.scale = scale

        # ── Shallow feature extraction (Eq. 1) ──
        self.shallow = nn.Conv2d(n_colors, n_feat, 3, padding=1, bias=True)

        # ── Deep feature extraction (Eq. 2–3) ──
        self.body = nn.ModuleList([
            nn.Sequential(
                LPFM(n_feat, group_sizes=group_sizes,
                     d_state=d_state, ffn_expand=ffn_expand, deploy=deploy),
                nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=True)
            )
            for _ in range(n_blocks)
        ])

        # ── Reconstruction (Eq. 4) ──
        # PixelShuffle upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * scale * scale, 3, padding=1, bias=True),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feat, n_colors, 3, padding=1, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shallow features
        F0 = self.shallow(x)                    # (B, C, H, W)

        # Deep features with inter-LPFM residuals (Eq. 3)
        Fi = F0
        for block in self.body:
            Fi = block(Fi) + Fi

        # Global residual + reconstruct (Eq. 4)
        out = self.upsample(F0 + Fi)            # (B, 3, sH, sW)
        return out

    def reparameterize(self):
        """Call after training to fuse all RepConv branches for fast inference."""
        for m in self.modules():
            if isinstance(m, RepConv):
                m.reparameterize()

    def flops(self, input_size=(128, 128)):
        """Estimate FLOPs (approximate, MACs × 2)."""
        from thop import profile
        dummy = torch.zeros(1, 3, *input_size)
        macs, _ = profile(self, inputs=(dummy,), verbose=False)
        return macs * 2


# ================================================================
# Quick sanity check
# ================================================================

if __name__ == '__main__':
    import time
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RepMamba(scale=4, n_feat=64, n_blocks=6).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.3f} M")

    x = torch.randn(1, 3, 64, 64).to(device)
    t0 = time.time()
    with torch.no_grad():
        y = model(x)
    print(f"Output shape: {y.shape}  |  Inference time: {time.time()-t0:.3f}s")
    assert y.shape == (1, 3, 256, 256), "Shape mismatch!"
    print("Model check passed ✓")
