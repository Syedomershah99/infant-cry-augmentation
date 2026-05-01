"""Class-conditional DDPM on standardized log-mel spectrograms.

The denoiser is a small UNet with two downsample stages (64x128 -> 32x64 -> 16x32),
a single self-attention block at the lowest resolution, and skip-connections back
up. Time and class are embedded into a single 256-d conditioning vector that
modulates each residual block via FiLM (scale + shift).

Why this size: 320 training spectrograms is small. A 5-10M-param UNet learns
enough structure for class-conditional sampling without overfitting hard, while
keeping a 30-epoch training run inside ~10 minutes on Mac MPS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- embeddings ----------


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device, dtype=torch.float32) / max(1, half - 1)
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class Conditioning(nn.Module):
    """Concatenated time + class embedding -> single conditioning vector."""

    def __init__(self, num_classes: int, time_dim: int = 128, class_dim: int = 128, out_dim: int = 256):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        # +1 for an "unconditional" / null class index, used for classifier-free guidance.
        self.class_emb = nn.Embedding(num_classes + 1, class_dim)
        self.fuse = nn.Sequential(
            nn.Linear(time_dim + class_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, t: torch.Tensor, class_idx: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([self.time_emb(t), self.class_emb(class_idx)], dim=-1))


# ---------- UNet building blocks ----------


class FiLMResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)  # FiLM scale + shift
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.cond_proj(F.silu(cond)).chunk(2, dim=-1)
        h = self.norm2(h)
        h = h * (1 + scale[..., None, None]) + shift[..., None, None]
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Lightweight self-attention block applied at the bottleneck resolution."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        qkv = qkv.view(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = qkv.unbind(dim=1)
        attn = torch.einsum("bhci,bhcj->bhij", q, k) / math.sqrt(c // self.num_heads)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhij,bhcj->bhci", attn, v).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


# ---------- UNet ----------


class CondUNet(nn.Module):
    """Conditional UNet with N down stages, a bottleneck, and N up stages.

    Each down stage emits two skip tensors (one per residual block); the up
    stages consume them in LIFO order. Downsample/upsample modules do not
    interact with the skip stack, which is the simplest layout that keeps the
    channel arithmetic obvious.
    """

    def __init__(
        self,
        num_classes: int = 5,
        base_ch: int = 64,
        ch_mults: tuple[int, ...] = (1, 2, 4),
        cond_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cond = Conditioning(num_classes=num_classes, out_dim=cond_dim)

        chs = [base_ch * m for m in ch_mults]
        self.in_conv = nn.Conv2d(1, chs[0], 3, padding=1)

        # Down path: list of (block_a, block_b, downsample-or-None) per stage.
        self.down_stages = nn.ModuleList()
        prev = chs[0]
        skip_chs: list[int] = []
        for i, ch in enumerate(chs):
            block_a = FiLMResBlock(prev, ch, cond_dim, dropout)
            prev = ch
            skip_chs.append(prev)
            block_b = FiLMResBlock(prev, ch, cond_dim, dropout)
            skip_chs.append(prev)
            down = Downsample(ch) if i < len(chs) - 1 else nn.Identity()
            self.down_stages.append(nn.ModuleList([block_a, block_b, down]))

        # Bottleneck
        self.mid1 = FiLMResBlock(chs[-1], chs[-1], cond_dim, dropout)
        self.attn = SelfAttention2d(chs[-1])
        self.mid2 = FiLMResBlock(chs[-1], chs[-1], cond_dim, dropout)

        # Up path: mirror the down layout. Each stage pops 2 skips.
        self.up_stages = nn.ModuleList()
        chs_rev = list(reversed(chs))
        prev = chs[-1]
        for i, ch in enumerate(chs_rev):
            skip_a = skip_chs.pop()
            block_a = FiLMResBlock(prev + skip_a, ch, cond_dim, dropout)
            prev = ch
            skip_b = skip_chs.pop()
            block_b = FiLMResBlock(prev + skip_b, ch, cond_dim, dropout)
            up = Upsample(ch) if i < len(chs_rev) - 1 else nn.Identity()
            self.up_stages.append(nn.ModuleList([block_a, block_b, up]))

        self.out_norm = nn.GroupNorm(8, chs[0])
        self.out_conv = nn.Conv2d(chs[0], 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cond = self.cond(t, y)
        h = self.in_conv(x)
        skips: list[torch.Tensor] = []
        for block_a, block_b, down in self.down_stages:
            h = block_a(h, cond)
            skips.append(h)
            h = block_b(h, cond)
            skips.append(h)
            h = down(h)
        h = self.mid1(h, cond)
        h = self.attn(h)
        h = self.mid2(h, cond)
        for block_a, block_b, up in self.up_stages:
            h = block_a(torch.cat([h, skips.pop()], dim=1), cond)
            h = block_b(torch.cat([h, skips.pop()], dim=1), cond)
            h = up(h)
        return self.out_conv(F.silu(self.out_norm(h)))


# ---------- DDPM training/sampling ----------


@dataclass
class DDPMConfig:
    timesteps: int = 1_000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    cfg_dropout: float = 0.1   # prob of dropping class label during training (for CFG)


class DDPM(nn.Module):
    """Standard DDPM with epsilon-prediction. CFG ready (null-class index)."""

    def __init__(self, model: CondUNet, cfg: DDPMConfig | None = None):
        super().__init__()
        self.model = model
        self.cfg = cfg or DDPMConfig()
        betas = torch.linspace(self.cfg.beta_start, self.cfg.beta_end, self.cfg.timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        # Buffers stay on whatever device .to() moves them to.
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1.0 - alpha_bar).sqrt())
        self.null_class = self.model.num_classes  # last index reserved as null in Conditioning

    # --- training ---

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sa = self.sqrt_alpha_bar.gather(0, t).view(-1, 1, 1, 1)
        sb = self.sqrt_one_minus_alpha_bar.gather(0, t).view(-1, 1, 1, 1)
        return sa * x0 + sb * noise

    def loss(self, x0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b = x0.size(0)
        device = x0.device
        t = torch.randint(0, self.cfg.timesteps, (b,), device=device)
        # Classifier-free guidance: drop labels at rate cfg_dropout
        if self.cfg.cfg_dropout > 0:
            drop = torch.rand(b, device=device) < self.cfg.cfg_dropout
            y = torch.where(drop, torch.full_like(y, self.null_class), y)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_pred = self.model(x_t, t, y)
        return F.mse_loss(eps_pred, noise)

    # --- sampling (DDIM, deterministic, fewer steps) ---

    @torch.no_grad()
    def ddim_sample(
        self,
        shape: tuple[int, ...],
        y: torch.Tensor,
        steps: int = 50,
        cfg_scale: float = 2.0,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """y: (B,) class indices. Returns generated x of given shape."""
        x = torch.randn(shape, device=device)
        ts = torch.linspace(self.cfg.timesteps - 1, 0, steps + 1, dtype=torch.long, device=device)
        for i in range(steps):
            t_cur = ts[i].expand(shape[0])
            t_next = ts[i + 1].expand(shape[0])
            ab_cur = self.alpha_bar.gather(0, t_cur).view(-1, 1, 1, 1)
            ab_next = self.alpha_bar.gather(0, t_next.clamp_min(0)).view(-1, 1, 1, 1)
            null_y = torch.full_like(y, self.null_class)
            eps_c = self.model(x, t_cur, y)
            if cfg_scale != 1.0:
                eps_u = self.model(x, t_cur, null_y)
                eps = eps_u + cfg_scale * (eps_c - eps_u)
            else:
                eps = eps_c
            x0_pred = (x - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
            x = ab_next.sqrt() * x0_pred + (1 - ab_next).sqrt() * eps
        return x


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
