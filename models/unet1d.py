"""
1D-CNN U-Net with Feature-wise Linear Modulation (FiLM) Conditioning.

Architecture Overview:
- Input Layer: Concatenates noisy target bending moments x_t (4 channels: Bx, By, R, theta)
  with deterministic mechanistic cutting bending moments x_sim (4 channels: Bx, By, R, theta).
- Conditioning: Embeds diffusion timestep t via SinusoidalPosEmb and tool wear VB via MLP,
  injecting their summed representation into every ResnetBlock via FiLM.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for the diffusion timestep t."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer for dynamic conditioning."""

    def __init__(self, condition_dim: int, num_features: int):
        super().__init__()
        self.cond_proj = nn.Linear(condition_dim, num_features * 2)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        style = self.cond_proj(condition).unsqueeze(-1)
        scale, shift = style.chunk(2, dim=1)
        return x * (1 + scale) + shift


class Downsample(nn.Module):
    """1D strided convolution for temporal downsampling by factor of 2."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """1D transposed convolution for temporal upsampling by factor of 2."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Block(nn.Module):
    """Basic 1D convolutional block with GroupNorm, SiLU, and FiLM modulation."""

    def __init__(self, dim: int, dim_out: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(8, dim_out)
        self.act = nn.SiLU()
        self.film = FiLM(cond_dim, dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, scale_shift: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.film(x, scale_shift)
        return self.dropout(x)


class ResnetBlock(nn.Module):
    """1D Residual Block conditioned on timestep and tool wear via FiLM."""

    def __init__(self, dim: int, dim_out: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block1 = Block(dim, dim_out, cond_dim, dropout)
        self.block2 = Block(dim_out, dim_out, cond_dim, dropout)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.block1(x, cond)
        h = self.block2(h, cond)
        return h + self.res_conv(x)


class UNet1D(nn.Module):
    """
    Physics-Guided 1D-CNN U-Net Backbone for Denoising Diffusion Probabilistic Model.
    """

    def __init__(self, cfg):
        super().__init__()
        dim = getattr(cfg, "model_dim", 64)
        out_channels = getattr(cfg, "out_channels", 4)
        sim_channels = getattr(cfg, "sim_channels", 4)
        cond_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )
        self.wear_mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.init_conv = nn.Conv1d(out_channels + sim_channels, dim, 7, padding=3)

        self.downs = nn.ModuleList([
            nn.ModuleList([ResnetBlock(dim, dim, cond_dim), Downsample(dim)]),
            nn.ModuleList([ResnetBlock(dim, dim * 2, cond_dim), Downsample(dim * 2)]),
            nn.ModuleList([ResnetBlock(dim * 2, dim * 4, cond_dim), Downsample(dim * 4)]),
        ])
        self.mid_block1 = ResnetBlock(dim * 4, dim * 4, cond_dim)
        self.mid_block2 = ResnetBlock(dim * 4, dim * 4, cond_dim)
        self.ups = nn.ModuleList([
            nn.ModuleList([Upsample(dim * 4), ResnetBlock(dim * 8, dim * 2, cond_dim)]),
            nn.ModuleList([Upsample(dim * 2), ResnetBlock(dim * 4, dim, cond_dim)]),
            nn.ModuleList([Upsample(dim), ResnetBlock(dim * 2, dim, cond_dim)]),
        ])
        self.final_conv = nn.Conv1d(dim, out_channels, 1)

    def forward(
        self,
        x_noisy: torch.Tensor,
        sim_signal: torch.Tensor,
        wear: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        w_emb = self.wear_mlp(wear)
        cond = t_emb + w_emb

        # Mechanics-integrated input prior: concatenate noisy target with mechanistic simulation
        x = torch.cat([x_noisy, sim_signal], dim=1)
        x = self.init_conv(x)

        h = []
        for block, down in self.downs:
            x = block(x, cond)
            h.append(x)
            x = down(x)

        x = self.mid_block1(x, cond)
        x = self.mid_block2(x, cond)

        for up, block in self.ups:
            x = up(x)
            residual = h.pop()
            x = torch.cat((x, residual), dim=1)
            x = block(x, cond)

        return self.final_conv(x)
