"""
v4.5 Global-Context CNN-Transformer Direct-SDF Generator (`GlobalGlyphTransformerV45`).

Key fixes over v4.0-v4.3 (from 审计v4.md + v4.4 direct-SDF discovery):
  1. Direct SDF output head (`pred_sdf = tanh(sdf_head([u0, warped_bases]))`):
     eliminates `base_blend` trunc_px=16 hard-clip 90-deg steps & `active_gate` dead zones.
  2. Expanded flow displacement (`max_shift_px = 14.0`, ~65 UPEM) with tiny non-zero init
     so Stage 1 never suffers zero-gradient deadlock.
  3. High-resolution `up0` (`128 -> 256` with `f0` 128x128 skip + full-res `ResBlock`).
  4. ~36.0M parameters, 256x256 full-glyph context, clean FP32/BF16 compatibility.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBlock(nn.Module):
    """Conv + GroupNorm + SiLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, groups: int = 8) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=pad, bias=False),
            nn.GroupNorm(min(groups, out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    """Translation-equivariant 3x3 residual conv block."""

    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        g = min(groups, channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class SincosPositionalEncoding(nn.Module):
    """Fixed 2D sinusoidal positional encoding for (grid_h * grid_w) patch tokens."""

    def __init__(self, grid_h: int, grid_w: int, d_model: int) -> None:
        super().__init__()
        half = d_model // 4
        freqs = torch.exp(torch.arange(half).float() * (-math.log(10000.0) / max(half, 1)))

        rows = torch.arange(grid_h).float()
        cols = torch.arange(grid_w).float()
        grid_row, grid_col = torch.meshgrid(rows, cols, indexing="ij")

        r_flat = grid_row.flatten()
        c_flat = grid_col.flatten()

        pe = torch.cat(
            [
                torch.sin(r_flat.unsqueeze(1) * freqs.unsqueeze(0)),
                torch.cos(r_flat.unsqueeze(1) * freqs.unsqueeze(0)),
                torch.sin(c_flat.unsqueeze(1) * freqs.unsqueeze(0)),
                torch.cos(c_flat.unsqueeze(1) * freqs.unsqueeze(0)),
            ],
            dim=-1,
        )  # [H*W, d_model]
        self.register_buffer("pos_embed", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed[:, : x.size(1), :]


class GlobalGlyphTransformerV45(nn.Module):
    """
    Hybrid CNN-Encoder + 4-Layer Transformer + High-Res CNN-Decoder + Direct SDF Head.

    Input:  x        [B, 4, 256, 256]  (sdf_ref, skel_sdf, cos2t, sin2t)
    Output: pred_sdf [B, 1, 256, 256]  in [-1, 1]
            flow_px  [B, 2, 256, 256]  in [-max_shift_px, +max_shift_px]
    """

    def __init__(
        self,
        *,
        in_channels: int = 4,
        canvas_size: int = 256,
        patch_size: int = 16,
        d_model: int = 768,
        n_heads: int = 12,
        n_transformer_layers: int = 4,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
        max_shift_px: float = 14.0,
    ) -> None:
        super().__init__()
        self.canvas_size = canvas_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.max_shift_px = max_shift_px

        self.grid_h = canvas_size // patch_size
        self.grid_w = canvas_size // patch_size

        c1, c2, c3, c4 = 48, 96, 192, 384

        # CNN Encoder: 256 -> 128 -> 64 -> 32 -> 16
        self.stem = nn.Sequential(CNNBlock(in_channels, c1, kernel_size=7, stride=2), ResBlock(c1))
        self.down1 = nn.Sequential(CNNBlock(c1, c2, kernel_size=3, stride=2), ResBlock(c2))
        self.down2 = nn.Sequential(CNNBlock(c2, c3, kernel_size=3, stride=2), ResBlock(c3))
        self.down3 = nn.Sequential(CNNBlock(c3, c4, kernel_size=3, stride=2), ResBlock(c4))

        # Patchify & Global Transformer (256 tokens @ d_model=768)
        self.patch_conv = nn.Conv2d(c4, d_model, kernel_size=1, bias=False)
        self.pos_enc = SincosPositionalEncoding(self.grid_h, self.grid_w, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffn_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers, enable_nested_tensor=False
        )
        self.unpatch_conv = nn.Conv2d(d_model, c4, kernel_size=1, bias=False)

        # High-Res CNN Decoder with skip connections: 16 -> 32 -> 64 -> 128 -> 256
        self.up3 = nn.Sequential(
            CNNBlock(c4 * 2, c3),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ResBlock(c3),
        )
        self.up2 = nn.Sequential(
            CNNBlock(c3 * 2, c2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ResBlock(c2),
        )
        self.up1 = nn.Sequential(
            CNNBlock(c2 * 2, c1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ResBlock(c1),
        )
        self.up0 = nn.Sequential(
            CNNBlock(c1 * 2, c1 // 2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ResBlock(c1 // 2),
        )

        # Flow & Direct SDF Heads (non-zero init prevents Stage 1 zero-gradient deadlock)
        c_final = c1 // 2
        self.flow_head = nn.Conv2d(c_final, 2, kernel_size=3, padding=1)
        nn.init.normal_(self.flow_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.flow_head.bias)

        self.sdf_head = nn.Sequential(
            CNNBlock(c_final + 2, c1),
            ResBlock(c1),
            nn.Conv2d(c1, 1, kernel_size=3, padding=1),
        )

    @staticmethod
    def _warp_tensor(x: torch.Tensor, flow_px: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        dx_norm = flow_px[:, 0] * (2.0 / max(w - 1, 1))
        dy_norm = flow_px[:, 1] * (2.0 / max(h - 1, 1))
        grid = torch.stack([xx.unsqueeze(0) + dx_norm, yy.unsqueeze(0) + dy_norm], dim=-1)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ch0 = x[:, 0:1]  # [B, 1, 256, 256]
        ch1 = x[:, 1:2]  # [B, 1, 256, 256]

        # CNN Encoder
        f0 = self.stem(x)     # [B,  48, 128, 128]
        f1 = self.down1(f0)   # [B,  96,  64,  64]
        f2 = self.down2(f1)   # [B, 192,  32,  32]
        f3 = self.down3(f2)   # [B, 384,  16,  16]

        # Patchify -> Transformer -> Unpatchify
        b = x.size(0)
        f3_proj = self.patch_conv(f3)                                            # [B, 768, 16, 16]
        tokens = f3_proj.permute(0, 2, 3, 1).reshape(b, self.grid_h * self.grid_w, self.d_model)
        tokens = self.transformer(self.pos_enc(tokens))                          # [B, 256, 768]
        spatial = tokens.reshape(b, self.grid_h, self.grid_w, self.d_model).permute(0, 3, 1, 2)
        spatial = self.unpatch_conv(spatial)                                     # [B, 384, 16, 16]

        # High-Res CNN Decoder
        u3 = self.up3(torch.cat([spatial, f3], dim=1))   # [B, 192,  32,  32]
        u2 = self.up2(torch.cat([u3, f2], dim=1))        # [B,  96,  64,  64]
        u1 = self.up1(torch.cat([u2, f1], dim=1))        # [B,  48, 128, 128]
        u0 = self.up0(torch.cat([u1, f0], dim=1))        # [B,  24, 256, 256]

        # Smooth Flow Warp + Direct SDF Prediction
        flow_px = self.max_shift_px * torch.tanh(self.flow_head(u0))             # [B, 2, 256, 256]
        warped_bases = self._warp_tensor(torch.cat([ch0, ch1], dim=1), flow_px)  # [B, 2, 256, 256]
        pred_sdf = torch.tanh(self.sdf_head(torch.cat([u0, warped_bases], dim=1)))  # [B, 1, 256, 256]

        return pred_sdf, flow_px

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
