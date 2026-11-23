"""
v4.0 Global-Context CNN-Transformer SDF Generator (`GlobalGlyphTransformer`).

Architecture (v4-lite: 4-layer Transformer, ~30M params):
  Input: (B, 4, 256, 256) deskinned Light channels [sdf_light, skel_sdf, cos2t, sin2t]

  CNN Encoder (4-stage downsample, 256->128->64->32->16):
    Stem:  Conv(4->48, 7x7, s2)  + ResBlock    -> 48@128
    Down1: Conv(48->96, 3x3, s2) + ResBlock    -> 96@64
    Down2: Conv(96->192, 3x3, s2)+ ResBlock    -> 192@32
    Down3: Conv(192->384, 3x3, s2)+ ResBlock   -> 384@16

  Patchify (ViT-style, 1x1 conv to avoid massive linear layers):
    Conv1x1(384->768) on 384@16x16 -> 768@16x16
    Reshape: (B, 768, 16, 16) -> (B, 256, 768)  [256 patches, each 768-dim]
    + 2D Sincos Positional Encoding

  Transformer Encoder: 4x [PreNorm MultiHeadSelfAttention(12 heads, d=768) + FFN(768->3072->768)]

  Unpatchify + CNN Decoder (4-stage upsample with skip connections):
    Reshape: (B, 256, 768) -> (B, 768, 16, 16)
    Conv1x1(768->384) -> 384@16
    Up3: Conv(384+384->192, 3x3) + Upsample2x + ResBlock  -> 192@32  (skip from Down3)
    Up2: Conv(192+192->96, 3x3)  + Upsample2x + ResBlock  -> 96@64   (skip from Down2)
    Up1: Conv(96+96->48, 3x3)    + Upsample2x + ResBlock  -> 48@128  (skip from Down1)
    Up0: Conv(48->24, 3x3, s2) + Upsample4x                -> 24@256

  SDF Heads (same warp+blend+gate logic as v3):
    FlowHead:   Conv(24->2, 3x3) -> tanh * max_shift
    RefineHead: Conv(26->48, 3x3) + GN + SiLU + Conv(48->2, 3x3) -> [alpha, delta_sdf]
    pred_sdf = (1-alpha)*warped_light + alpha*warped_skel + gate*delta_sdf
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
    """
    2D Sincos positional encoding for patch tokens. Fixed (non-learnable).
    For grid position (row, col), generates d_model-dim vector via sin/cos at multiple frequencies.
    """

    def __init__(self, grid_h: int, grid_w: int, d_model: int) -> None:
        super().__init__()
        half = d_model // 4
        freqs = torch.exp(torch.arange(half).float() * (-math.log(10000.0) / max(half, 1)))

        rows = torch.arange(grid_h).float()
        cols = torch.arange(grid_w).float()
        grid_row, grid_col = torch.meshgrid(rows, cols, indexing="ij")

        r_flat = grid_row.flatten()
        c_flat = grid_col.flatten()

        r_sin = torch.sin(r_flat.unsqueeze(1) * freqs.unsqueeze(0))
        r_cos = torch.cos(r_flat.unsqueeze(1) * freqs.unsqueeze(0))
        c_sin = torch.sin(c_flat.unsqueeze(1) * freqs.unsqueeze(0))
        c_cos = torch.cos(c_flat.unsqueeze(1) * freqs.unsqueeze(0))

        pe = torch.cat([r_sin, r_cos, c_sin, c_cos], dim=-1)
        self.register_buffer("pos_embed", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed[:, : x.size(1), :]


class GlobalGlyphTransformer(nn.Module):
    """
    Hybrid CNN-Encoder + Transformer + CNN-Decoder for full-glyph SDF generation.

    Input:  (B, 4, 256, 256)
    Output: (pred_sdf: (B, 1, 256, 256), flow_px: (B, 2, 256, 256))
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
        max_shift_px: float = 8.0,
        delta_scale: float = 0.95,
        decoder_version: str = "v41",
    ) -> None:
        super().__init__()
        self.canvas_size = canvas_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.max_shift_px = max_shift_px
        self.delta_scale = delta_scale
        self.decoder_version = decoder_version

        self.grid_h = canvas_size // patch_size
        self.grid_w = canvas_size // patch_size
        n_patches = self.grid_h * self.grid_w

        c1, c2, c3, c4 = 48, 96, 192, 384

        # CNN Encoder
        self.stem = nn.Sequential(
            CNNBlock(in_channels, c1, kernel_size=7, stride=2),
            ResBlock(c1),
        )
        self.down1 = nn.Sequential(
            CNNBlock(c1, c2, kernel_size=3, stride=2),
            ResBlock(c2),
        )
        self.down2 = nn.Sequential(
            CNNBlock(c2, c3, kernel_size=3, stride=2),
            ResBlock(c3),
        )
        self.down3 = nn.Sequential(
            CNNBlock(c3, c4, kernel_size=3, stride=2),
            ResBlock(c4),
        )

        # Patchify: 1x1 conv (384 -> d_model) on 384@16x16, then reshape
        self.patch_conv = nn.Conv2d(c4, d_model, kernel_size=1, bias=False)
        self.pos_enc = SincosPositionalEncoding(self.grid_h, self.grid_w, d_model)

        # Transformer Encoder
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

        # Unpatchify: 1x1 conv (d_model -> c4) on d_model@16x16, then reshape
        self.unpatch_conv = nn.Conv2d(d_model, c4, kernel_size=1, bias=False)

        # CNN Decoder with skip connections
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
        if decoder_version == "v40":
            # Legacy v4.0 decoder (kept only for loading legacy v4_zh_sdf_global.pt)
            self.up0 = nn.Sequential(
                nn.Conv2d(c1, c1 // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(min(8, c1 // 2), c1 // 2),
                nn.SiLU(inplace=True),
                nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            )
        else:
            # v4.1 high-resolution decoder: fuse f0 (128x128) skip, upsample 128->256, refine at full 256x256
            self.up0 = nn.Sequential(
                CNNBlock(c1 * 2, c1 // 2),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ResBlock(c1 // 2),
            )

        # SDF Heads
        c_final = c1 // 2
        self.flow_head = nn.Conv2d(c_final, 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

        self.refine_head = nn.Sequential(
            CNNBlock(c_final + 2, c1),
            nn.Conv2d(c1, 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.refine_head[-1].weight)
        nn.init.zeros_(self.refine_head[-1].bias)

    @staticmethod
    def _warp_tensor(x: torch.Tensor, flow_px: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        dx_norm = flow_px[:, 0] * (2.0 / max(w - 1, 1))
        dy_norm = flow_px[:, 1] * (2.0 / max(h - 1, 1))
        grid = torch.stack([xx.unsqueeze(0) + dx_norm, yy.unsqueeze(0) + dy_norm], dim=-1)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, H*W, C). Input is already at patch-grid resolution."""
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)   # (B, H, W, C)
        x = x.reshape(b, h * w, c)   # (B, N, C)
        return x

    def _from_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, C) -> (B, C, H, W)."""
        b, n, c = x.shape
        x = x.reshape(b, self.grid_h, self.grid_w, c)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)                        # (B, C, H, W)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ch0 = x[:, 0:1]
        ch1 = x[:, 1:2]

        # CNN Encoder
        f0 = self.stem(x)
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        f3 = self.down3(f2)  # (B, 384, 16, 16)

        # Patchify: 1x1 conv + reshape to tokens
        f3_proj = self.patch_conv(f3)  # (B, 768, 16, 16)
        tokens = self._to_tokens(f3_proj)  # (B, 256, 768)
        tokens = self.pos_enc(tokens)
        tokens = self.transformer(tokens)  # (B, 256, 768)

        # Unpatchify: reshape + 1x1 conv
        spatial = self._from_tokens(tokens)  # (B, 768, 16, 16)
        spatial = self.unpatch_conv(spatial)  # (B, 384, 16, 16)

        # CNN Decoder
        u3 = self.up3(torch.cat([spatial, f3], dim=1))  # (B, 192, 32, 32)
        u2 = self.up2(torch.cat([u3, f2], dim=1))       # (B, 96, 64, 64)
        u1 = self.up1(torch.cat([u2, f1], dim=1))       # (B, 48, 128, 128)
        if self.decoder_version == "v40":
            u0 = self.up0(u1)                           # (B, 24, 256, 256) legacy
        else:
            u0 = self.up0(torch.cat([u1, f0], dim=1))   # (B, 24, 256, 256) v4.1 with 128x128 skip

        # SDF Heads
        flow_px = self.max_shift_px * torch.tanh(self.flow_head(u0))
        warped_bases = self._warp_tensor(torch.cat([ch0, ch1], dim=1), flow_px)

        head_out = self.refine_head(torch.cat([u0, warped_bases], dim=1))
        alpha_uni = torch.sigmoid(head_out[:, 0:1])
        delta_sdf = self.delta_scale * torch.tanh(head_out[:, 1:2])

        base_blend = (1.0 - alpha_uni) * warped_bases[:, 0:1] + alpha_uni * warped_bases[:, 1:2]
        union_sdf = torch.maximum(ch0, ch1)
        active_gate = torch.sigmoid((union_sdf + 0.96) * 25.0)
        pred_sdf = torch.clamp(base_blend + active_gate * delta_sdf, -1.0, 1.0)

        return pred_sdf, flow_px

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
