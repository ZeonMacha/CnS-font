"""
Deskinned v3.0 Local Stroke SDF Generator (`LocalStrokeSDFNetV3`, Plan 3.0 §1.2).

Input channels (4, H, W):
  [0] Noto Sans Light (`wght=300` + jitter in training) SDF in [-1, 1]
  [1] Uniform-Width Medial Skeleton SDF (`r_u=40u` + jitter in training) in [-1, 1]
  [2] Stroke Tangent Orientation cos(2θ)
  [3] Stroke Tangent Orientation sin(2θ)
Zero Noto Serif channels anywhere.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import LocalResBlock


class LocalStrokeSDFNetV3(nn.Module):
    """Restricted-ERF local Conv U-Net conditioned strictly on deskinned Light skeleton priors."""

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        max_shift_px: float = 8.0,
        delta_scale: float = 0.95,
    ) -> None:
        super().__init__()
        self.max_shift_px = max_shift_px
        self.delta_scale = delta_scale
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 3

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, c1), c1),
            nn.SiLU(inplace=True),
            LocalResBlock(c1),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, c2), c2),
            nn.SiLU(inplace=True),
            LocalResBlock(c2),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, c3), c3),
            nn.SiLU(inplace=True),
            LocalResBlock(c3),
            LocalResBlock(c3),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(c3 + c2, c2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, c2), c2),
            nn.SiLU(inplace=True),
            LocalResBlock(c2),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(c2 + c1, c1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, c1), c1),
            nn.SiLU(inplace=True),
            LocalResBlock(c1),
        )

        self.flow_head = nn.Conv2d(c1, 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

        self.refine_head = nn.Sequential(
            nn.Conv2d(c1 + 2, c1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, c1), c1),
            nn.SiLU(inplace=True),
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ch0 = x[:, 0:1]
        ch1 = x[:, 1:2]

        f1 = self.stem(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)

        u2 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, f2], dim=1))

        u1 = F.interpolate(u2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, f1], dim=1))

        flow_px = self.max_shift_px * torch.tanh(self.flow_head(u1))
        warped_bases = self._warp_tensor(torch.cat([ch0, ch1], dim=1), flow_px)

        head_out = self.refine_head(torch.cat([u1, warped_bases], dim=1))
        alpha_uni = torch.sigmoid(head_out[:, 0:1])
        delta_sdf = self.delta_scale * torch.tanh(head_out[:, 1:2])

        base_blend = (1.0 - alpha_uni) * warped_bases[:, 0:1] + alpha_uni * warped_bases[:, 1:2]
        union_sdf = torch.maximum(ch0, ch1)
        active_gate = torch.sigmoid((union_sdf + 0.96) * 25.0)
        pred_sdf = torch.clamp(base_blend + active_gate * delta_sdf, -1.0, 1.0)
        return pred_sdf, flow_px
