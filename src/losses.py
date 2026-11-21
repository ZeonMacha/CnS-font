"""
Boundary-Focused SDF, Surface-Normal Gradient, Soft-Contour IoU, and Flow Smoothness Losses.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StrokeSDFLoss(nn.Module):
    """Composite loss focused tightly on the zero-level-set vector contour (SDF == 0)."""

    def __init__(
        self,
        boundary_weight: float = 6.0,
        grad_weight: float = 1.5,
        iou_weight: float = 0.5,
        flow_tv_weight: float = 0.08,
        flow_mag_weight: float = 0.01,
        trunc_px: float = 16.0,
    ) -> None:
        super().__init__()
        self.boundary_weight = boundary_weight
        self.grad_weight = grad_weight
        self.iou_weight = iou_weight
        self.flow_tv_weight = flow_tv_weight
        self.flow_mag_weight = flow_mag_weight
        self.trunc_px = trunc_px

        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]) / 8.0
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _spatial_grad(self, sdf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gx = F.conv2d(sdf, self.sobel_x.to(dtype=sdf.dtype), padding=1)
        gy = F.conv2d(sdf, self.sobel_y.to(dtype=sdf.dtype), padding=1)
        return gx, gy

    def forward(
        self,
        pred_sdf: torch.Tensor,
        target_sdf: torch.Tensor,
        cond_x: torch.Tensor,
        flow_px: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Crop 4px border on training patches to eliminate any Conv zero-padding border effect
        if pred_sdf.shape[-1] >= 64:
            pred_sdf = pred_sdf[:, :, 4:-4, 4:-4]
            target_sdf = target_sdf[:, :, 4:-4, 4:-4]
            cond_x = cond_x[:, :, 4:-4, 4:-4]
            flow_px = flow_px[:, :, 4:-4, 4:-4]
        # 1. Emphasize narrow band around target boundary AND Serif-vs-Sans serif diff zones
        near_edge = ((target_sdf.abs() < 0.35) | (cond_x[:, 2:3].abs() > 0.08)).float()
        weights = 1.0 + self.boundary_weight * near_edge
        loss_sdf = (weights * (pred_sdf - target_sdf).abs()).mean()

        # 2. Sobel surface-normal consistency (preserves crisp 90-deg corners & sharp hook tips)
        pgx, pgy = self._spatial_grad(pred_sdf)
        tgx, tgy = self._spatial_grad(target_sdf)
        edge_band = (target_sdf.abs() < 0.45).float() + 0.1
        loss_grad = (edge_band * ((pgx - tgx).abs() + (pgy - tgy).abs())).mean()

        # 3. Sub-pixel differentiable soft-ink Dice/IoU loss around SDF = 0
        sharpness = self.trunc_px * 2.0
        pred_ink = torch.sigmoid(pred_sdf * sharpness)
        targ_ink = torch.sigmoid(target_sdf * sharpness)
        inter = (pred_ink * targ_ink).sum(dim=(1, 2, 3))
        union = (pred_ink + targ_ink - pred_ink * targ_ink).sum(dim=(1, 2, 3)) + 1e-6
        loss_iou = (1.0 - inter / union).mean()

        # 4. Smoothness & magnitude regularization on local deformation field
        tv_x = (flow_px[:, :, :, 1:] - flow_px[:, :, :, :-1]).abs().mean()
        tv_y = (flow_px[:, :, 1:, :] - flow_px[:, :, :-1, :]).abs().mean()
        loss_tv = tv_x + tv_y
        loss_mag = (flow_px ** 2).mean()

        total = (
            loss_sdf
            + self.grad_weight * loss_grad
            + self.iou_weight * loss_iou
            + self.flow_tv_weight * loss_tv
            + self.flow_mag_weight * loss_mag
        )
        metrics = {
            "total": float(total.detach()),
            "sdf_l1": float(loss_sdf.detach()),
            "grad_l1": float(loss_grad.detach()),
            "soft_iou": float((inter / union).mean().detach()),
        }
        return total, metrics
