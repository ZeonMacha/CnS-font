"""
v3.0 Training Losses (`Plan3.0.md §2.3`):
  1. Target-only boundary-weighted SDF L1 + Sobel gradient loss (NO Noto dependency!).
  2. Area-normalized straight-stem surface-normal loss (labels from SkipPro ground truth).
  3. Differentiable Soft-clDice topological skeleton loss.
  4. Dead-loss monitor (< 1% change over 3 consecutive epochs triggers explicit alert).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSkeletonize(nn.Module):
    """Differentiable iterative morphological thinning for soft-clDice."""

    def __init__(self, num_iter: int = 5) -> None:
        super().__init__()
        self.num_iter = num_iter

    @staticmethod
    def _soft_erode(img: torch.Tensor) -> torch.Tensor:
        p1 = -F.max_pool2d(-img, (3, 1), stride=(1, 1), padding=(1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), stride=(1, 1), padding=(0, 1))
        return torch.minimum(p1, p2)

    @staticmethod
    def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(img, (3, 3), stride=(1, 1), padding=(1, 1))

    def _soft_open(self, img: torch.Tensor) -> torch.Tensor:
        return self._soft_dilate(self._soft_erode(img))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img1 = self._soft_open(img)
        skel = F.relu(img - img1)
        for _ in range(self.num_iter - 1):
            img = self._soft_erode(img)
            img1 = self._soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)
        return skel


class DeskinnedStrokeLossV3(nn.Module):
    """Composite v3 loss with area-normalized GT straight-stem normal alignment + Soft-clDice."""

    def __init__(
        self,
        boundary_weight: float = 8.0,
        grad_weight: float = 2.0,
        stem_norm_weight: float = 1.6,
        cldice_weight: float = 0.8,
        iou_weight: float = 0.6,
        trunc_px: float = 16.0,
    ) -> None:
        super().__init__()
        self.boundary_weight = boundary_weight
        self.grad_weight = grad_weight
        self.stem_norm_weight = stem_norm_weight
        self.cldice_weight = cldice_weight
        self.iou_weight = iou_weight
        self.trunc_px = trunc_px
        self.soft_skel = SoftSkeletonize(num_iter=5)

        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]) / 8.0
        dxx = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]])
        dyy = torch.tensor([[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]])
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
        self.register_buffer("dxx", dxx.view(1, 1, 3, 3))
        self.register_buffer("dyy", dyy.view(1, 1, 3, 3))

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
        if pred_sdf.shape[-1] >= 64:
            pred_sdf = pred_sdf[:, :, 4:-4, 4:-4]
            target_sdf = target_sdf[:, :, 4:-4, 4:-4]
            flow_px = flow_px[:, :, 4:-4, 4:-4]

        # 1. Target-only boundary weighted L1 (NEVER conditioned on reference diff channels!)
        near_edge = (target_sdf.abs() < 0.35).float()
        weights = 1.0 + self.boundary_weight * near_edge
        loss_sdf = (weights * (pred_sdf - target_sdf).abs()).mean()

        # 2. Surface-normal gradient consistency
        pgx, pgy = self._spatial_grad(pred_sdf)
        tgx, tgy = self._spatial_grad(target_sdf)
        edge_band = (target_sdf.abs() < 0.45).float() + 0.1
        loss_grad = (edge_band * ((pgx - tgx).abs() + (pgy - tgy).abs())).mean()

        # 3. Area-normalized straight-stem normal loss labeled strictly by SkipPro GT curvature
        txx = F.conv2d(target_sdf, self.dxx.to(dtype=target_sdf.dtype), padding=1)
        tyy = F.conv2d(target_sdf, self.dyy.to(dtype=target_sdf.dtype), padding=1)
        t_curv = txx.abs() + tyy.abs()
        stem_mask = ((target_sdf.abs() < 0.30) & (t_curv < 0.16)).float()

        p_norm = torch.sqrt(pgx * pgx + pgy * pgy + 1e-6)
        t_norm = torch.sqrt(tgx * tgx + tgy * tgy + 1e-6)
        pnx, pny = pgx / p_norm, pgy / p_norm
        tnx, tny = tgx / t_norm, tgy / t_norm
        stem_denom = stem_mask.sum().clamp_min(16.0)
        loss_stem_norm = (stem_mask * ((pnx - tnx).abs() + (pny - tny).abs())).sum() / stem_denom

        # 4. Soft-clDice topological loss in fp32 for numerical stability
        p32 = pred_sdf.float()
        t32 = target_sdf.float()
        pred_ink = torch.sigmoid(p32 * 16.0)
        targ_ink = torch.sigmoid(t32 * 16.0)
        s_pred = self.soft_skel(pred_ink)
        s_targ = self.soft_skel(targ_ink)
        tprec = (s_pred * targ_ink).sum(dim=(1, 2, 3)) / (s_pred.sum(dim=(1, 2, 3)) + 1e-5)
        tsens = (s_targ * pred_ink).sum(dim=(1, 2, 3)) / (s_targ.sum(dim=(1, 2, 3)) + 1e-5)
        cl_dice = (2.0 * tprec * tsens / (tprec + tsens + 1e-5)).mean()
        loss_cldice = 1.0 - cl_dice

        # 5. Soft IoU + Flow TV
        inter = (pred_ink * targ_ink).sum(dim=(1, 2, 3))
        union = (pred_ink + targ_ink - pred_ink * targ_ink).sum(dim=(1, 2, 3)) + 1e-6
        loss_iou = (1.0 - inter / union).mean()
        tv_x = (flow_px[:, :, :, 1:] - flow_px[:, :, :, :-1]).abs().mean()
        tv_y = (flow_px[:, :, 1:, :] - flow_px[:, :, :-1, :]).abs().mean()
        loss_flow = 0.08 * (tv_x + tv_y) + 0.01 * (flow_px ** 2).mean()

        total = (
            loss_sdf
            + self.grad_weight * loss_grad
            + self.stem_norm_weight * loss_stem_norm
            + self.cldice_weight * loss_cldice
            + self.iou_weight * loss_iou
            + loss_flow
        )
        return total, {
            "sdf_l1": float(loss_sdf.detach()),
            "grad_l1": float(loss_grad.detach()),
            "stem_norm": float(loss_stem_norm.detach()),
            "cldice": float(loss_cldice.detach()),
            "soft_iou": float((inter / union).mean().detach()),
            "stem_frac": float(stem_mask.mean().detach()),
        }


class DeadLossMonitor:
    """Flag any monitored loss term whose step change < 1% across 3 epochs AND total change < 3% since start."""

    def __init__(self, keys: Tuple[str, ...] = ("sdf_l1", "grad_l1", "stem_norm", "cldice")) -> None:
        self.keys = keys
        self.history: List[Dict[str, float]] = []
        self.alerts: List[str] = []

    def step(self, epoch: int, metrics: Dict[str, float]) -> List[str]:
        self.history.append({k: metrics[k] for k in self.keys if k in metrics})
        new_alerts: List[str] = []
        if len(self.history) >= 4:
            for k in self.keys:
                if k not in self.history[0]:
                    continue
                vals = [h[k] for h in self.history[-4:]]
                rels = [abs(vals[i + 1] - vals[i]) / max(abs(vals[i]), 1e-6) for i in range(3)]
                cum_rel = abs(vals[-1] - self.history[0][k]) / max(abs(self.history[0][k]), 1e-6)
                if all(r < 0.01 for r in rels) and cum_rel < 0.03:
                    msg = f"DeadLossAlert ep={epoch}: `{k}` stalled (<1%/ep and {cum_rel*100:.2f}% total: {vals[-4:]})"
                    new_alerts.append(msg)
                    self.alerts.append(msg)
        return new_alerts
