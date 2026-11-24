"""
v4.5 Composite Loss Functions for Global-Context Direct-SDF Training.

Fixes over v4.0-v4.3 (from 审计v4.md):
  1. Re-calibrated flow regularization (`flow_tv_weight=0.06`, `flow_l2_weight=0.005`)
     to prevent v4.3's 33x flow_tv oscillation while supporting `max_shift_px=14.0`.
  2. Small-stroke / dot foreground preservation boost (`+2.0x` on thin interior ink `0 < t < 0.25`)
     so small dots (e.g. `义` top dot, `孓` diagonal) are never erased.
  3. FP32 narrow-band Eikonal + Curvature L1 + Curvature TV smoothness.
  4. Dead `OCRClassifierHead` removed.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSkeletonizeV45(nn.Module):
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


class GlobalGlyphLossV45(nn.Module):
    """Composite v4.5 loss for 256x256 full-glyph direct-SDF training."""

    def __init__(
        self,
        boundary_weight: float = 8.0,
        thin_ink_boost: float = 2.5,
        grad_weight: float = 2.0,
        stem_norm_weight: float = 1.6,
        cldice_weight: float = 0.85,
        iou_weight: float = 0.6,
        flow_tv_weight: float = 0.06,
        flow_l2_weight: float = 0.005,
        eikonal_weight: float = 0.25,
        curv_weight: float = 0.50,
        curv_tv_weight: float = 0.15,
        trunc_px: float = 16.0,
    ) -> None:
        super().__init__()
        self.boundary_weight = boundary_weight
        self.thin_ink_boost = thin_ink_boost
        self.grad_weight = grad_weight
        self.stem_norm_weight = stem_norm_weight
        self.cldice_weight = cldice_weight
        self.iou_weight = iou_weight
        self.flow_tv_weight = flow_tv_weight
        self.flow_l2_weight = flow_l2_weight
        self.eikonal_weight = eikonal_weight
        self.curv_weight = curv_weight
        self.curv_tv_weight = curv_tv_weight
        self.trunc_px = trunc_px
        self.soft_skel = SoftSkeletonizeV45(num_iter=5)

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
        flow_px: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        crop = 4
        p = pred_sdf[:, :, crop:-crop, crop:-crop]
        t = target_sdf[:, :, crop:-crop, crop:-crop]
        f = flow_px[:, :, crop:-crop, crop:-crop]

        # 1. Boundary-weighted + thin-ink/dot-boosted SDF L1
        near_edge = (t.abs() < 0.35).float()
        thin_ink = ((t > 0.0) & (t < 0.28)).float()
        weights = 1.0 + self.boundary_weight * near_edge + self.thin_ink_boost * thin_ink
        loss_sdf = (weights * (p - t).abs()).mean()

        # 2. Surface-normal gradient consistency
        pgx, pgy = self._spatial_grad(p)
        tgx, tgy = self._spatial_grad(t)
        edge_band = (t.abs() < 0.45).float() + 0.1
        loss_grad = (edge_band * ((pgx - tgx).abs() + (pgy - tgy).abs())).mean()

        # 3. Area-normalized straight-stem normal loss
        txx = F.conv2d(t, self.dxx.to(dtype=t.dtype), padding=1)
        tyy = F.conv2d(t, self.dyy.to(dtype=t.dtype), padding=1)
        t_curv = txx.abs() + tyy.abs()
        stem_mask = ((t.abs() < 0.30) & (t_curv < 0.16)).float()

        p_norm = torch.sqrt(pgx * pgx + pgy * pgy + 1e-6)
        t_norm = torch.sqrt(tgx * tgx + tgy * tgy + 1e-6)
        pnx, pny = pgx / p_norm, pgy / p_norm
        tnx, tny = tgx / t_norm, tgy / t_norm
        loss_stem_norm = (stem_mask * ((pnx - tnx).abs() + (pny - tny).abs())).sum() / stem_mask.sum().clamp_min(16.0)

        # 4. Soft-clDice & 5. Soft IoU (FP32)
        p32 = p.float()
        t32 = t.float()
        pred_ink = torch.sigmoid(p32 * 16.0)
        targ_ink = torch.sigmoid(t32 * 16.0)
        s_pred = self.soft_skel(pred_ink)
        s_targ = self.soft_skel(targ_ink)
        tprec = (s_pred * targ_ink).sum(dim=(1, 2, 3)) / (s_pred.sum(dim=(1, 2, 3)) + 1e-5)
        tsens = (s_targ * pred_ink).sum(dim=(1, 2, 3)) / (s_targ.sum(dim=(1, 2, 3)) + 1e-5)
        cl_dice = (2.0 * tprec * tsens / (tprec + tsens + 1e-5)).mean()
        loss_cldice = 1.0 - cl_dice

        inter = (pred_ink * targ_ink).sum(dim=(1, 2, 3))
        union = (pred_ink + targ_ink - pred_ink * targ_ink).sum(dim=(1, 2, 3)) + 1e-6
        soft_iou_val = (inter / union).mean()
        loss_iou = 1.0 - soft_iou_val

        # 6. Re-calibrated Flow TV + L2
        tv_x = (f[:, :, :, 1:] - f[:, :, :, :-1]).abs().mean()
        tv_y = (f[:, :, 1:, :] - f[:, :, :-1, :]).abs().mean()
        loss_flow = self.flow_tv_weight * (tv_x + tv_y) + self.flow_l2_weight * (f ** 2).mean()

        # 7. FP32 Narrow-band Eikonal & Mean-Curvature Alignment + Curvature TV
        pgx32, pgy32 = self._spatial_grad(p32)
        tgx32, tgy32 = self._spatial_grad(t32)
        p_gnorm = torch.sqrt(pgx32 * pgx32 + pgy32 * pgy32 + 1e-6)
        t_gnorm = torch.sqrt(tgx32 * tgx32 + tgy32 * tgy32 + 1e-6)
        expected_gnorm = 1.0 / max(self.trunc_px, 1.0)
        eik_mask = ((t32.abs() < 0.35) & (t_gnorm > 0.5 * expected_gnorm)).float()
        loss_eikonal = (eik_mask * ((p_gnorm - expected_gnorm) / expected_gnorm) ** 2).sum() / eik_mask.sum().clamp_min(16.0)

        pnx32, pny32 = pgx32 / p_gnorm, pgy32 / p_gnorm
        tnx32, tny32 = tgx32 / t_gnorm, tgy32 / t_gnorm
        dpnx_dx, _ = self._spatial_grad(pnx32)
        _, dpny_dy = self._spatial_grad(pny32)
        dtnx_dx, _ = self._spatial_grad(tnx32)
        _, dtny_dy = self._spatial_grad(tny32)
        kappa_p = (dpnx_dx + dpny_dy).clamp(-2.0, 2.0)
        kappa_t = (dtnx_dx + dtny_dy).clamp(-2.0, 2.0)

        t_curv32 = t_curv.float()
        curve_mask = ((t32.abs() < 0.30) & (t_gnorm > 0.5 * expected_gnorm) & (t_curv32 >= 0.04) & (t_curv32 < 0.55)).float()
        loss_curv = (curve_mask * (kappa_p - kappa_t).abs()).sum() / curve_mask.sum().clamp_min(16.0)

        smooth_band = ((t32.abs() < 0.30) & (t_gnorm > 0.5 * expected_gnorm) & (t_curv32 < 0.55)).float()
        cm_x = smooth_band[:, :, :, 1:] * smooth_band[:, :, :, :-1]
        cm_y = smooth_band[:, :, 1:, :] * smooth_band[:, :, :-1, :]
        tv_kx = (cm_x * (kappa_p[:, :, :, 1:] - kappa_p[:, :, :, :-1]).abs()).sum() / cm_x.sum().clamp_min(16.0)
        tv_ky = (cm_y * (kappa_p[:, :, 1:, :] - kappa_p[:, :, :-1, :]).abs()).sum() / cm_y.sum().clamp_min(16.0)
        loss_curv_tv = tv_kx + tv_ky

        total = (
            loss_sdf
            + self.grad_weight * loss_grad
            + self.stem_norm_weight * loss_stem_norm
            + self.cldice_weight * loss_cldice
            + self.iou_weight * loss_iou
            + loss_flow
            + self.eikonal_weight * loss_eikonal
            + self.curv_weight * loss_curv
            + self.curv_tv_weight * loss_curv_tv
        )

        metrics = {
            "sdf_l1": float(loss_sdf.detach()),
            "grad_l1": float(loss_grad.detach()),
            "stem_norm": float(loss_stem_norm.detach()),
            "cldice": float(loss_cldice.detach()),
            "soft_iou": float(soft_iou_val.detach()),
            "flow_tv": float((tv_x + tv_y).detach()),
            "eikonal": float(loss_eikonal.detach()),
            "curv_l1": float(loss_curv.detach()),
            "curv_tv": float(loss_curv_tv.detach()),
        }
        return total, metrics
