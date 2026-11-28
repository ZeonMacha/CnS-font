"""
v4.6 Composite Loss Functions for Dual-Dataset Direct-SDF + Uncertainty Training.

Enhancements over `GlobalGlyphLossV45` (per 审计v46_进度.md §3 & R5):
  1. Per-sample confidence / dataset weighting (`sample_weight: [B]`) so optional
     pseudo-labeled simplified-radical samples can be safely weighted relative to
     ground-truth SkipPro samples.
  2. Zero-level-set Eikonal fog-suppression loss (`fog_weight`): explicitly penalizes
     flat grey bands (`|∇φ| < 0.65 / trunc_px`) right at the zero contour (`|t| < 0.18`),
     directly addressing R5 (纟 SDF fog instability).
  3. Heteroscedastic Laplace NLL loss (`uncert_weight`) on `pred_logvar` in FP32 so the
     predicted uncertainty channel calibrates to actual boundary SDF error without
     weakening the primary SDF L1 gradient.
  4. Shared `compute_sdf_fog_ratio` metric for both validation logging and QA gating.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.v45.losses_v45 import SoftSkeletonizeV45


def compute_sdf_fog_ratio(
    sdf: torch.Tensor,
    trunc_px: float = 16.0,
    band: float = 0.15,
    min_grad_ratio: float = 0.55,
) -> torch.Tensor:
    """
    Compute per-sample fraction of zero-level-set band pixels whose gradient magnitude
    falls below `min_grad_ratio * (1 / trunc_px)`.

    Input:  sdf [B, 1, H, W] (float)
    Output: fog_ratio [B] in [0, 1]
    """
    s = sdf.float()
    sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=s.device) / 8.0
    sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=s.device) / 8.0
    gx = F.conv2d(s, sobel_x.view(1, 1, 3, 3), padding=1)
    gy = F.conv2d(s, sobel_y.view(1, 1, 3, 3), padding=1)
    gnorm = torch.sqrt(gx * gx + gy * gy + 1e-8)

    expected = 1.0 / max(trunc_px, 1.0)
    near_zero = (s.abs() < band).float()
    low_grad = (gnorm < min_grad_ratio * expected).float()
    fog_px = (near_zero * low_grad).sum(dim=(1, 2, 3))
    band_px = near_zero.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return fog_px / band_px


class GlobalGlyphLossV46(nn.Module):
    """Composite v4.6 loss with per-sample weighting, fog suppression & uncertainty calibration."""

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
        fog_weight: float = 0.45,
        curv_weight: float = 0.50,
        curv_tv_weight: float = 0.15,
        uncert_weight: float = 0.08,
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
        self.fog_weight = fog_weight
        self.curv_weight = curv_weight
        self.curv_tv_weight = curv_tv_weight
        self.uncert_weight = uncert_weight
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

    @staticmethod
    def _weighted_batch_mean(per_sample: torch.Tensor, sample_weight: Optional[torch.Tensor]) -> torch.Tensor:
        if sample_weight is None:
            return per_sample.mean()
        w = sample_weight.to(device=per_sample.device, dtype=per_sample.dtype).clamp_min(1e-4)
        return (per_sample * w).sum() / w.sum().clamp_min(1e-4)

    def forward(
        self,
        pred_sdf: torch.Tensor,
        target_sdf: torch.Tensor,
        flow_px: torch.Tensor,
        pred_logvar: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        crop = 4
        p = pred_sdf[:, :, crop:-crop, crop:-crop]
        t = target_sdf[:, :, crop:-crop, crop:-crop]
        f = flow_px[:, :, crop:-crop, crop:-crop]

        # Normalize per-sample weights to broadcast over [B, 1, H, W]
        if sample_weight is not None:
            sw = sample_weight.to(device=p.device, dtype=torch.float32).clamp_min(1e-4)
            sw_norm = sw / sw.mean().clamp_min(1e-4)
            sw4 = sw_norm.view(-1, 1, 1, 1)
        else:
            sw_norm = None
            sw4 = None

        # 1. Boundary-weighted + thin-ink/dot-boosted SDF L1
        near_edge = (t.abs() < 0.35).float()
        thin_ink = ((t > 0.0) & (t < 0.28)).float()
        weights = 1.0 + self.boundary_weight * near_edge + self.thin_ink_boost * thin_ink
        err_l1 = (p - t).abs()
        per_sdf = (weights * err_l1).mean(dim=(1, 2, 3))
        loss_sdf = self._weighted_batch_mean(per_sdf, sw_norm)

        # 2. Surface-normal gradient consistency
        pgx, pgy = self._spatial_grad(p)
        tgx, tgy = self._spatial_grad(t)
        edge_band = (t.abs() < 0.45).float() + 0.1
        per_grad = (edge_band * ((pgx - tgx).abs() + (pgy - tgy).abs())).mean(dim=(1, 2, 3))
        loss_grad = self._weighted_batch_mean(per_grad, sw_norm)

        # 3. Area-normalized straight-stem normal loss
        txx = F.conv2d(t, self.dxx.to(dtype=t.dtype), padding=1)
        tyy = F.conv2d(t, self.dyy.to(dtype=t.dtype), padding=1)
        t_curv = txx.abs() + tyy.abs()
        stem_mask = ((t.abs() < 0.30) & (t_curv < 0.16)).float()
        if sw4 is not None:
            stem_mask = stem_mask * sw4.to(dtype=stem_mask.dtype)

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
        cl_dice_per = 2.0 * tprec * tsens / (tprec + tsens + 1e-5)
        loss_cldice = 1.0 - self._weighted_batch_mean(cl_dice_per, sw_norm)

        inter = (pred_ink * targ_ink).sum(dim=(1, 2, 3))
        union = (pred_ink + targ_ink - pred_ink * targ_ink).sum(dim=(1, 2, 3)) + 1e-6
        soft_iou_per = inter / union
        soft_iou_val = self._weighted_batch_mean(soft_iou_per, sw_norm)
        loss_iou = 1.0 - soft_iou_val

        # 6. Flow TV + L2
        tv_x = (f[:, :, :, 1:] - f[:, :, :, :-1]).abs().mean()
        tv_y = (f[:, :, 1:, :] - f[:, :, :-1, :]).abs().mean()
        loss_flow = self.flow_tv_weight * (tv_x + tv_y) + self.flow_l2_weight * (f ** 2).mean()

        # 7. FP32 Narrow-band Eikonal + Zero-Level-Set Fog Penalty + Mean-Curvature
        pgx32, pgy32 = self._spatial_grad(p32)
        tgx32, tgy32 = self._spatial_grad(t32)
        p_gnorm = torch.sqrt(pgx32 * pgx32 + pgy32 * pgy32 + 1e-6)
        t_gnorm = torch.sqrt(tgx32 * tgx32 + tgy32 * tgy32 + 1e-6)
        expected_gnorm = 1.0 / max(self.trunc_px, 1.0)
        eik_mask = ((t32.abs() < 0.35) & (t_gnorm > 0.5 * expected_gnorm)).float()
        if sw4 is not None:
            eik_mask = eik_mask * sw4
        loss_eikonal = (eik_mask * ((p_gnorm - expected_gnorm) / expected_gnorm) ** 2).sum() / eik_mask.sum().clamp_min(16.0)

        # R5 Fog suppression: penalize gradient deficit near zero-level set (|∇φ| < 0.65 * expected)
        zero_band = ((t32.abs() < 0.18) & (t_gnorm > 0.5 * expected_gnorm)).float()
        if sw4 is not None:
            zero_band = zero_band * sw4
        grad_deficit = F.relu(0.65 * expected_gnorm - p_gnorm) / expected_gnorm
        loss_fog = (zero_band * (grad_deficit ** 2)).sum() / zero_band.sum().clamp_min(16.0)

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
        if sw4 is not None:
            curve_mask = curve_mask * sw4
        loss_curv = (curve_mask * (kappa_p - kappa_t).abs()).sum() / curve_mask.sum().clamp_min(16.0)

        smooth_band = ((t32.abs() < 0.30) & (t_gnorm > 0.5 * expected_gnorm) & (t_curv32 < 0.55)).float()
        if sw4 is not None:
            smooth_band = smooth_band * sw4
        cm_x = smooth_band[:, :, :, 1:] * smooth_band[:, :, :, :-1]
        cm_y = smooth_band[:, :, 1:, :] * smooth_band[:, :, :-1, :]
        tv_kx = (cm_x * (kappa_p[:, :, :, 1:] - kappa_p[:, :, :, :-1]).abs()).sum() / cm_x.sum().clamp_min(16.0)
        tv_ky = (cm_y * (kappa_p[:, :, 1:, :] - kappa_p[:, :, :-1, :]).abs()).sum() / cm_y.sum().clamp_min(16.0)
        loss_curv_tv = tv_kx + tv_ky

        # 8. Heteroscedastic Laplace NLL for predicted uncertainty (detached |p - t| so SDF L1 stays sharp)
        if pred_logvar is not None and self.uncert_weight > 0.0:
            lv = pred_logvar[:, :, crop:-crop, crop:-crop].float().clamp(-6.0, 2.0)
            abs_err_det = (p32 - t32).abs().detach().clamp_min(1e-4)
            u_mask = (t32.abs() < 0.45).float()
            if sw4 is not None:
                u_mask = u_mask * sw4
            nll = abs_err_det * torch.exp(-lv) + lv
            loss_uncert = (u_mask * nll).sum() / u_mask.sum().clamp_min(16.0)
        else:
            loss_uncert = torch.zeros((), device=p.device, dtype=torch.float32)

        total = (
            loss_sdf
            + self.grad_weight * loss_grad
            + self.stem_norm_weight * loss_stem_norm
            + self.cldice_weight * loss_cldice
            + self.iou_weight * loss_iou
            + loss_flow
            + self.eikonal_weight * loss_eikonal
            + self.fog_weight * loss_fog
            + self.curv_weight * loss_curv
            + self.curv_tv_weight * loss_curv_tv
            + self.uncert_weight * loss_uncert
        )

        metrics = {
            "sdf_l1": float(loss_sdf.detach()),
            "grad_l1": float(loss_grad.detach()),
            "stem_norm": float(loss_stem_norm.detach()),
            "cldice": float(loss_cldice.detach()),
            "soft_iou": float(soft_iou_val.detach()),
            "flow_tv": float((tv_x + tv_y).detach()),
            "eikonal": float(loss_eikonal.detach()),
            "fog_loss": float(loss_fog.detach()),
            "curv_l1": float(loss_curv.detach()),
            "curv_tv": float(loss_curv_tv.detach()),
            "uncert_nll": float(loss_uncert.detach()),
        }
        return total, metrics
