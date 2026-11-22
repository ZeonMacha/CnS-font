"""
v4.0 Composite Loss Functions for Global-Context SDF Training.

Loss terms:
  1. Boundary-weighted SDF L1 (target-only, no reference channels)
  2. Sobel gradient consistency
  3. Area-normalized straight-stem normal loss (GT-labeled)
  4. Soft-clDice topological skeleton loss
  5. Soft IoU
  6. Flow TV regularization
  7. [Optional] OCR auxiliary classification loss (cross-entropy, weight 0.1)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSkeletonizeV4(nn.Module):
    """Differentiable iterative morphological thinning for soft-clDice (same as v3)."""

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


class GlobalGlyphLossV4(nn.Module):
    """
    Composite v4 loss for full-glyph SDF training.

    Same core terms as v3 (DeskinnedStrokeLossV3) but designed for
    256×256 full-glyph inputs. Includes optional OCR auxiliary loss.
    """

    def __init__(
        self,
        boundary_weight: float = 8.0,
        grad_weight: float = 2.0,
        stem_norm_weight: float = 1.6,
        cldice_weight: float = 0.8,
        iou_weight: float = 0.6,
        flow_tv_weight: float = 0.08,
        flow_l2_weight: float = 0.01,
        trunc_px: float = 16.0,
    ) -> None:
        super().__init__()
        self.boundary_weight = boundary_weight
        self.grad_weight = grad_weight
        self.stem_norm_weight = stem_norm_weight
        self.cldice_weight = cldice_weight
        self.iou_weight = iou_weight
        self.flow_tv_weight = flow_tv_weight
        self.flow_l2_weight = flow_l2_weight
        self.trunc_px = trunc_px
        self.soft_skel = SoftSkeletonizeV4(num_iter=5)

        # Sobel kernels
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
        ocr_logits: Optional[torch.Tensor] = None,
        ocr_targets: Optional[torch.Tensor] = None,
        ocr_weight: float = 0.1,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred_sdf: (B, 1, H, W) predicted SDF
            target_sdf: (B, 1, H, W) target SDF
            flow_px: (B, 2, H, W) flow field
            ocr_logits: (B, n_classes) optional OCR classification logits
            ocr_targets: (B,) optional OCR classification targets
            ocr_weight: weight for OCR auxiliary loss

        Returns:
            total_loss: scalar
            metrics: dict of individual loss components
        """
        H, W = pred_sdf.shape[-2:]
        crop = 4  # Trim 4px border for gradient computations

        p = pred_sdf[:, :, crop:-crop, crop:-crop]
        t = target_sdf[:, :, crop:-crop, crop:-crop]
        f = flow_px[:, :, crop:-crop, crop:-crop]

        # 1. Boundary-weighted SDF L1
        near_edge = (t.abs() < 0.35).float()
        weights = 1.0 + self.boundary_weight * near_edge
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
        stem_denom = stem_mask.sum().clamp_min(16.0)
        loss_stem_norm = (stem_mask * ((pnx - tnx).abs() + (pny - tny).abs())).sum() / stem_denom

        # 4. Soft-clDice topological loss (fp32 for stability)
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

        # 5. Soft IoU
        inter = (pred_ink * targ_ink).sum(dim=(1, 2, 3))
        union = (pred_ink + targ_ink - pred_ink * targ_ink).sum(dim=(1, 2, 3)) + 1e-6
        soft_iou_val = (inter / union).mean()
        loss_iou = 1.0 - soft_iou_val

        # 6. Flow TV + L2
        tv_x = (f[:, :, :, 1:] - f[:, :, :, :-1]).abs().mean()
        tv_y = (f[:, :, 1:, :] - f[:, :, :-1, :]).abs().mean()
        loss_flow = self.flow_tv_weight * (tv_x + tv_y) + self.flow_l2_weight * (f ** 2).mean()

        # 7. [Optional] OCR auxiliary loss
        loss_ocr = torch.tensor(0.0, device=p.device)
        if ocr_logits is not None and ocr_targets is not None:
            loss_ocr = F.cross_entropy(ocr_logits, ocr_targets)

        total = (
            loss_sdf
            + self.grad_weight * loss_grad
            + self.stem_norm_weight * loss_stem_norm
            + self.cldice_weight * loss_cldice
            + self.iou_weight * loss_iou
            + loss_flow
            + ocr_weight * loss_ocr
        )

        metrics = {
            "sdf_l1": float(loss_sdf.detach()),
            "grad_l1": float(loss_grad.detach()),
            "stem_norm": float(loss_stem_norm.detach()),
            "cldice": float(loss_cldice.detach()),
            "soft_iou": float(soft_iou_val.detach()),
            "flow_tv": float((tv_x + tv_y).detach()),
            "stem_frac": float(stem_mask.mean().detach()),
        }
        if ocr_logits is not None:
            metrics["ocr_loss"] = float(loss_ocr.detach())

        return total, metrics


class OCRClassifierHead(nn.Module):
    """
    Optional OCR auxiliary head attached to the Transformer encoder output.
    Global Average Pool → MLP → n_classes softmax.

    Trained with cross-entropy loss as an auxiliary supervision signal.
    The OCR classifier runs on the 3060 (not the 4090) during inference.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_classes: int = 8105,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, d_model, 1, 1) — applied to spatial features
            nn.Flatten(),             # (B, d_model)
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, tokens: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        """
        Args:
            tokens: (B, N_patches, d_model) Transformer output
            grid_h: spatial height of patch grid
            grid_w: spatial width of patch grid

        Returns:
            logits: (B, n_classes)
        """
        b, n, d = tokens.shape
        # Reshape to spatial: (B, d, grid_h, grid_w)
        spatial = tokens.permute(0, 2, 1).reshape(b, d, grid_h, grid_w)
        return self.net(spatial)
