"""
v2.0 Hessian-Straightness-Regularized Local Stroke SDF Fallback Generator.

Fixes BOTH root causes of v1 neural generation defects:
  1. Uses `uharfbuzz` vector contours (`language="ja"` for training on SkipPro-B,
     `language="zh-Hans"` for inference!) -> ZERO Windows `raqm=False` JP fallback!
  2. Trains WITHOUT `PatchGAN` (no high-frequency edge ripples) and WITH explicit
     Second-Derivative Hessian Straightness Loss along straight stems + Total Variation
     of ∇SDF, followed by `optimize_synthesized_contours` (0 mid-stem nodes!).
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.glyph_renderer import CanvasSpec
from src.losses import StrokeSDFLoss
from src.model import LocalStrokeSDFNet
from src.sdf_utils import mask_to_normalized_sdf
from src.v2.curve_optimizer import optimize_synthesized_contours
from src.v2.hb_outlines import (
    DrawCmd,
    HBOutlineExtractor,
    SkipProOutlineExtractor,
    rasterize_cmds,
)
from src.vectorizer import sdf_to_cff_commands


def cmds_to_dual_sdf_input(
    cmds_serif: List[DrawCmd],
    cmds_sans: List[DrawCmd],
    canvas_size: int = 256,
    trunc_px: float = 16.0,
) -> np.ndarray:
    """
    Build exact platform-independent 4-channel SDF input from HarfBuzz vector contours:
      [0] Serif Bold SDF
      [1] Sans SemiBold SDF
      [2] (Serif - Sans) difference map (triangular serifs)
      [3] Internal stroke-depth map
    """
    s_hi = canvas_size * 2
    mask_sr = rasterize_cmds(cmds_serif, canvas_size=s_hi, padding=96.0)
    mask_sn = rasterize_cmds(cmds_sans, canvas_size=s_hi, padding=96.0)

    sdf_sr = mask_to_normalized_sdf(mask_sr, canvas_size, trunc_px=trunc_px)
    sdf_sn = mask_to_normalized_sdf(mask_sn, canvas_size, trunc_px=trunc_px)
    sdf_diff = np.clip((sdf_sr - sdf_sn) * 2.0, -1.0, 1.0)
    depth = np.clip(sdf_sr, 0.0, 1.0) * 2.0 - 1.0
    return np.stack([sdf_sr, sdf_sn, sdf_diff, depth], axis=0).astype(np.float32)


class HessianStraightnessLoss(nn.Module):
    """
    Penalizes second-derivative ripples (∂²/∂x², ∂²/∂y²) and gradient roughness on
    straight stem bands where Noto Sans & Serif agree (plan2.0.md §1 #5).
    """

    def __init__(self, trunc_px: float = 16.0) -> None:
        super().__init__()
        self.base = StrokeSDFLoss(boundary_weight=10.0, grad_weight=2.2, iou_weight=0.8, trunc_px=trunc_px)
        dxx = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]])
        dyy = torch.tensor([[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]])
        self.register_buffer("dxx", dxx.view(1, 1, 3, 3))
        self.register_buffer("dyy", dyy.view(1, 1, 3, 3))

    def forward(
        self,
        pred_sdf: torch.Tensor,
        target_sdf: torch.Tensor,
        cond_x: torch.Tensor,
        flow_px: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        loss_base, m = self.base(pred_sdf, target_sdf, cond_x, flow_px)

        p = pred_sdf[:, :, 4:-4, 4:-4] if pred_sdf.shape[-1] >= 64 else pred_sdf
        t = target_sdf[:, :, 4:-4, 4:-4] if target_sdf.shape[-1] >= 64 else target_sdf
        cx = cond_x[:, :, 4:-4, 4:-4] if cond_x.shape[-1] >= 64 else cond_x

        # Smoothness along straight stem edge bands (low serif-vs-sans discrepancy + near boundary)
        stem_band = ((t.abs() < 0.35) & (cx[:, 2:3].abs() < 0.12)).float()
        pxx = F.conv2d(p, self.dxx.to(dtype=p.dtype), padding=1)
        pyy = F.conv2d(p, self.dyy.to(dtype=p.dtype), padding=1)
        txx = F.conv2d(t, self.dxx.to(dtype=t.dtype), padding=1)
        tyy = F.conv2d(t, self.dyy.to(dtype=t.dtype), padding=1)
        loss_hess = (stem_band * ((pxx - txx).abs() + (pyy - tyy).abs())).mean()

        total = loss_base + 1.8 * loss_hess
        m["hess"] = float(loss_hess.detach())
        return total, m


def train_zh_hessian_sdf_model(
    skip_otf: Path,
    noto_serif: Path,
    noto_sans: Path,
    ckpt_out: Path,
    cache_dir: Path,
    canvas_size: int = 256,
    patch_size: int = 96,
    base_channels: int = 64,
    epochs: int = 16,
    batch_size: int = 64,
    lr: float = 1.2e-3,
) -> Path:
    """
    Build HarfBuzz-exact `ja` SDF training cache (no raqm dependency!) and train
    Hessian-regularized `LocalStrokeSDFNet` (no PatchGAN ripples) on CUDA GPU.
    """
    if ckpt_out.exists():
        print(f"[v2-SDF] Reusing existing checkpoint -> {ckpt_out}", flush=True)
        return ckpt_out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    trunc_px = 16.0 * (canvas_size / 256.0)
    max_shift_px = 6.0 * (canvas_size / 256.0)

    skip_ext = SkipProOutlineExtractor(skip_otf)
    train_chars = sorted(chr(cp) for cp in skip_ext.cmap if 0x4E00 <= cp <= 0x9FFF)
    # Strictly hold out test benchmark chars from training
    holdout = set("志洳菜嘉改栅畴逡红语银饭伞丌习乡卫专书东言直令骨微道")
    train_chars = [c for c in train_chars if c not in holdout]
    skip_ext.close()

    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"v2_hb_n{len(train_chars)}_s{canvas_size}"
    in_p = cache_dir / f"inputs_{tag}.npy"
    tg_p = cache_dir / f"targets_{tag}.npy"
    ec_p = cache_dir / f"edges_{tag}.npz"

    if in_p.exists() and tg_p.exists() and ec_p.exists():
        print(f"[v2-SDF] Loading HarfBuzz SDF cache from {cache_dir}...", flush=True)
        inputs_i8 = np.load(in_p)
        targets_i8 = np.load(tg_p)
        ec_npz = np.load(ec_p)
        edge_coords = [ec_npz[f"c{i}"] for i in range(len(train_chars))]
    else:
        print(f"[v2-SDF] Building HarfBuzz-exact SDF cache for {len(train_chars)} chars...", flush=True)
        n = len(train_chars)
        inputs_i8 = np.empty((n, 4, canvas_size, canvas_size), dtype=np.int8)
        targets_i8 = np.empty((n, 1, canvas_size, canvas_size), dtype=np.int8)
        edge_coords = [np.empty((1, 2), dtype=np.int16)] * n
        half = patch_size // 2
        tls = threading.local()

        def _proc(item: Tuple[int, str]):
            i, ch = item
            if not hasattr(tls, "sr"):
                tls.sr = HBOutlineExtractor(noto_serif, ttc_index=0, weight=700)
                tls.sn = HBOutlineExtractor(noto_sans, ttc_index=0, weight=620)
                tls.sk = SkipProOutlineExtractor(skip_otf)
            _, c_sr = tls.sr.get_gid_and_cmds(ch, "ja")
            _, c_sn = tls.sn.get_gid_and_cmds(ch, "ja")
            c_sk = tls.sk.get_cmds(ch)
            x_f = cmds_to_dual_sdf_input(c_sr, c_sn, canvas_size=canvas_size, trunc_px=trunc_px)
            m_sk = rasterize_cmds(c_sk, canvas_size=canvas_size * 2, padding=96.0)
            y_f = mask_to_normalized_sdf(m_sk, canvas_size, trunc_px=trunc_px)[None, :, :]

            ys, xs = np.where(np.abs(y_f[0]) < 0.42)
            if len(ys) > 0:
                coords = np.stack([np.clip(ys, half, canvas_size - half), np.clip(xs, half, canvas_size - half)], axis=1).astype(np.int16)
                if len(coords) > 220:
                    coords = coords[np.random.choice(len(coords), 220, replace=False)]
            else:
                coords = np.array([[canvas_size // 2, canvas_size // 2]], dtype=np.int16)

            return (
                i,
                np.clip(np.round(x_f * 127.0), -127, 127).astype(np.int8),
                np.clip(np.round(y_f * 127.0), -127, 127).astype(np.int8),
                coords,
            )

        workers = min(10, max(4, (os.cpu_count() or 8)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, xi, yi, coords in tqdm(ex.map(_proc, enumerate(train_chars)), total=n, desc="v2 HarfBuzz SDF", mininterval=5.0):
                inputs_i8[i] = xi
                targets_i8[i] = yi
                edge_coords[i] = coords

        np.save(in_p, inputs_i8)
        np.save(tg_p, targets_i8)
        np.savez_compressed(ec_p, **{f"c{i}": c for i, c in enumerate(edge_coords)})

    model = LocalStrokeSDFNet(4, base_channels=base_channels, max_shift_px=max_shift_px).to(device)
    criterion = HessianStraightnessLoss(trunc_px=trunc_px).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=5e-5)

    n_chars = len(inputs_i8)
    steps_per_epoch = (n_chars * 2) // batch_size
    half_p = patch_size // 2

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        l1_sum, hess_sum, iou_sum = 0.0, 0.0, 0.0
        for _ in range(steps_per_epoch):
            c_ids = np.random.randint(0, n_chars, size=batch_size)
            bx_i8 = np.empty((batch_size, 4, patch_size, patch_size), dtype=np.int8)
            by_i8 = np.empty((batch_size, 1, patch_size, patch_size), dtype=np.int8)
            for bi, cid in enumerate(c_ids):
                coords = edge_coords[cid]
                cy, cx = coords[np.random.randint(len(coords))]
                cy = int(np.clip(cy + np.random.randint(-half_p // 3, half_p // 3 + 1), half_p, canvas_size - half_p))
                cx = int(np.clip(cx + np.random.randint(-half_p // 3, half_p // 3 + 1), half_p, canvas_size - half_p))
                y0, x0 = cy - half_p, cx - half_p
                bx_i8[bi] = inputs_i8[cid, :, y0 : y0 + patch_size, x0 : x0 + patch_size]
                by_i8[bi] = targets_i8[cid, :, y0 : y0 + patch_size, x0 : x0 + patch_size]
            bx = torch.from_numpy(bx_i8).to(device, non_blocking=True).float().mul_(1.0 / 127.0)
            by = torch.from_numpy(by_i8).to(device, non_blocking=True).float().mul_(1.0 / 127.0)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred, flow = model(bx)
                loss, m = criterion(pred, by, bx, flow)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            l1_sum += m["sdf_l1"]
            hess_sum += m["hess"]
            iou_sum += m["soft_iou"]
        sched.step()
        print(
            f"  [v2-SDF] Epoch {ep:02d}/{epochs:02d} ({time.time()-t0:.1f}s) | "
            f"sdf_l1={l1_sum/steps_per_epoch:.4f} | hess={hess_sum/steps_per_epoch:.4f} | iou={iou_sum/steps_per_epoch:.4f}",
            flush=True,
        )

    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_out)
    print(f"[v2-SDF] Saved Hessian-regularized checkpoint -> {ckpt_out}", flush=True)
    return ckpt_out


class ZHSDFSynthesizer:
    """
    Runs inference on `uharfbuzz(zh-Hans)` inputs -> extracts zero-level-set in 1000 UPEM
    matching `rasterize_cmds(..., padding=96.0)` coordinate system -> applies
    `optimize_synthesized_contours` (0 mid-stem nodes, 0.85u concave waist)!
    """

    def __init__(
        self,
        ckpt_path: Path,
        serif_sc: HBOutlineExtractor,
        sans_sc: HBOutlineExtractor,
        canvas_size: int = 256,
        base_channels: int = 64,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.canvas_size = canvas_size
        self.trunc_px = 16.0 * (canvas_size / 256.0)
        self.serif_sc = serif_sc
        self.sans_sc = sans_sc

        self.model = LocalStrokeSDFNet(4, base_channels=base_channels, max_shift_px=6.0 * (canvas_size / 256.0)).to(self.device)
        state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

        # Match exact CanvasSpec corresponding to `rasterize_cmds(padding=96.0)`:
        # In rasterize_cmds: px = (x_upem + 96) * scale; py = (880 + 96 - y_upem) * scale, scale = S / 1192
        self.upem_spec = CanvasSpec(
            canvas_size=canvas_size,
            em_ratio=1000.0 / 1192.0,
            upem=1000,
            cjk_ascender=880,
        )

    def _vectorize_one(self, args: Tuple[str, np.ndarray, int]) -> Tuple[str, List[DrawCmd]]:
        ch, sdf_arr, upsample_factor = args
        raw_cmds = sdf_to_cff_commands(sdf_arr, self.upem_spec, upsample_factor=upsample_factor)
        clean_cmds = optimize_synthesized_contours(raw_cmds, stem_concave_sagitta=0.85)
        return ch, clean_cmds

    def synthesize_batch(
        self,
        chars: List[str],
        batch_size: int = 32,
        upsample_factor: int = 3,
        cache_path: Optional[Path] = None,
    ) -> dict[str, List[DrawCmd]]:
        results: Dict[str, List[DrawCmd]] = {}
        cached_raw: Dict[str, list] = {}
        if cache_path is not None and cache_path.exists():
            try:
                cached_raw = json.loads(cache_path.read_text(encoding="utf-8"))
                for ch in chars:
                    if ch in cached_raw:
                        results[ch] = [
                            (str(op), [(float(pt[0]), float(pt[1])) for pt in pts])
                            for op, pts in cached_raw[ch]
                        ]
            except Exception as e:
                print(f"[v2-SDF] Cache read warning ({cache_path}): {e}", flush=True)

        missing = [c for c in chars if c not in results]
        if not missing:
            return results

        amp_dtype = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        workers = min(8, max(4, (os.cpu_count() or 4)))
        with torch.no_grad(), ThreadPoolExecutor(max_workers=workers) as ex:
            for st in range(0, len(missing), batch_size):
                sub = missing[st : st + batch_size]
                bx_list = []
                for ch in sub:
                    _, c_sr = self.serif_sc.get_gid_and_cmds(ch, "zh-Hans")
                    _, c_sn = self.sans_sc.get_gid_and_cmds(ch, "zh-Hans")
                    bx_list.append(cmds_to_dual_sdf_input(c_sr, c_sn, self.canvas_size, self.trunc_px))
                xt = torch.from_numpy(np.stack(bx_list, axis=0)).to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=self.device.type == "cuda"):
                    pred_t, _ = self.model(xt)
                pred_np = pred_t[:, 0].float().cpu().numpy()

                tasks = [(ch, pred_np[i], upsample_factor) for i, ch in enumerate(sub)]
                for ch, clean_cmds in ex.map(self._vectorize_one, tasks):
                    results[ch] = clean_cmds
                    if cache_path is not None:
                        cached_raw[ch] = [[op, [[round(float(x), 1), round(float(y), 1)] for x, y in pts]] for op, pts in clean_cmds]

        if cache_path is not None and missing:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_p = cache_path.with_suffix(".tmp")
            tmp_p.write_text(json.dumps(cached_raw, ensure_ascii=False), encoding="utf-8")
            tmp_p.replace(cache_path)
            print(f"[v2-SDF] Updated Tier3 vector cache ({len(cached_raw)} glyphs) -> {cache_path}", flush=True)
        return results
