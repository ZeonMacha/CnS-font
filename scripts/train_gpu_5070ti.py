#!/usr/bin/env python3
"""
Full-Scale GPU Training & Evaluation Pipeline Optimized for NVIDIA RTX 5070 Ti.

Features:
  - Trains across ALL ~7,870 shared CJK characters in `FOT-SkipPro-B.OTF` (strictly
    holding out the 8 `zi2zi-JiT` failure chars + 72 validation chars = 80 val chars).
  - High-resolution `384x384` canvas (`768x768` supersampled sub-pixel SDF, `1536x1536`
    iso-contour vectorization) stored as compact `int8` SDF tensors (~2.9 GB RAM).
  - Multi-threaded parallel dataset builder + cached `.npz`/`.npy` for instant reload.
  - Restricted-Receptive-Field `LocalStrokeSDFNet` (`base_channels=64`, `128x128` patches)
    + `34x34` Spectral-Norm `LocalPatchDiscriminator` (Hinge GAN) + EMA weights + AMP bf16.
  - Exports validation comparison sheets, Simplified-12 showcase, AND optionally compiles
    the full GB2312 missing character set (2,983 glyphs) into `CnS-SkipPro-B-SC.otf`!
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_m1_poc import (
    compute_iou,
    render_simplified_12_sheet,
    render_validation_8_sheet,
)
from src.dataset import UNSEEN_SIMPLIFIED_12, ZI2ZI_JIT_FAILURE_8
from src.glyph_renderer import (
    AlignedFontRenderer,
    CanvasSpec,
    get_font_cjk_codepoints,
    get_gb2312_sets,
)
from src.losses import StrokeSDFLoss
from src.model import LocalPatchDiscriminator, LocalStrokeSDFNet
from src.otf_builder import evaluate_topology_qa, pack_glyphs_into_otf
from src.sdf_utils import build_dual_skeleton_input, mask_to_normalized_sdf
from src.vectorizer import DrawCmd, sdf_to_cff_commands


class FastInt8PatchDataset(Dataset):
    """High-throughput RAM-resident int8 SDF patch dataset (~2.9 GB for 7,800 chars @ 384px)."""

    def __init__(
        self,
        inputs_i8: np.ndarray,
        targets_i8: np.ndarray,
        edge_coords: List[np.ndarray],
        patch_size: int = 128,
        patches_per_char: int = 4,
    ) -> None:
        self.inputs_i8 = inputs_i8
        self.targets_i8 = targets_i8
        self.edge_coords = edge_coords
        self.canvas_size = inputs_i8.shape[-1]
        self.patch_size = patch_size
        self.patches_per_char = patches_per_char

    def __len__(self) -> int:
        return len(self.inputs_i8) * self.patches_per_char

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        c_idx = idx // self.patches_per_char
        p = self.patch_size
        half = p // 2
        coords = self.edge_coords[c_idx]

        if np.random.rand() < 0.88 and len(coords) > 0:
            cy, cx = coords[np.random.randint(len(coords))]
            cy = int(np.clip(cy + np.random.randint(-half // 3, half // 3 + 1), half, self.canvas_size - half))
            cx = int(np.clip(cx + np.random.randint(-half // 3, half // 3 + 1), half, self.canvas_size - half))
        else:
            cy = np.random.randint(half, self.canvas_size - half + 1)
            cx = np.random.randint(half, self.canvas_size - half + 1)

        y0, x0 = cy - half, cx - half
        x_patch = self.inputs_i8[c_idx, :, y0 : y0 + p, x0 : x0 + p].astype(np.float32) / 127.0
        y_patch = self.targets_i8[c_idx, :, y0 : y0 + p, x0 : x0 + p].astype(np.float32) / 127.0
        return torch.from_numpy(x_patch), torch.from_numpy(y_patch)


def build_or_load_full_cache(
    train_chars: List[str],
    skip_otf: Path,
    noto_serif: Path,
    noto_sans: Path,
    canvas_size: int,
    patch_size: int,
    trunc_px: float,
    cache_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"n{len(train_chars)}_s{canvas_size}"
    in_path = cache_dir / f"inputs_{tag}.npy"
    tg_path = cache_dir / f"targets_{tag}.npy"
    ec_path = cache_dir / f"edges_{tag}.npz"

    if in_path.exists() and tg_path.exists() and ec_path.exists():
        print(f"[Cache] Loading pre-built SDF arrays from {cache_dir}...")
        inputs_i8 = np.load(in_path)
        targets_i8 = np.load(tg_path)
        ec_npz = np.load(ec_path)
        edge_coords = [ec_npz[f"c{i}"] for i in range(len(train_chars))]
        return inputs_i8, targets_i8, edge_coords

    print(f"[Builder] Rendering & computing sub-pixel SDF for {len(train_chars)} chars @ {canvas_size}x{canvas_size}...")
    n = len(train_chars)
    inputs_i8 = np.empty((n, 4, canvas_size, canvas_size), dtype=np.int8)
    targets_i8 = np.empty((n, 1, canvas_size, canvas_size), dtype=np.int8)
    edge_coords: List[np.ndarray] = [np.empty((1, 2), dtype=np.int16)] * n
    half = patch_size // 2

    # Thread-local renderers since FreeType face handles are not thread-safe across threads
    import threading
    tls = threading.local()

    def _process_one(item: Tuple[int, str]) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        i, ch = item
        if not hasattr(tls, "serif"):
            spec = CanvasSpec(canvas_size=canvas_size)
            tls.serif = AlignedFontRenderer(noto_serif, spec, ttc_index=0, variation_weight=700)
            tls.sans = AlignedFontRenderer(noto_sans, spec, ttc_index=0, variation_weight=620)
            tls.skip = AlignedFontRenderer(skip_otf, spec, ttc_index=0)

        x_f = build_dual_skeleton_input(ch, tls.serif, tls.sans, trunc_px=trunc_px)
        t_mask = tls.skip.render_binary_hi(ch)
        y_f = mask_to_normalized_sdf(t_mask, canvas_size, trunc_px=trunc_px)[None, :, :]

        # Focus sampling heavily on stroke boundaries AND Serif-vs-Sans triangular serif zones!
        serif_zone = np.abs(x_f[2]) > 0.12
        edge_zone = np.abs(y_f[0]) < 0.42
        ys_s, xs_s = np.where(serif_zone)
        ys_e, xs_e = np.where(edge_zone)

        pts_list = []
        if len(ys_s) > 0:
            c_s = np.stack([np.clip(ys_s, half, canvas_size - half), np.clip(xs_s, half, canvas_size - half)], axis=1)
            if len(c_s) > 160:
                c_s = c_s[np.random.choice(len(c_s), 160, replace=False)]
            pts_list.append(c_s)
        if len(ys_e) > 0:
            c_e = np.stack([np.clip(ys_e, half, canvas_size - half), np.clip(xs_e, half, canvas_size - half)], axis=1)
            if len(c_e) > 200:
                c_e = c_e[np.random.choice(len(c_e), 200, replace=False)]
            pts_list.append(c_e)

        if pts_list:
            coords = np.concatenate(pts_list, axis=0).astype(np.int16)
        else:
            coords = np.array([[canvas_size // 2, canvas_size // 2]], dtype=np.int16)

        x_i8 = np.clip(np.round(x_f * 127.0), -127, 127).astype(np.int8)
        y_i8 = np.clip(np.round(y_f * 127.0), -127, 127).astype(np.int8)
        return i, x_i8, y_i8, coords

    workers = min(12, max(4, (os.cpu_count() or 8)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, x_i8, y_i8, coords in tqdm(
            ex.map(_process_one, enumerate(train_chars)),
            total=n,
            desc="Building SDF Cache",
            mininterval=5.0,
        ):
            inputs_i8[i] = x_i8
            targets_i8[i] = y_i8
            edge_coords[i] = coords

    np.save(in_path, inputs_i8)
    np.save(tg_path, targets_i8)
    np.savez_compressed(ec_path, **{f"c{i}": c for i, c in enumerate(edge_coords)})
    return inputs_i8, targets_i8, edge_coords


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.996) -> None:
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
    for b_ema, b in zip(ema_model.buffers(), model.buffers()):
        b_ema.data.copy_(b.data)


def main() -> None:
    parser = argparse.ArgumentParser(description="RTX 5070 Ti Full-Dataset Patch-Conv SDF + PatchGAN Trainer")
    parser.add_argument("--skip_otf", type=Path, default=ROOT / "FOT-SkipPro-B.OTF")
    parser.add_argument("--noto_serif", type=Path, required=True)
    parser.add_argument("--noto_sans", type=Path, required=True)
    parser.add_argument("--canvas_size", type=int, default=384)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--num_train_chars", type=int, default=0, help="0 = ALL ~7,830 training chars")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--gan_start_epoch", type=int, default=4)
    parser.add_argument("--gan_weight", type=float, default=0.06)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.2e-3)
    parser.add_argument("--upsample_factor", type=int, default=3)
    parser.add_argument("--build_gb2312", action="store_true", help="Also generate all 2,983 missing GB2312 chars")
    parser.add_argument("--out_dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    if use_amp:
        torch.backends.cudnn.benchmark = True
        print(f"[GPU] Using {torch.cuda.get_device_name(0)} | AMP={amp_dtype}")

    cjk_skip = get_font_cjk_codepoints(args.skip_otf, 0)
    cjk_serif = get_font_cjk_codepoints(args.noto_serif, 0)
    cjk_sans = get_font_cjk_codepoints(args.noto_sans, 0)
    shared = sorted(cjk_skip & cjk_serif & cjk_sans)

    must_holdout = {ord(c) for c in ZI2ZI_JIT_FAILURE_8}
    pool = [chr(cp) for cp in shared if cp not in must_holdout]
    rng = random.Random(42)
    rng.shuffle(pool)
    val_chars = list(ZI2ZI_JIT_FAILURE_8) + pool[:72]  # 80 strictly held-out validation chars
    train_chars = pool[72:]
    if 0 < args.num_train_chars < len(train_chars):
        train_chars = train_chars[: args.num_train_chars]

    assert all(c not in train_chars for c in val_chars), "Leakage check failed!"
    print(f"[Dataset] Train chars: {len(train_chars)} | Held-out Val chars: {len(val_chars)}")

    trunc_px = 16.0 * (args.canvas_size / 256.0)
    max_shift_px = 6.5 * (args.canvas_size / 256.0)

    inputs_i8, targets_i8, edge_coords = build_or_load_full_cache(
        train_chars=train_chars,
        skip_otf=args.skip_otf,
        noto_serif=args.noto_serif,
        noto_sans=args.noto_sans,
        canvas_size=args.canvas_size,
        patch_size=args.patch_size,
        trunc_px=trunc_px,
        cache_dir=ROOT / "datasets",
    )

    n_chars = len(inputs_i8)
    patches_per_char = 2
    steps_per_epoch = (n_chars * patches_per_char) // args.batch_size
    half_p = args.patch_size // 2

    def sample_gpu_batch() -> Tuple[torch.Tensor, torch.Tensor]:
        c_ids = np.random.randint(0, n_chars, size=args.batch_size)
        bx_i8 = np.empty((args.batch_size, 4, args.patch_size, args.patch_size), dtype=np.int8)
        by_i8 = np.empty((args.batch_size, 1, args.patch_size, args.patch_size), dtype=np.int8)
        for bi, cid in enumerate(c_ids):
            coords = edge_coords[cid]
            if np.random.rand() < 0.88 and len(coords) > 0:
                cy, cx = coords[np.random.randint(len(coords))]
                cy = int(np.clip(cy + np.random.randint(-half_p // 3, half_p // 3 + 1), half_p, args.canvas_size - half_p))
                cx = int(np.clip(cx + np.random.randint(-half_p // 3, half_p // 3 + 1), half_p, args.canvas_size - half_p))
            else:
                cy = np.random.randint(half_p, args.canvas_size - half_p + 1)
                cx = np.random.randint(half_p, args.canvas_size - half_p + 1)
            y0, x0 = cy - half_p, cx - half_p
            bx_i8[bi] = inputs_i8[cid, :, y0 : y0 + args.patch_size, x0 : x0 + args.patch_size]
            by_i8[bi] = targets_i8[cid, :, y0 : y0 + args.patch_size, x0 : x0 + args.patch_size]
        bx_t = torch.from_numpy(bx_i8).to(device, non_blocking=True).float().mul_(1.0 / 127.0)
        by_t = torch.from_numpy(by_i8).to(device, non_blocking=True).float().mul_(1.0 / 127.0)
        return bx_t, by_t

    model = LocalStrokeSDFNet(4, base_channels=args.base_channels, max_shift_px=max_shift_px).to(device)
    ema_model = copy.deepcopy(model).eval()
    disc = LocalPatchDiscriminator(in_channels=5, base_channels=48).to(device)

    criterion = StrokeSDFLoss(boundary_weight=10.0, grad_weight=2.0, iou_weight=0.8, trunc_px=trunc_px).to(device)
    opt_g = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr * 0.8, betas=(0.5, 0.99), weight_decay=1e-4)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs, eta_min=5e-5)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=args.epochs, eta_min=5e-5)

    for epoch in range(1, args.epochs + 1):
        model.train()
        disc.train()
        t_ep = time.time()
        use_gan = epoch >= args.gan_start_epoch
        g_loss_sum, d_loss_sum, iou_sum, steps = 0.0, 0.0, 0.0, 0

        for _ in range(steps_per_epoch):
            bx, by = sample_gpu_batch()

            # 1. Update Discriminator (Hinge GAN on inner border-free patch)
            if use_gan:
                opt_d.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    with torch.no_grad():
                        fake_sdf, _ = model(bx)
                    d_real = disc(bx[:, :, 4:-4, 4:-4], by[:, :, 4:-4, 4:-4])
                    d_fake = disc(bx[:, :, 4:-4, 4:-4], fake_sdf[:, :, 4:-4, 4:-4].detach())
                    loss_d = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
                loss_d.backward()
                opt_d.step()
                d_loss_sum += float(loss_d.detach())

            # 2. Update Generator
            opt_g.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred_sdf, flow_px = model(bx)
                loss_g, m = criterion(pred_sdf, by, bx, flow_px)
                if use_gan:
                    d_pred = disc(bx[:, :, 4:-4, 4:-4], pred_sdf[:, :, 4:-4, 4:-4])
                    loss_adv = -d_pred.mean()
                    loss_g = loss_g + args.gan_weight * loss_adv
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_g.step()
            update_ema(ema_model, model, decay=0.996)

            g_loss_sum += m["sdf_l1"]
            iou_sum += m["soft_iou"]
            steps += 1

        sched_g.step()
        if use_gan:
            sched_d.step()
        print(
            f"  Epoch {epoch:02d}/{args.epochs:02d} ({time.time()-t_ep:.1f}s) | "
            f"sdf_l1={g_loss_sum/steps:.4f} | patch_iou={iou_sum/steps:.4f} | "
            f"d_loss={d_loss_sum/max(steps,1):.4f}",
            flush=True,
        )

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"cns_local_sdf_s{args.canvas_size}_c{args.base_channels}.pt"
    torch.save(ema_model.state_dict(), ckpt_path)
    print(f"[Checkpoint] Saved EMA weights -> {ckpt_path}")

    # --- Evaluate on 80 Held-Out Validation Chars ---
    ema_model.eval()
    spec = CanvasSpec(canvas_size=args.canvas_size)
    serif_jp = AlignedFontRenderer(args.noto_serif, spec, ttc_index=0, variation_weight=700)
    sans_jp = AlignedFontRenderer(args.noto_sans, spec, ttc_index=0, variation_weight=620)
    skip_rd = AlignedFontRenderer(args.skip_otf, spec, ttc_index=0)

    ious_serif, ious_sans, ious_ours, phantoms, recalls = [], [], [], [], []
    val8_cmds: Dict[int, List[DrawCmd]] = {}

    with torch.no_grad():
        for ch in val_chars:
            x_np = build_dual_skeleton_input(ch, serif_jp, sans_jp, trunc_px=trunc_px)
            gt_sdf = mask_to_normalized_sdf(skip_rd.render_binary_hi(ch), args.canvas_size, trunc_px=trunc_px)
            xt = torch.from_numpy(x_np[None]).to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred_t, _ = ema_model(xt)
            pred_sdf = pred_t[0, 0].float().cpu().numpy()

            ious_serif.append(compute_iou(x_np[0] > 0.0, gt_sdf > 0.0))
            ious_sans.append(compute_iou(x_np[1] > 0.0, gt_sdf > 0.0))
            ious_ours.append(compute_iou(pred_sdf > 0.0, gt_sdf > 0.0))
            qa = evaluate_topology_qa(pred_sdf, x_np, trunc_px=trunc_px, max_allowed_drift_px=8.0 * args.canvas_size / 256.0)
            phantoms.append(qa["phantom_ink_rate"])
            recalls.append(qa["skeleton_recall"])

            if ch in ZI2ZI_JIT_FAILURE_8:
                val8_cmds[ord(ch)] = sdf_to_cff_commands(pred_sdf, spec, upsample_factor=args.upsample_factor)

    # --- Synthesize 12 Showcase Simplified Chars (+ Optional Full GB2312 2,983 Chars!) ---
    serif_sc = AlignedFontRenderer(args.noto_serif, spec, ttc_index=2, variation_weight=700)
    sans_sc = AlignedFontRenderer(args.noto_sans, spec, ttc_index=2, variation_weight=620)

    simp12_cmds: Dict[int, List[DrawCmd]] = {}
    with torch.no_grad():
        for ch in UNSEEN_SIMPLIFIED_12:
            x_np = build_dual_skeleton_input(ch, serif_sc, sans_sc, trunc_px=trunc_px)
            xt = torch.from_numpy(x_np[None]).to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred_t, _ = ema_model(xt)
            pred_sdf = pred_t[0, 0].float().cpu().numpy()
            simp12_cmds[ord(ch)] = sdf_to_cff_commands(pred_sdf, spec, upsample_factor=args.upsample_factor)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    poc_otf = args.out_dir / "CnS-SkipPro-B-M1-PoC.otf"
    pack_glyphs_into_otf(args.skip_otf, poc_otf, {**val8_cmds, **simp12_cmds})

    val_png = args.out_dir / "gpu_val_8_zi2zi_failures.png"
    simp_png = args.out_dir / "gpu_simplified_12_showcase.png"
    render_validation_8_sheet(ZI2ZI_JIT_FAILURE_8, args.skip_otf, poc_otf, args.noto_serif, args.noto_sans, val_png)
    render_simplified_12_sheet(UNSEEN_SIMPLIFIED_12, poc_otf, args.noto_serif, args.noto_sans, simp_png)

    metrics = {
        "train_chars": len(train_chars),
        "val_chars": len(val_chars),
        "canvas_size": args.canvas_size,
        "epochs": args.epochs,
        "val_mean_iou_raw_serif": round(float(np.mean(ious_serif)), 4),
        "val_mean_iou_raw_sans": round(float(np.mean(ious_sans)), 4),
        "val_mean_iou_ours": round(float(np.mean(ious_ours)), 4),
        "val8_failure_set_iou_ours": round(float(np.mean(ious_ours[:8])), 4),
        "max_phantom_ink_rate": round(float(np.max(phantoms)), 5),
        "min_skeleton_recall": round(float(np.min(recalls)), 4),
    }

    if args.build_gb2312:
        _, gb_all = get_gb2312_sets()
        avail_sc = get_font_cjk_codepoints(args.noto_serif, 2) & get_font_cjk_codepoints(args.noto_sans, 2)
        missing_gb = sorted((gb_all - cjk_skip) & avail_sc)
        print(f"[GB2312] Generating & vectorizing all {len(missing_gb)} missing GB2312 characters...")
        gb_cmds: Dict[int, List[DrawCmd]] = {}
        qa_flagged: Dict[str, dict] = {}
        bsize = 32
        with torch.no_grad():
            for st in tqdm(range(0, len(missing_gb), bsize), desc="GB2312 Synthesis", mininterval=5.0):
                batch_cps = missing_gb[st : st + bsize]
                bx_list = [build_dual_skeleton_input(chr(cp), serif_sc, sans_sc, trunc_px=trunc_px) for cp in batch_cps]
                xt = torch.from_numpy(np.stack(bx_list, axis=0)).to(device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    pt, _ = ema_model(xt)
                p_np = pt[:, 0].float().cpu().numpy()

                def _vec_item(idx_cp):
                    i, cp = idx_cp
                    qa = evaluate_topology_qa(
                        p_np[i], bx_list[i], trunc_px=trunc_px, max_allowed_drift_px=8.0 * args.canvas_size / 256.0
                    )
                    cmds = sdf_to_cff_commands(p_np[i], spec, upsample_factor=args.upsample_factor)
                    return cp, qa, cmds

                with ThreadPoolExecutor(max_workers=8) as vex:
                    for cp, qa, cmds in vex.map(_vec_item, enumerate(batch_cps)):
                        anc = sum(len(pts) for _, pts in cmds)
                        if qa["phantom_ink_rate"] > 0.015 or qa["skeleton_recall"] < 0.86 or anc > 300:
                            qa_flagged[chr(cp)] = {**qa, "anchor_count": anc}
                        gb_cmds[cp] = cmds

        full_otf = args.out_dir / "CnS-SkipPro-B-GB2312-SC.otf"
        packed_gb = pack_glyphs_into_otf(args.skip_otf, full_otf, gb_cmds)
        metrics["gb2312_synthesized_glyphs"] = packed_gb
        metrics["gb2312_qa_flagged_count"] = len(qa_flagged)
        (args.out_dir / "gb2312_qa_outliers.json").write_text(
            json.dumps(qa_flagged, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[GB2312] Compiled complete GB2312 font ({packed_gb} new glyphs) -> {full_otf}")

    (args.out_dir / "gpu_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[GPU SUMMARY]")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
