#!/usr/bin/env python3
"""
Milestone M2: Build Deskinned Light Cache, Run 4-Group Controlled Ablation (A/B/C/D),
Verify Dead-Loss Health & Style-Leak Reduction, and Train Full 18-Epoch `v3_zh_sdf_light.pt`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.glyph_renderer import CanvasSpec
from src.sdf_utils import mask_to_normalized_sdf
from src.v2.hb_outlines import HBOutlineExtractor, SkipProOutlineExtractor, cmds_to_bbox, rasterize_cmds
from src.v2.zh_sdf_fallback import HessianStraightnessLoss
from src.v2.curve_optimizer import _flatten_contour_poly, split_cmds_into_contours
from src.v3.hb_light import (
    SansLightExtractor,
    build_v3_deskinned_channels,
    extract_clean_skeleton,
    measure_skippro_hv_halfwidths,
    pack_v3_input_tensor,
)
from src.v3.losses_v3 import DeadLossMonitor, DeskinnedStrokeLossV3
from src.v3.model_v3 import LocalStrokeSDFNetV3
from src.vectorizer import sdf_to_cff_commands


def _build_or_load_v3_cache(
    skip_otf: Path,
    noto_sans: Path,
    cache_dir: Path,
    canvas_size: int = 256,
    patch_size: int = 96,
):
    skip_ext = SkipProOutlineExtractor(skip_otf)
    train_chars = sorted(chr(cp) for cp in skip_ext.cmap if 0x4E00 <= cp <= 0x9FFF)
    holdout = set("志洳菜嘉改栅畴逡红语银饭伞丌习乡卫专书东言直令骨微道")
    train_chars = [c for c in train_chars if c not in holdout]
    skip_ext.close()

    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"v3_light_n{len(train_chars)}_s{canvas_size}"
    sdf_l_p = cache_dir / f"v3_sdflight_{tag}.npy"
    dtsk_p = cache_dir / f"v3_dtskel_{tag}.npy"
    cos_p = cache_dir / f"v3_cos2t_{tag}.npy"
    sin_p = cache_dir / f"v3_sin2t_{tag}.npy"

    # Reuse v2 targets & edges if already present on 192.168.0.144
    v2_tag = f"v2_hb_n{len(train_chars)}_s{canvas_size}"
    tg_p = cache_dir / f"targets_{v2_tag}.npy"
    ec_p = cache_dir / f"edges_{v2_tag}.npz"
    v2_in_p = cache_dir / f"inputs_{v2_tag}.npy"

    n = len(train_chars)
    if sdf_l_p.exists() and dtsk_p.exists() and cos_p.exists() and sin_p.exists() and tg_p.exists() and ec_p.exists():
        print(f"[v3-Cache] Loading existing deskinned Light cache from {cache_dir}...", flush=True)
        sdf_l = np.load(sdf_l_p)
        dt_sk = np.load(dtsk_p)
        cos2t = np.load(cos_p)
        sin2t = np.load(sin_p)
        targets_i8 = np.load(tg_p)
        ec_npz = np.load(ec_p)
        edge_coords = [ec_npz[f"c{i}"] for i in range(n)]
    else:
        print(f"[v3-Cache] Building deskinned Sans Light cache for {n} chars...", flush=True)
        sdf_l = np.empty((n, canvas_size, canvas_size), dtype=np.int8)
        dt_sk = np.empty((n, canvas_size, canvas_size), dtype=np.uint8)
        cos2t = np.empty((n, canvas_size, canvas_size), dtype=np.int8)
        sin2t = np.empty((n, canvas_size, canvas_size), dtype=np.int8)
        need_targets = not (tg_p.exists() and ec_p.exists())
        if not need_targets:
            targets_i8 = np.load(tg_p)
            ec_npz = np.load(ec_p)
            edge_coords = [ec_npz[f"c{i}"] for i in range(n)]
        else:
            targets_i8 = np.empty((n, 1, canvas_size, canvas_size), dtype=np.int8)
            edge_coords = [np.empty((1, 2), dtype=np.int16)] * n
        half = patch_size // 2
        trunc_px = 16.0 * (canvas_size / 256.0)
        tls = threading.local()

        def _worker(item: Tuple[int, str]):
            i, ch = item
            if not hasattr(tls, "snL"):
                tls.snL = SansLightExtractor(noto_sans, language="ja", weight=300)
                if need_targets:
                    tls.sk = SkipProOutlineExtractor(skip_otf)
            c_l = tls.snL.get_cmds(ch, unioned=False)
            sl, dtu, c2, s2 = build_v3_deskinned_channels(c_l, canvas_size=canvas_size, padding=96.0, trunc_px=trunc_px)
            y_i8 = None
            coords = None
            if need_targets:
                m_sk = rasterize_cmds(tls.sk.get_cmds(ch), canvas_size=canvas_size * 2, padding=96.0)
                y_f = mask_to_normalized_sdf(m_sk, canvas_size, trunc_px=trunc_px)[None, :, :]
                ys, xs = np.where(np.abs(y_f[0]) < 0.42)
                if len(ys) > 0:
                    coords = np.stack([np.clip(ys, half, canvas_size - half), np.clip(xs, half, canvas_size - half)], axis=1).astype(np.int16)
                    if len(coords) > 220:
                        coords = coords[np.random.choice(len(coords), 220, replace=False)]
                else:
                    coords = np.array([[canvas_size // 2, canvas_size // 2]], dtype=np.int16)
                y_i8 = np.clip(np.round(y_f * 127.0), -127, 127).astype(np.int8)

            return (
                i,
                np.clip(np.round(sl * 127.0), -127, 127).astype(np.int8),
                np.clip(np.round(dtu), 0, 255).astype(np.uint8),
                np.clip(np.round(c2 * 127.0), -127, 127).astype(np.int8),
                np.clip(np.round(s2 * 127.0), -127, 127).astype(np.int8),
                y_i8,
                coords,
            )

        workers = min(10, max(4, (os.cpu_count() or 8)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for res in tqdm(ex.map(_worker, enumerate(train_chars)), total=n, desc="v3 Light Cache", mininterval=5.0):
                i, sl_i8, dtu_u8, c2_i8, s2_i8, y_i8, coords = res
                sdf_l[i] = sl_i8
                dt_sk[i] = dtu_u8
                cos2t[i] = c2_i8
                sin2t[i] = s2_i8
                if need_targets and y_i8 is not None and coords is not None:
                    targets_i8[i] = y_i8
                    edge_coords[i] = coords

        np.save(sdf_l_p, sdf_l)
        np.save(dtsk_p, dt_sk)
        np.save(cos_p, cos2t)
        np.save(sin_p, sin2t)
        if need_targets:
            np.save(tg_p, targets_i8)
            np.savez_compressed(ec_p, **{f"c{i}": c for i, c in enumerate(edge_coords)})

    v2_in = np.load(v2_in_p) if v2_in_p.exists() else None
    return train_chars, sdf_l, dt_sk, cos2t, sin2t, targets_i8, edge_coords, v2_in


def _build_group_batch(
    group: str,
    c_ids: np.ndarray,
    y0_arr: np.ndarray,
    x0_arr: np.ndarray,
    patch_size: int,
    sdf_l: np.ndarray,
    dt_sk: np.ndarray,
    cos2t: np.ndarray,
    sin2t: np.ndarray,
    targets_i8: np.ndarray,
    v2_in: np.ndarray | None,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bs = len(c_ids)
    by_i8 = np.empty((bs, 1, patch_size, patch_size), dtype=np.int8)
    if group == "A" and v2_in is not None:
        bx_i8 = np.empty((bs, 4, patch_size, patch_size), dtype=np.int8)
        for bi, (cid, y0, x0) in enumerate(zip(c_ids, y0_arr, x0_arr)):
            bx_i8[bi] = v2_in[cid, :, y0 : y0 + patch_size, x0 : x0 + patch_size]
            by_i8[bi] = targets_i8[cid, :, y0 : y0 + patch_size, x0 : x0 + patch_size]
        return (
            torch.from_numpy(bx_i8).to(device, non_blocking=True).float().mul_(1.0 / 127.0),
            torch.from_numpy(by_i8).to(device, non_blocking=True).float().mul_(1.0 / 127.0),
        )

    b_sl = np.empty((bs, patch_size, patch_size), dtype=np.int8)
    b_dt = np.empty((bs, patch_size, patch_size), dtype=np.uint8)
    b_c2 = np.empty((bs, patch_size, patch_size), dtype=np.int8)
    b_s2 = np.empty((bs, patch_size, patch_size), dtype=np.int8)
    for bi, (cid, y0, x0) in enumerate(zip(c_ids, y0_arr, x0_arr)):
        b_sl[bi] = sdf_l[cid, y0 : y0 + patch_size, x0 : x0 + patch_size]
        b_dt[bi] = dt_sk[cid, y0 : y0 + patch_size, x0 : x0 + patch_size]
        b_c2[bi] = cos2t[cid, y0 : y0 + patch_size, x0 : x0 + patch_size]
        b_s2[bi] = sin2t[cid, y0 : y0 + patch_size, x0 : x0 + patch_size]
        by_i8[bi] = targets_i8[cid, :, y0 : y0 + patch_size, x0 : x0 + patch_size]

    tsl = torch.from_numpy(b_sl).to(device, non_blocking=True).float().mul_(1.0 / 127.0)
    tdt = torch.from_numpy(b_dt).to(device, non_blocking=True).float()
    tc2 = torch.from_numpy(b_c2).to(device, non_blocking=True).float().mul_(1.0 / 127.0)
    ts2 = torch.from_numpy(b_s2).to(device, non_blocking=True).float().mul_(1.0 / 127.0)
    by = torch.from_numpy(by_i8).to(device, non_blocking=True).float().mul_(1.0 / 127.0)

    trunc_u = 16.0 * (1192.0 / 256.0)
    if group == "B":
        bx = torch.stack([tsl, tsl, torch.zeros_like(tsl), torch.zeros_like(tsl)], dim=1)
    elif group == "C":
        tuni = torch.clamp((40.0 - tdt) / trunc_u, -1.0, 1.0)
        bx = torch.stack([tsl, tuni, torch.zeros_like(tsl), torch.zeros_like(tsl)], dim=1)
    else:
        # Group D: Continuous weight & uniform-radius jitter on GPU (Plan 3.0 §1.2)
        dw = (torch.rand(bs, 1, 1, device=device) * 0.18 - 0.08)
        ru = (torch.rand(bs, 1, 1, device=device) * 16.0 + 32.0)
        tsl_j = torch.clamp(tsl + dw, -1.0, 1.0)
        tuni_j = torch.clamp((ru - tdt) / trunc_u, -1.0, 1.0)
        bx = torch.stack([tsl_j, tuni_j, tc2, ts2], dim=1)
    return bx, by


def _train_group(
    group: str,
    epochs: int,
    train_indices: np.ndarray,
    data_tuple: tuple,
    batch_size: int = 64,
    patch_size: int = 96,
    canvas_size: int = 256,
    base_channels: int = 64,
    lr: float = 1.2e-3,
    save_ckpt: Path | None = None,
):
    _, sdf_l, dt_sk, cos2t, sin2t, targets_i8, edge_coords, v2_in = data_tuple
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    model = LocalStrokeSDFNetV3(4, base_channels=base_channels, max_shift_px=8.0, delta_scale=0.95).to(device)
    criterion = (HessianStraightnessLoss(trunc_px=16.0) if group == "A" else DeskinnedStrokeLossV3(trunc_px=16.0)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=5e-5)
    monitor = DeadLossMonitor(keys=("sdf_l1", "grad_l1", "stem_norm", "cldice") if group != "A" else ("sdf_l1", "grad_l1", "hess"))

    steps_per_ep = (len(train_indices) * 2) // batch_size
    half_p = patch_size // 2
    epoch_logs: List[dict] = []

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        acc: Dict[str, float] = {}
        for _ in range(steps_per_ep):
            c_ids = np.random.choice(train_indices, size=batch_size, replace=True)
            y0_arr = np.empty(batch_size, dtype=np.int32)
            x0_arr = np.empty(batch_size, dtype=np.int32)
            for bi, cid in enumerate(c_ids):
                coords = edge_coords[cid]
                cy, cx = coords[np.random.randint(len(coords))]
                cy = int(np.clip(cy + np.random.randint(-half_p // 3, half_p // 3 + 1), half_p, canvas_size - half_p))
                cx = int(np.clip(cx + np.random.randint(-half_p // 3, half_p // 3 + 1), half_p, canvas_size - half_p))
                y0_arr[bi] = cy - half_p
                x0_arr[bi] = cx - half_p

            bx, by = _build_group_batch(
                group, c_ids, y0_arr, x0_arr, patch_size, sdf_l, dt_sk, cos2t, sin2t, targets_i8, v2_in, device
            )
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred, flow = model(bx)
                loss, m = criterion(pred, by, bx, flow)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in m.items():
                acc[k] = acc.get(k, 0.0) + float(v)
        sched.step()

        avg_m = {k: round(v / steps_per_ep, 5) for k, v in acc.items()}
        alerts = monitor.step(ep, avg_m)
        avg_m["epoch"] = ep
        avg_m["sec"] = round(time.time() - t0, 1)
        epoch_logs.append(avg_m)
        print(f"  [Group {group}] Ep {ep:02d}/{epochs:02d} ({avg_m['sec']:.1f}s) | {avg_m}", flush=True)
        for al in alerts:
            print(f"  [ALERT Group {group}] {al}", flush=True)

    if save_ckpt is not None:
        save_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_ckpt)
        print(f"[Group {group}] Saved checkpoint -> {save_ckpt}", flush=True)

    return model, epoch_logs, monitor.alerts


def _evaluate_ablation_holdout(
    group: str,
    model: torch.nn.Module,
    val_chars: List[str],
    skip_otf: Path,
    noto_sans: Path,
    noto_serif: Path,
    canvas_size: int = 256,
) -> dict:
    """Measure Leak = IoU(pred, Sans620)/IoU(pred, SkipProGT), skeleton Chamfer, H/V ratio err, Hausdorff."""
    device = next(model.parameters()).device
    model.eval()
    sk_ext = SkipProOutlineExtractor(skip_otf)
    sn620 = HBOutlineExtractor(noto_sans, ttc_index=2, weight=620)
    snL = SansLightExtractor(noto_sans, language="ja", weight=300)
    sr700 = HBOutlineExtractor(noto_serif, ttc_index=0, weight=700) if group == "A" else None
    snJ620 = HBOutlineExtractor(noto_sans, ttc_index=0, weight=620) if group == "A" else None
    upem_spec = CanvasSpec(canvas_size=canvas_size, em_ratio=1000.0 / 1192.0, upem=1000, cjk_ascender=880)

    leaks, chamfers, hv_errs, hausdorffs = [], [], [], []
    with torch.no_grad():
        for ch in val_chars:
            gt_cmds = sk_ext.get_cmds(ch)
            _, sn6_cmds = sn620.get_gid_and_cmds(ch, "zh-Hans")
            if not gt_cmds or not sn6_cmds:
                continue

            if group == "A":
                from src.v2.zh_sdf_fallback import cmds_to_dual_sdf_input

                _, c1 = sr700.get_gid_and_cmds(ch, "ja")
                _, c2 = snJ620.get_gid_and_cmds(ch, "ja")
                x_np = cmds_to_dual_sdf_input(c1, c2, canvas_size=canvas_size, trunc_px=16.0)
            else:
                c_l = snL.get_cmds(ch, unioned=False)
                sl, dtu, c2, s2 = build_v3_deskinned_channels(c_l, canvas_size=canvas_size)
                if group == "B":
                    x_np = np.stack([sl, sl, np.zeros_like(sl), np.zeros_like(sl)], axis=0).astype(np.float32)
                elif group == "C":
                    tuni = np.clip((40.0 - dtu) / (16.0 * 1192.0 / canvas_size), -1.0, 1.0)
                    x_np = np.stack([sl, tuni, np.zeros_like(sl), np.zeros_like(sl)], axis=0).astype(np.float32)
                else:
                    x_np = pack_v3_input_tensor(sl, dtu, c2, s2, uniform_half_width_u=40.0, canvas_size=canvas_size)

            xt = torch.from_numpy(x_np[None]).to(device)
            pred_sdf = model(xt)[0][0, 0].float().cpu().numpy()
            m_pred = pred_sdf >= 0.0
            m_gt = rasterize_cmds(gt_cmds, canvas_size=canvas_size, padding=96.0)
            m_620 = rasterize_cmds(sn6_cmds, canvas_size=canvas_size, padding=96.0)

            iou_gt = int((m_pred & m_gt).sum()) / float(max(int((m_pred | m_gt).sum()), 1))
            iou_620 = int((m_pred & m_620).sum()) / float(max(int((m_pred | m_620).sum()), 1))
            leaks.append(iou_620 / max(iou_gt, 1e-4))

            # Skeleton Chamfer vs SkipPro GT
            upem_px = 1192.0 / canvas_size
            sp = skeletonize(m_pred)
            sg = skeletonize(m_gt)
            if sp.any() and sg.any():
                from scipy.ndimage import distance_transform_edt as edt

                ch_d = 0.5 * (float((edt(~sg) * upem_px)[sp].mean()) + float((edt(~sp) * upem_px)[sg].mean()))
                chamfers.append(ch_d)

            # Contour Hausdorff vs SkipPro GT & H/V stroke width ratio error
            pred_cmds = sdf_to_cff_commands(pred_sdf, upem_spec, upsample_factor=2)
            pts_p = np.vstack([_flatten_contour_poly(c, step_upem=12.0)[0] for c in split_cmds_into_contours(pred_cmds)] or [np.zeros((1, 2))])
            pts_g = np.vstack([_flatten_contour_poly(c, step_upem=12.0)[0] for c in split_cmds_into_contours(gt_cmds)] or [np.zeros((1, 2))])
            if len(pts_p) > 3 and len(pts_g) > 3:
                hd = max(float(cKDTree(pts_g).query(pts_p)[0].max()), float(cKDTree(pts_p).query(pts_g)[0].max()))
                hausdorffs.append(hd)

            hp, vp = _mask_hv_widths(m_pred, 1192.0 / canvas_size)
            hg, vg = _mask_hv_widths(m_gt, 1192.0 / canvas_size)
            if hp > 0 and vp > 0 and hg > 0 and vg > 0:
                hv_errs.append(abs((hp / vp) - (hg / vg)))

    sk_ext.close()
    return {
        "n_eval": len(leaks),
        "leak_ratio_mean": round(float(np.mean(leaks)), 4),
        "skel_chamfer_gt_mean": round(float(np.mean(chamfers)), 2),
        "hv_ratio_err_mean": round(float(np.mean(hv_errs)) if hv_errs else 0.0, 4),
        "contour_hausdorff_mean": round(float(np.mean(hausdorffs)), 2),
    }


def _mask_hv_widths(mask: np.ndarray, upem_per_px: float) -> Tuple[float, float]:
    import cv2
    from scipy.ndimage import distance_transform_edt as edt

    if not mask.any():
        return 0.0, 0.0
    dt = edt(mask.astype(np.uint8)) * upem_per_px
    skel = skeletonize(mask)
    ys, xs = np.where(skel)
    if len(ys) < 6:
        return 0.0, 0.0
    gx = cv2.Sobel(dt.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dt.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    hs, vs = [], []
    for y, x in zip(ys, xs):
        ax, ay = abs(float(gx[y, x])), abs(float(gy[y, x]))
        r = float(dt[y, x])
        if ay > 2.2 * ax:
            hs.append(r)
        elif ax > 2.2 * ay:
            vs.append(r)
    return (float(np.median(hs)) if hs else 0.0), (float(np.median(vs)) if vs else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M2 4-Group Deskinned Ablation & Train v3 Model")
    parser.add_argument("--skip_otf", type=Path, default=ROOT / "FOT-SkipPro-B.OTF")
    parser.add_argument("--noto_sans", type=Path, required=True)
    parser.add_argument("--noto_serif", type=Path, required=True, help="Used ONLY for Group A ablation baseline")
    parser.add_argument("--cache_dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--ablation_epochs", type=int, default=6)
    parser.add_argument("--full_epochs", type=int, default=18)
    parser.add_argument("--ckpt_out", type=Path, default=ROOT / "checkpoints" / "v3_zh_sdf_light.pt")
    parser.add_argument("--out_json", type=Path, default=ROOT / "outputs" / "v3_m2_ablation_report.json")
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    data_tuple = _build_or_load_v3_cache(args.skip_otf, args.noto_sans, args.cache_dir)
    train_chars = data_tuple[0]

    # Pick 80 strictly JP==SC validation chars from train_chars for fair holdout evaluation across A/B/C/D
    sn_jp = SansLightExtractor(args.noto_sans, "ja", 300)
    sn_sc = SansLightExtractor(args.noto_sans, "zh-Hans", 300)
    same_idx = [i for i, ch in enumerate(train_chars) if sn_jp.shape_gid(ch, "ja") == sn_sc.shape_gid(ch, "zh-Hans")]
    rng = np.random.default_rng(13)
    val_idx = sorted(rng.choice(same_idx, size=min(80, len(same_idx)), replace=False).tolist())
    val_set = set(val_idx)
    fit_idx = np.array([i for i in range(len(train_chars)) if i not in val_set], dtype=np.int32)
    val_chars = [train_chars[i] for i in val_idx]
    print(f"[M2] Fit chars: {len(fit_idx)} | Strictly held-out JP==SC val chars: {len(val_chars)}", flush=True)

    ablation_results: Dict[str, dict] = {}
    groups = ["A", "B", "C", "D"] if data_tuple[-1] is not None else ["B", "C", "D"]
    for grp in groups:
        print(f"\n=== M2 Ablation Group {grp} ({args.ablation_epochs} epochs) ===", flush=True)
        model_g, ep_logs, alerts = _train_group(
            grp,
            epochs=args.ablation_epochs,
            train_indices=fit_idx,
            data_tuple=data_tuple,
        )
        eval_m = _evaluate_ablation_holdout(grp, model_g, val_chars, args.skip_otf, args.noto_sans, args.noto_serif)
        ablation_results[grp] = {"eval": eval_m, "epoch_logs": ep_logs, "dead_loss_alerts": alerts}
        print(f"[Group {grp} Eval] {eval_m}", flush=True)

    # Full 18-epoch training of Group D on non-holdout chars
    print(f"\n=== Training Final Group D Deskinned Light Model ({args.full_epochs} epochs) ===", flush=True)
    _, full_logs, full_alerts = _train_group(
        "D",
        epochs=args.full_epochs,
        train_indices=fit_idx,
        data_tuple=data_tuple,
        save_ckpt=args.ckpt_out,
    )

    leak_a = ablation_results.get("A", {}).get("eval", {}).get("leak_ratio_mean", 1.0)
    leak_d = ablation_results["D"]["eval"]["leak_ratio_mean"]
    leak_drop_pct = (leak_a - leak_d) / max(leak_a, 1e-4) * 100.0

    report = {
        "ablation_groups": ablation_results,
        "leak_reduction_pct_vs_v2": round(leak_drop_pct, 2),
        "full_d_epoch_logs": full_logs,
        "full_d_dead_loss_alerts": full_alerts,
        "checkpoint": str(args.ckpt_out),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[M2 Summary] leak_A={leak_a:.4f} vs leak_D={leak_d:.4f} (reduction={leak_drop_pct:.1f}%) -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
