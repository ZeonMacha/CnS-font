#!/usr/bin/env python3
"""
Evaluate v4 model on holdout characters, compare with v3 baseline.

Metrics:
  - SDF L1 (lower is better)
  - Soft IoU (higher is better)
  - Skeleton Chamfer distance vs SkipPro GT (lower is better)
  - Contour Hausdorff distance (lower is better)
  - Leak ratio: IoU(pred, Sans620) / IoU(pred, SkipPro GT) (lower is better)
"""

import sys
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt as edt

sys.path.insert(0, ".")

from src.v4.model_v4 import GlobalGlyphTransformer
from src.v2.hb_outlines import HBOutlineExtractor, SkipProOutlineExtractor, rasterize_cmds
from src.sdf_utils import mask_to_normalized_sdf
from src.glyph_renderer import CanvasSpec
from src.vectorizer import sdf_to_cff_commands

# Inlined to avoid pathops dependency
def split_cmds_into_contours(cmds: List) -> List[List]:
    """Split a flat DrawCmd sequence into closed contour DrawCmd lists."""
    contours: List[List] = []
    curr: List = []
    for op, pts in cmds:
        curr.append((op, list(pts)))
        if op in ("closePath", "endPath"):
            if curr:
                contours.append(curr)
                curr = []
    if curr:
        contours.append(curr)
    return contours

def _flatten_contour_poly(c_cmds: List, step_upem: float = 16.0) -> Tuple[np.ndarray, float]:
    """Flatten a closed contour DrawCmd list to dense polyline + exact signed area in UPEM."""
    pts: List[Tuple[float, float]] = []
    curr = (0.0, 0.0)
    for op, cpts in c_cmds:
        if op == "moveTo":
            curr = cpts[0]
            pts.append(curr)
        elif op == "lineTo":
            curr = cpts[0]
            pts.append(curr)
        elif op == "curveTo":
            p0 = np.array(curr, dtype=np.float64)
            p1 = np.array(cpts[0], dtype=np.float64)
            p2 = np.array(cpts[1], dtype=np.float64)
            p3 = np.array(cpts[2], dtype=np.float64)
            chord = float(np.linalg.norm(p3 - p0))
            n_seg = max(3, min(16, int(chord / step_upem) + 1))
            for t in np.linspace(0.0, 1.0, n_seg + 1)[1:]:
                u = 1.0 - t
                q = (u ** 3) * p0 + 3.0 * (u ** 2) * t * p1 + 3.0 * u * (t ** 2) * p2 + (t ** 3) * p3
                pts.append((float(q[0]), float(q[1])))
            curr = cpts[2]

    if len(pts) < 3:
        return np.zeros((0, 2), dtype=np.float64), 0.0
    arr = np.array(pts, dtype=np.float64)
    if np.linalg.norm(arr[0] - arr[-1]) < 1e-3 and len(arr) > 3:
        arr = arr[:-1]
    x, y = arr[:, 0], arr[:, 1]
    area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return arr, area


def evaluate_v4(
    ckpt_path: Path,
    skip_otf: Path,
    noto_sans: Path,
    holdout_chars: list,
    canvas_size: int = 256,
    d_model: int = 768,
    n_transformer_layers: int = 4,
    n_heads: int = 12,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GlobalGlyphTransformer(
        in_channels=4, canvas_size=canvas_size,
        d_model=d_model, n_heads=n_heads,
        n_transformer_layers=n_transformer_layers,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    skip_ext = SkipProOutlineExtractor(skip_otf)
    sn = HBOutlineExtractor(noto_sans, ttc_index=0, weight=None)
    sn620 = HBOutlineExtractor(noto_sans, ttc_index=2, weight=620) if _has_weight(noto_sans) else None

    upem_spec = CanvasSpec(canvas_size=canvas_size, em_ratio=1000.0 / 1192.0, upem=1000, cjk_ascender=880)

    results = []
    for ch in holdout_chars:
        if not skip_ext.has_char(ch):
            continue

        # Build input
        c_l = sn.get_gid_and_cmds(ch, "ja")[1]
        if not c_l:
            continue

        from src.v4.dataset_v4 import build_deskinned_channels, pack_input_tensor
        sl, dtu, c2, s2 = build_deskinned_channels(c_l, canvas_size=canvas_size, padding=96.0, trunc_px=16.0)
        x_np = pack_input_tensor(sl, dtu, c2, s2, uniform_half_width_u=40.0, canvas_size=canvas_size, padding=96.0, trunc_px=16.0)
        x_t = torch.from_numpy(x_np[None]).to(device)

        with torch.no_grad():
            pred_sdf, _ = model(x_t)
        pred_np = pred_sdf[0, 0].float().cpu().numpy()

        # Target
        gt_cmds = skip_ext.get_cmds(ch)
        m_gt = rasterize_cmds(gt_cmds, canvas_size=canvas_size, padding=96.0)
        gt_sdf = mask_to_normalized_sdf(m_gt, canvas_size, trunc_px=16.0)

        # SDF L1
        sdf_l1 = float(np.abs(pred_np - gt_sdf).mean())

        # Soft IoU
        pred_ink = (pred_np >= 0).astype(np.float32)
        gt_ink = (gt_sdf >= 0).astype(np.float32)
        inter = (pred_ink * gt_ink).sum()
        union = (pred_ink + gt_ink - pred_ink * gt_ink).sum() + 1e-6
        iou = float(inter / union)

        # Skeleton Chamfer
        sp = skeletonize(pred_np >= 0)
        sg = skeletonize(m_gt)
        chamfer = None
        if sp.any() and sg.any():
            upem_px = 1192.0 / canvas_size
            dt_sg = edt(~sg) * upem_px
            dt_sp = edt(~sp) * upem_px
            chamfer = 0.5 * (float(dt_sg[sp].mean()) + float(dt_sp[sg].mean()))

        # Contour Hausdorff
        hausdorff = None
        pred_cmds = sdf_to_cff_commands(pred_np, upem_spec, upsample_factor=2)
        if pred_cmds:
            pts_p = np.vstack([_flatten_contour_poly(c, step_upem=12.0)[0] for c in split_cmds_into_contours(pred_cmds)] or [np.zeros((1, 2))])
            pts_g = np.vstack([_flatten_contour_poly(c, step_upem=12.0)[0] for c in split_cmds_into_contours(gt_cmds)] or [np.zeros((1, 2))])
            if len(pts_p) > 3 and len(pts_g) > 3:
                hd = max(float(cKDTree(pts_g).query(pts_p)[0].max()), float(cKDTree(pts_p).query(pts_g)[0].max()))
                hausdorff = hd

        # Leak ratio (vs Sans 620 if available)
        leak = None
        if sn620 is not None:
            _, sn6_cmds = sn620.get_gid_and_cmds(ch, "zh-Hans")
            if sn6_cmds:
                m_620 = rasterize_cmds(sn6_cmds, canvas_size=canvas_size, padding=96.0)
                iou_620 = float((pred_ink * (m_620 > 0).astype(np.float32)).sum() / max((pred_ink + (m_620 > 0).astype(np.float32) - pred_ink * (m_620 > 0).astype(np.float32)).sum(), 1))
                leak = iou_620 / max(iou, 1e-4)

        results.append({
            "char": ch,
            "sdf_l1": round(sdf_l1, 4),
            "iou": round(iou, 4),
            "skeleton_chamfer": round(chamfer, 2) if chamfer else None,
            "contour_hausdorff": round(hausdorff, 2) if hausdorff else None,
            "leak_ratio": round(leak, 4) if leak else None,
        })

    skip_ext.close()

    # Aggregate
    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    summary = {
        "n_eval": len(results),
        "sdf_l1_mean": _mean([r["sdf_l1"] for r in results]),
        "iou_mean": _mean([r["iou"] for r in results]),
        "skeleton_chamfer_mean": _mean([r["skeleton_chamfer"] for r in results]),
        "contour_hausdorff_mean": _mean([r["contour_hausdorff"] for r in results]),
        "leak_ratio_mean": _mean([r["leak_ratio"] for r in results]),
        "per_char": results,
    }
    return summary


def _has_weight(font_path: Path) -> bool:
    """Check if font has fvar table (variable weight)."""
    try:
        from fontTools.ttLib import TTFont
        f = TTFont(str(font_path), fontNumber=0)
        has = "fvar" in f.keys()
        f.close()
        return has
    except Exception:
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--skip_otf", type=Path, default=Path("FOT-SkipPro-B.OTF"))
    parser.add_argument("--noto_sans", type=Path, required=True)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--out_json", type=Path, default=Path("outputs/v4_eval_report.json"))
    args = parser.parse_args()

    holdout = list("志洳菜嘉改栅畴逡红语银饭伞丌习乡卫专书东言直令骨微道龙丝")
    print(f"Evaluating on {len(holdout)} holdout chars...")

    result = evaluate_v4(
        ckpt_path=args.ckpt,
        skip_otf=args.skip_otf,
        noto_sans=args.noto_sans,
        holdout_chars=holdout,
        d_model=args.d_model,
        n_transformer_layers=args.n_layers,
        n_heads=args.n_heads,
    )

    # Print summary
    print(f"\n=== v4 Evaluation Summary ===")
    print(f"Chars evaluated: {result['n_eval']}")
    print(f"SDF L1 (mean):    {result['sdf_l1_mean']}")
    print(f"Soft IoU (mean):  {result['iou_mean']}")
    print(f"Skel Chamfer:     {result['skeleton_chamfer_mean']}")
    print(f"Contour Hausdorff:{result['contour_hausdorff_mean']}")
    print(f"Leak ratio:       {result['leak_ratio_mean']}")

    # Save
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {args.out_json}")


if __name__ == "__main__":
    main()
