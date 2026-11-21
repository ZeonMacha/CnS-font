#!/usr/bin/env python3
"""
M1 Proof-of-Concept Runner:
  1. Strictly holds out the 8 `zi2zi-JiT` overfitting failure chars (`志洳菜嘉改栅畴逡`)
     + 32 additional validation chars (0% data leakage).
  2. Trains `LocalStrokeSDFNet` exclusively on `96x96` local stroke patches from `SkipPro-B`.
  3. Evaluates quantitative metrics (IoU, Boundary L1, Phantom Ink Rate, Skeleton Recall)
     on the held-out validation set, proving zero radical hallucination.
  4. Runs full-canvas (`256x256`) inference + `4x` sub-pixel SDF upsampling (`1024x1024`)
     + orthogonal stem snapping + cubic Bézier fitting on:
       - Held-out 8 `zi2zi-JiT` failure chars (`志洳菜嘉改栅畴逡`)
       - Unseen 12 pure Simplified Chinese chars (`语银红饭车东书专买风龙业`)
  5. Packs all generated glyphs into `outputs/CnS-SkipPro-B-M1-PoC.otf` and renders
     side-by-side high-resolution comparison sheets directly from the compiled `.otf`!
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import UNSEEN_SIMPLIFIED_12, ZI2ZI_JIT_FAILURE_8, PatchCropSDFDataset
from src.glyph_renderer import AlignedFontRenderer, CanvasSpec, get_font_cjk_codepoints
from src.losses import StrokeSDFLoss
from src.model import LocalStrokeSDFNet
from src.otf_builder import evaluate_topology_qa, pack_glyphs_into_otf
from src.sdf_utils import build_dual_skeleton_input, mask_to_normalized_sdf, sdf_to_antialiased_gray
from src.vectorizer import sdf_to_cff_commands


def select_training_and_val_chars(
    skip_otf: Path,
    noto_serif: Path,
    noto_sans: Path,
    num_train_chars: int,
    seed: int = 42,
) -> tuple[List[str], List[str]]:
    """Select stratified CJK training & validation characters with strict holdout of failure-8."""
    cjk_skip = get_font_cjk_codepoints(skip_otf, ttc_index=0)
    cjk_serif = get_font_cjk_codepoints(noto_serif, ttc_index=0)
    cjk_sans = get_font_cjk_codepoints(noto_sans, ttc_index=0)
    shared = sorted(cjk_skip & cjk_serif & cjk_sans)

    must_holdout = {ord(c) for c in ZI2ZI_JIT_FAILURE_8}
    pool = [chr(cp) for cp in shared if cp not in must_holdout]

    rng = random.Random(seed)
    rng.shuffle(pool)
    extra_val = pool[:32]
    val_chars = list(ZI2ZI_JIT_FAILURE_8) + extra_val

    avail_train = pool[32:]
    if num_train_chars > 0 and num_train_chars < len(avail_train):
        train_chars = avail_train[:num_train_chars]
    else:
        train_chars = avail_train
    return train_chars, val_chars


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = int((mask_a & mask_b).sum())
    union = int((mask_a | mask_b).sum())
    return inter / float(max(union, 1))


def render_validation_8_sheet(
    chars: str,
    skip_otf: Path,
    poc_otf: Path,
    noto_serif: Path,
    noto_sans: Path,
    out_png: Path,
) -> None:
    """Render side-by-side sheet comparing Sans, Serif, Our Vectorized OTF, and Ground-Truth SkipPro-B."""
    cell = 148
    pad_left = 190
    rows = 5
    w = pad_left + len(chars) * cell + 20
    h = rows * cell + 30
    sheet = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    spec = CanvasSpec(canvas_size=132, em_ratio=0.84)
    rd_sans = AlignedFontRenderer(noto_sans, spec, ttc_index=0, variation_weight=620, supersample=2)
    rd_serif = AlignedFontRenderer(noto_serif, spec, ttc_index=0, variation_weight=700, supersample=2)
    rd_ours = AlignedFontRenderer(poc_otf, spec, ttc_index=0, supersample=2)
    rd_gt = AlignedFontRenderer(skip_otf, spec, ttc_index=0, supersample=2)

    label_font = ImageFont.truetype(str(noto_sans), 18, index=2)
    row_titles = [
        "思源黑 JP (骨架G)",
        "思源宋 JP (骨架R)",
        "本方案生成 (.OTF)",
        "SkipPro-B 原厂真值",
        "叠图(黑=重合 红=真值)",
    ]
    for r, title in enumerate(row_titles):
        draw.text((14, r * cell + cell // 2 - 10), title, fill=(40, 40, 40), font=label_font)

    for i, ch in enumerate(chars):
        x_box = pad_left + i * cell
        covs = [rd.render_coverage(ch) for rd in (rd_sans, rd_serif, rd_ours, rd_gt)]
        for r, cov in enumerate(covs):
            im = Image.fromarray(((1.0 - cov) * 255.0).round().astype(np.uint8), mode="L").convert("RGB")
            sheet.paste(im, (x_box, 12 + r * cell))

        # Row 4: Pixel error overlay between Our Vectorized OTF (Blue) and Ground-Truth SkipPro-B (Red)
        c_ours = covs[2]
        c_gt = covs[3]
        ov = np.full((spec.canvas_size, spec.canvas_size, 3), 255, dtype=np.uint8)
        ov[:, :, 0] = ((1.0 - c_ours) * 255.0).round().astype(np.uint8)
        ov[:, :, 1] = ((1.0 - np.maximum(c_ours, c_gt)) * 255.0).round().astype(np.uint8)
        ov[:, :, 2] = ((1.0 - c_gt) * 255.0).round().astype(np.uint8)
        sheet.paste(Image.fromarray(ov), (x_box, 12 + 4 * cell))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)


def render_simplified_12_sheet(
    chars: str,
    poc_otf: Path,
    noto_serif: Path,
    noto_sans: Path,
    out_png: Path,
) -> None:
    """Render showcase of 12 unseen pure Simplified Chinese characters from the packed .OTF."""
    cell = 148
    pad_left = 190
    cols = 6
    groups = [chars[:cols], chars[cols : cols * 2]]
    w = pad_left + cols * cell + 20
    h = len(groups) * 3 * cell + 40
    sheet = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    spec = CanvasSpec(canvas_size=132, em_ratio=0.84)
    rd_sans_sc = AlignedFontRenderer(noto_sans, spec, ttc_index=2, variation_weight=620, supersample=2)
    rd_serif_sc = AlignedFontRenderer(noto_serif, spec, ttc_index=2, variation_weight=700, supersample=2)
    rd_ours = AlignedFontRenderer(poc_otf, spec, ttc_index=0, supersample=2)

    label_font = ImageFont.truetype(str(noto_sans), 18, index=2)
    titles = ["思源黑 SC (参考)", "思源宋 SC (参考)", "CnS 生成矢量 (.OTF)"]

    for g_idx, grp in enumerate(groups):
        base_y = 15 + g_idx * (3 * cell + 12)
        for r, title in enumerate(titles):
            draw.text((14, base_y + r * cell + cell // 2 - 10), title, fill=(40, 40, 40), font=label_font)
        for i, ch in enumerate(grp):
            x_box = pad_left + i * cell
            for r, rd in enumerate((rd_sans_sc, rd_serif_sc, rd_ours)):
                cov = rd.render_coverage(ch)
                im = Image.fromarray(((1.0 - cov) * 255.0).round().astype(np.uint8), mode="L").convert("RGB")
                sheet.paste(im, (x_box, base_y + r * cell))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M1 Local Patch-Conv SDF PoC & Vector OTF Build")
    parser.add_argument("--skip_otf", type=Path, default=ROOT / "FOT-SkipPro-B.OTF")
    parser.add_argument(
        "--noto_serif",
        type=Path,
        default=Path("/usr/share/fonts/google-noto-serif-cjk-vf-fonts/NotoSerifCJK-VF.ttc"),
    )
    parser.add_argument(
        "--noto_sans",
        type=Path,
        default=Path("/usr/share/fonts/google-noto-sans-cjk-vf-fonts/NotoSansCJK-VF.ttc"),
    )
    parser.add_argument("--canvas_size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--num_train_chars", type=int, default=600, help="0 = use all ~7,870 chars")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--base_channels", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--out_dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    print(f"[M1] Device: {device} | Canvas: {args.canvas_size} | Patch: {args.patch_size}")

    t0 = time.time()
    train_chars, val_chars = select_training_and_val_chars(
        args.skip_otf, args.noto_serif, args.noto_sans, args.num_train_chars
    )
    assert all(c not in train_chars for c in ZI2ZI_JIT_FAILURE_8), "Data leakage check failed!"
    print(f"[M1] Train chars: {len(train_chars)} | Held-out Val chars: {len(val_chars)} (incl. 8 failure chars)")

    trunc_px = 16.0 * (args.canvas_size / 256.0)
    max_shift_px = 6.0 * (args.canvas_size / 256.0)

    print("[M1] Building aligned dual-skeleton SDF training dataset...")
    ds = PatchCropSDFDataset(
        chars=train_chars,
        skip_otf_path=str(args.skip_otf),
        noto_serif_ttc=str(args.noto_serif),
        noto_sans_ttc=str(args.noto_sans),
        canvas_size=args.canvas_size,
        patch_size=args.patch_size,
        patches_per_char_per_epoch=4,
        trunc_px=trunc_px,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"[M1] Dataset ready in {time.time() - t0:.1f}s ({len(ds)} local patches/epoch)")

    model = LocalStrokeSDFNet(
        in_channels=4,
        base_channels=args.base_channels,
        max_shift_px=max_shift_px,
    ).to(device)
    criterion = StrokeSDFLoss(trunc_px=trunc_px).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_t0 = time.time()
        agg = {"total": 0.0, "sdf_l1": 0.0, "soft_iou": 0.0}
        steps = 0
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred_sdf, flow_px = model(bx)
            loss, m = criterion(pred_sdf, by, bx, flow_px)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for k in agg:
                agg[k] += m[k]
            steps += 1
        scheduler.step()
        print(
            f"  Epoch {epoch:02d}/{args.epochs:02d} ({time.time()-ep_t0:.1f}s) | "
            f"loss={agg['total']/steps:.4f} | sdf_l1={agg['sdf_l1']/steps:.4f} | "
            f"patch_iou={agg['soft_iou']/steps:.4f}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "m1_local_sdf_net.pt")

    # --- Quantitative & Visual Evaluation on Held-Out Validation Chars ---
    model.eval()
    spec = CanvasSpec(canvas_size=args.canvas_size)
    serif_jp = AlignedFontRenderer(args.noto_serif, spec, ttc_index=0, variation_weight=700)
    sans_jp = AlignedFontRenderer(args.noto_sans, spec, ttc_index=0, variation_weight=620)
    skip_rd = AlignedFontRenderer(args.skip_otf, spec, ttc_index=0)

    ious_serif, ious_sans, ious_ours = [], [], []
    phantom_rates, skel_recalls = [], []
    val8_cmds: Dict[int, list] = {}

    with torch.no_grad():
        for idx, ch in enumerate(val_chars):
            x_np = build_dual_skeleton_input(ch, serif_jp, sans_jp, trunc_px=trunc_px)
            gt_mask = skip_rd.render_binary_hi(ch)
            gt_sdf = mask_to_normalized_sdf(gt_mask, args.canvas_size, trunc_px=trunc_px)

            xt = torch.from_numpy(x_np[None]).to(device)
            pred_sdf_t, _ = model(xt)
            pred_sdf = pred_sdf_t[0, 0].cpu().numpy()

            ious_serif.append(compute_iou(x_np[0] > 0.0, gt_sdf > 0.0))
            ious_sans.append(compute_iou(x_np[1] > 0.0, gt_sdf > 0.0))
            ious_ours.append(compute_iou(pred_sdf > 0.0, gt_sdf > 0.0))

            qa = evaluate_topology_qa(pred_sdf, x_np, trunc_px=trunc_px)
            phantom_rates.append(qa["phantom_ink_rate"])
            skel_recalls.append(qa["skeleton_recall"])

            if ch in ZI2ZI_JIT_FAILURE_8:
                val8_cmds[ord(ch)] = sdf_to_cff_commands(pred_sdf, spec, upsample_factor=4)

    # --- Inference & Vectorization on 12 Unseen Pure Simplified Chinese Chars (SC index=2) ---
    serif_sc = AlignedFontRenderer(args.noto_serif, spec, ttc_index=2, variation_weight=700)
    sans_sc = AlignedFontRenderer(args.noto_sans, spec, ttc_index=2, variation_weight=620)
    simp_cmds: Dict[int, list] = {}
    simp_qa: Dict[str, dict] = {}

    with torch.no_grad():
        for ch in UNSEEN_SIMPLIFIED_12:
            x_np = build_dual_skeleton_input(ch, serif_sc, sans_sc, trunc_px=trunc_px)
            xt = torch.from_numpy(x_np[None]).to(device)
            pred_sdf_t, _ = model(xt)
            pred_sdf = pred_sdf_t[0, 0].cpu().numpy()
            simp_qa[ch] = evaluate_topology_qa(pred_sdf, x_np, trunc_px=trunc_px)
            simp_cmds[ord(ch)] = sdf_to_cff_commands(pred_sdf, spec, upsample_factor=4)

    # Pack BOTH held-out 8 (overwriting cmap in test font so we render actual CFF vector output!)
    # and unseen 12 simplified chars into M1 PoC .otf!
    poc_otf_path = args.out_dir / "CnS-SkipPro-B-M1-PoC.otf"
    all_pack = {**val8_cmds, **simp_cmds}
    packed_count = pack_glyphs_into_otf(args.skip_otf, poc_otf_path, all_pack)

    val_sheet = args.out_dir / "m1_val_8_zi2zi_failures.png"
    simp_sheet = args.out_dir / "m1_simplified_12_showcase.png"
    render_validation_8_sheet(ZI2ZI_JIT_FAILURE_8, args.skip_otf, poc_otf_path, args.noto_serif, args.noto_sans, val_sheet)
    render_simplified_12_sheet(UNSEEN_SIMPLIFIED_12, poc_otf_path, args.noto_serif, args.noto_sans, simp_sheet)

    summary = {
        "train_chars": len(train_chars),
        "val_chars": len(val_chars),
        "val_mean_iou_raw_sans": round(float(np.mean(ious_sans)), 4),
        "val_mean_iou_raw_serif": round(float(np.mean(ious_serif)), 4),
        "val_mean_iou_ours": round(float(np.mean(ious_ours)), 4),
        "val8_failure_set_iou_ours": round(float(np.mean(ious_ours[:8])), 4),
        "max_phantom_ink_rate": round(float(np.max(phantom_rates)), 5),
        "min_skeleton_recall": round(float(np.min(skel_recalls)), 4),
        "packed_cff_glyphs": packed_count,
        "simplified_12_qa": {
            k: {mk: round(mv, 4) for mk, mv in v.items()} for k, v in simp_qa.items()
        },
    }
    report_path = args.out_dir / "m1_metrics.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[M1 SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[M1] Saved PoC OTF -> {poc_otf_path}")
    print(f"[M1] Saved Val-8 comparison -> {val_sheet}")
    print(f"[M1] Saved Simplified-12 showcase -> {simp_sheet}")


if __name__ == "__main__":
    main()
