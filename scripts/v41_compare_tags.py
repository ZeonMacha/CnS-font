#!/usr/bin/env python3
"""
Side-by-side visual & quantitative comparison between any two CnS v4.x version tags.

Runs on the remote GPU/font host (or locally if Pillow & fontTools are installed).
Example:
  python scripts/v41_compare_tags.py --a v4 --b v41
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v3_merge_fonts import merge_split_fonts

DEFAULT_RADICAL_CHARS = "语饮汤绸坚劳览识银学驿乌袅仪轻车变锅"
DEFAULT_REGRESS_CHARS = "举领义护飞丝绸伞龙论"
DEFAULT_VARIANT_CHARS = "言直令骨微道反食鬼"


def _resolve_merged_font(tag: str, tmpd: Path) -> Path:
    full_p = ROOT / "outputs" / f"CnS-SkipPro-B-{tag}-Full.otf"
    if full_p.exists():
        return full_p
    miss_p = ROOT / "outputs" / f"CnS-SkipPro-B-{tag}-Missing.otf"
    var_p = ROOT / "outputs" / f"CnS-SkipPro-B-{tag}-SCVariants.otf"
    if not miss_p.exists() or not var_p.exists():
        raise FileNotFoundError(f"Missing OTFs for tag '{tag}': {miss_p} / {var_p}")
    out_p = tmpd / f"merged_{tag}.otf"
    merge_split_fonts(ROOT / "FOT-SkipPro-B.OTF", miss_p, var_p, out_p, subroutinize=False)
    return out_p


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two CnS v4.x font tags visually & quantitatively")
    parser.add_argument("--a", required=True, help="Baseline version tag (e.g. v4 or v41)")
    parser.add_argument("--b", required=True, help="Candidate version tag (e.g. v41 or v42)")
    parser.add_argument("--noto_sans", default="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    parser.add_argument("--chars", default="", help="Optional custom test string (appended as extra row)")
    parser.add_argument("--out_png", type=Path, default=None)
    args = parser.parse_args()

    out_png = args.out_png or (ROOT / "outputs" / f"compare_{args.a}_vs_{args.b}.png")

    ma = _load_json(ROOT / "outputs" / f"{args.a}_metrics.json")
    mb = _load_json(ROOT / "outputs" / f"{args.b}_metrics.json")
    man_a = _load_json(ROOT / "outputs" / f"{args.a}_glyph_manifest.json")
    man_b = _load_json(ROOT / "outputs" / f"{args.b}_glyph_manifest.json")

    # Quantitative summary
    print(f"\n=== Quantitative Comparison: {args.a} vs {args.b} ===")
    for k in ("missing_packed", "variants_packed", "blocked_to_review", "gate_pass_rate", "merged_tg_total_covered"):
        print(f"  {k:26s}: {args.a}={ma.get(k, '-')}  ->  {args.b}={mb.get(k, '-')}")

    check_chars = list(dict.fromkeys(DEFAULT_REGRESS_CHARS + DEFAULT_RADICAL_CHARS + args.chars))
    ch_a, ch_b, nd_a, nd_b = [], [], [], []
    for c in check_chars:
        da, db = man_a.get(c, {}), man_b.get(c, {})
        if isinstance(da.get("chamfer"), (int, float)) and isinstance(db.get("chamfer"), (int, float)):
            ch_a.append(float(da["chamfer"]))
            ch_b.append(float(db["chamfer"]))
        if isinstance(da.get("on_curve_nodes"), (int, float)) and isinstance(db.get("on_curve_nodes"), (int, float)):
            nd_a.append(float(da["on_curve_nodes"]))
            nd_b.append(float(db["on_curve_nodes"]))
    if ch_a:
        print(f"  sample_avg_chamfer (n={len(ch_a):2d}): {args.a}={sum(ch_a)/len(ch_a):.2f}  ->  {args.b}={sum(ch_b)/len(ch_b):.2f}")
    if nd_a:
        print(f"  sample_avg_nodes   (n={len(nd_a):2d}): {args.a}={sum(nd_a)/len(nd_a):.1f}  ->  {args.b}={sum(nd_b)/len(nd_b):.1f}")

    # Visual sheet
    groups: List[tuple[str, str]] = [
        ("简体偏旁/异体 (前9)", DEFAULT_RADICAL_CHARS[:9]),
        ("简体偏旁/异体 (后9)", DEFAULT_RADICAL_CHARS[9:]),
        ("重点回归 (10字)", DEFAULT_REGRESS_CHARS),
        ("陆标异体对照 (9字)", DEFAULT_VARIANT_CHARS),
    ]
    if args.chars.strip():
        groups.append(("自定义抽检", args.chars.strip()[:16]))

    with tempfile.TemporaryDirectory() as tmpd:
        td = Path(tmpd)
        fa_p = _resolve_merged_font(args.a, td)
        fb_p = _resolve_merged_font(args.b, td)

        f_ui = ImageFont.truetype(args.noto_sans, 26, index=2)
        f_lbl = ImageFont.truetype(args.noto_sans, 22, index=2)
        f_ref = ImageFont.truetype(args.noto_sans, 68, index=2)
        f_a = ImageFont.truetype(str(fa_p), 68)
        f_b = ImageFont.truetype(str(fb_p), 68)

        height = 90 + len(groups) * 290
        im = Image.new("RGB", (1660, height), (255, 255, 255))
        d = ImageDraw.Draw(im)
        d.text(
            (30, 18),
            f"CnS Font Comparison: {args.a} (red) vs {args.b} (black) | Merged with Orig SkipPro-B",
            fill=(20, 20, 30),
            font=f_ui,
        )

        y = 70
        for title, txt in groups:
            d.rectangle([28, y, 1630, y + 32], fill=(40, 46, 62))
            d.text((40, y + 3), title, fill=(248, 250, 255), font=f_lbl)
            y += 38
            for lbl, font, col in [
                ("Noto SC Ref", f_ref, (95, 100, 115)),
                (f"{args.a} Merged", f_a, (115, 35, 35)),
                (f"{args.b} Merged", f_b, (15, 20, 30)),
            ]:
                d.text((36, y + 20), lbl, fill=(70, 75, 90), font=f_lbl)
                d.text((230, y), txt, fill=col, font=font)
                y += 80
            y += 12

        out_png.parent.mkdir(parents=True, exist_ok=True)
        im.save(out_png)
        print(f"Saved comparison PNG -> {out_png} ({out_png.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
