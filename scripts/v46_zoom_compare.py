#!/usr/bin/env python3
"""
v4.6 single-glyph zoom comparator.

Renders each character as a large cell (default 300 px) side-by-side across several
OTF sources plus the Noto Sans SC reference (and optionally the SkipPro JP/traditional
style donor), with on-curve node overlay so vectorization quality (stem snapping,
node count, wobble, speckles) is visible at a glance.

Example:
  python scripts/v46_zoom_compare.py --noto_sans /usr/share/fonts/google-noto-sans-cjk-vf-fonts/NotoSansCJK-VF.ttc \
      --src v44=outputs/CnS-SkipPro-B-v44-Full.otf --src v45=outputs/CnS-SkipPro-B-v45-Full.otf \
      --chars 红纪约级纲纳纵线练组细终绍经 --out outputs/zoom_v44_v45_si.png --nodes
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2.hb_outlines import DrawCmd, SkipProOutlineExtractor, split_cmds_into_contours  # noqa: E402
from src.v3.hb_light import SansLightExtractor  # noqa: E402

PAD = 96.0


def _flatten(c_cmds: List[DrawCmd], step: float = 6.0) -> np.ndarray:
    pts: List[Tuple[float, float]] = []
    cur = (0.0, 0.0)
    for op, cp in c_cmds:
        if op == "moveTo":
            cur = cp[0]
            pts.append(cur)
        elif op == "lineTo":
            cur = cp[0]
            pts.append(cur)
        elif op == "curveTo":
            p0, p1, p2, p3 = (np.array(p, dtype=np.float64) for p in (cur, cp[0], cp[1], cp[2]))
            n = max(4, min(48, int(np.linalg.norm(p3 - p0) / step) + 2))
            for t in np.linspace(0, 1, n + 1)[1:]:
                u = 1 - t
                q = u ** 3 * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t ** 3 * p3
                pts.append((float(q[0]), float(q[1])))
            cur = cp[2]
    return np.array(pts, dtype=np.float64) if pts else np.zeros((0, 2))


VIEW = (-PAD, -120.0 - PAD, 1000.0 + PAD, 880.0 + PAD)  # xmin, ymin, xmax, ymax in UPEM


def render_cell(cmds: List[DrawCmd], size: int, nodes: bool, ss: int = 4) -> Image.Image:
    """Anti-aliased nonzero-winding render of DrawCmds with optional on-curve node overlay."""
    S = size * ss
    xmin, ymin, xmax, ymax = VIEW
    span = max(xmax - xmin, ymax - ymin)
    scale = S / span

    def to_px(p):
        return ((p[0] - xmin) * scale, (ymax - p[1]) * scale)

    mask = np.zeros((S, S), dtype=np.int32)
    contours = split_cmds_into_contours(cmds) if cmds else []
    for c in contours:
        poly = _flatten(c)
        if len(poly) < 3:
            continue
        x, y = poly[:, 0], poly[:, 1]
        area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        arr = np.round(np.array([to_px(p) for p in poly])).astype(np.int32)
        m = np.zeros((S, S), dtype=np.uint8)
        cv2.fillPoly(m, [arr], 1)
        mask += m.astype(np.int32) * (1 if area >= 0 else -1)
    ink = (mask != 0).astype(np.float32)
    ink = cv2.resize(ink, (size, size), interpolation=cv2.INTER_AREA)
    img = Image.fromarray((255 - ink * 255).clip(0, 255).astype(np.uint8)).convert("RGB")
    if nodes and contours:
        d = ImageDraw.Draw(img)
        sc2 = size / span
        for c in contours:
            for op, cp in c:
                if op in ("moveTo", "lineTo", "curveTo"):
                    p = cp[-1]
                    px, py = (p[0] - xmin) * sc2, (ymax - p[1]) * sc2
                    col = (230, 40, 40) if op == "curveTo" else (30, 120, 255)
                    d.ellipse([px - 2, py - 2, px + 2, py + 2], fill=col)
    return img


def count_nodes(cmds: List[DrawCmd]) -> Tuple[int, int]:
    n_curve = sum(1 for op, _ in cmds if op == "curveTo")
    n_line = sum(1 for op, _ in cmds if op == "lineTo")
    return n_line, n_curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noto_sans", type=Path, default=Path(os.environ.get("CNS_NOTO_SANS", "/usr/share/fonts/google-noto-sans-cjk-vf-fonts/NotoSansCJK-VF.ttc")))
    ap.add_argument("--noto_weight", type=int, default=300)
    ap.add_argument("--src", action="append", default=[], help="name=path.otf (repeatable)")
    ap.add_argument("--chars", type=str, required=True)
    ap.add_argument("--donor", type=str, default="", help="optional per-char SkipPro style donors, same length as --chars ('.' = none)")
    ap.add_argument("--skip_otf", type=Path, default=ROOT / "FOT-SkipPro-B.OTF")
    ap.add_argument("--cell", type=int, default=300)
    ap.add_argument("--nodes", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--view", type=str, default="", help="xmin,ymin,xmax,ymax UPEM crop (e.g. left radical: -40,-120,480,880)")
    args = ap.parse_args()
    global VIEW
    if args.view:
        VIEW = tuple(float(v) for v in args.view.split(","))

    sn_sc = SansLightExtractor(args.noto_sans, language="zh-Hans", weight=args.noto_weight)
    srcs: List[Tuple[str, SkipProOutlineExtractor]] = []
    for s in args.src:
        name, path = s.split("=", 1)
        srcs.append((name, SkipProOutlineExtractor(path)))
    skip = SkipProOutlineExtractor(args.skip_otf) if args.donor else None

    chars = list(args.chars)
    cols = ["Noto SC w%d" % args.noto_weight] + [n for n, _ in srcs] + (["SkipPro donor"] if skip else [])
    cell = args.cell
    head_h, label_h = 34, 26
    W = 70 + cell * len(cols)
    H = head_h + (cell + label_h) * len(chars)
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(str(args.noto_sans), 22)
        font.set_variation_by_axes([400])
    except Exception:
        font = ImageFont.load_default()

    for j, c in enumerate(cols):
        draw.text((70 + j * cell + 8, 6), c, fill=(0, 0, 0), font=font)

    for i, ch in enumerate(chars):
        y0 = head_h + i * (cell + label_h)
        draw.text((10, y0 + cell // 2 - 14), ch, fill=(0, 0, 0), font=font)
        cells: List[Tuple[List[DrawCmd], str]] = [(sn_sc.get_cmds(ch), "")]
        for name, ext in srcs:
            cm = ext.get_cmds(ch)
            nl, nc = count_nodes(cm)
            cells.append((cm, f"L{nl} C{nc}" if cm else "(absent)"))
        if skip:
            dch = args.donor[i] if i < len(args.donor) else "."
            cells.append((skip.get_cmds(dch) if dch != "." else [], dch if dch != "." else ""))
        for j, (cm, lab) in enumerate(cells):
            x0 = 70 + j * cell
            img = render_cell(cm, cell, nodes=args.nodes and j > 0)
            canvas.paste(img, (x0, y0))
            draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], outline=(200, 200, 200))
            # baseline / em guides
            if lab:
                draw.text((x0 + 6, y0 + cell + 2), lab, fill=(90, 90, 90), font=font)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"[v46-zoom] wrote {args.out} ({W}x{H})")


if __name__ == "__main__":
    main()
