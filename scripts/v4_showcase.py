#!/usr/bin/env python3
"""
Generate v4 vs v3 side-by-side showcase image.

Renders sample glyphs from both v3 and v4 output fonts and creates a comparison image.

Usage:
    python scripts/v4_showcase.py \
        --skip_otf FOT-SkipPro-B.OTF \
        --v3_missing outputs/CnS-SkipPro-B-v3-Missing.otf \
        --v4_missing outputs/CnS-SkipPro-B-v4-Missing.otf \
        --out_png outputs/v4_vs_v3_showcase.png \
        --n_chars 24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def render_char(font, ch: str, size: int = 128) -> np.ndarray:
    """Render a single character to a grayscale numpy array using a PIL font."""
    try:
        img = Image.new("L", (size, size), 255)
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), ch, font=font)
        if bbox[2] - bbox[0] < 2:
            return np.full((size, size), 255, dtype=np.uint8)
        x = (size - (bbox[2] - bbox[0])) // 2 - bbox[0]
        y = (size - (bbox[3] - bbox[1])) // 2 - bbox[1]
        draw.text((x, y), ch, font=font, fill=0)
        return np.array(img)
    except Exception:
        return np.full((size, size), 255, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="v4 vs v3 showcase")
    parser.add_argument("--noto_sans", type=Path, default=None)
    parser.add_argument("--skip_otf", type=Path, default=ROOT / "FOT-SkipPro-B.OTF")
    parser.add_argument("--v3_missing", type=Path, default=ROOT / "outputs" / "CnS-SkipPro-B-v3-Missing.otf")
    parser.add_argument("--v4_missing", type=Path, default=ROOT / "outputs" / "CnS-SkipPro-B-v4-Missing.otf")
    parser.add_argument("--out_png", type=Path, default=ROOT / "outputs" / "v4_vs_v3_showcase.png")
    parser.add_argument("--n_chars", type=int, default=24)
    parser.add_argument("--char_size", type=int, default=128)
    args = parser.parse_args()

    sample_chars = list("志洳菜嘉改栅畴逡红语银饭伞丌习乡卫专书东言直令骨微道龙丝")
    if len(sample_chars) > args.n_chars:
        sample_chars = sample_chars[: args.n_chars]

    cell = args.char_size
    label_h = 24
    pad = 8
    cols = len(sample_chars)

    # Determine rows
    has_native = args.noto_sans is not None and args.noto_sans.exists()
    has_v3 = args.v3_missing.exists()
    has_v4 = args.v4_missing.exists()

    rows = []
    if has_native:
        rows.append(("SkipPro-B (native)", "native"))
    if has_v3:
        rows.append(("v3 (Local U-Net, 2.8M)", "v3"))
    if has_v4:
        rows.append(("v4 (Transformer, 36M)", "v4"))

    n_rows = len(rows)
    if n_rows == 0:
        print("No fonts available to render. Exiting.")
        return

    img_w = cols * (cell + pad) + pad
    img_h = (n_rows + 1) * (cell + label_h + pad) + pad

    canvas = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Load fonts
    label_font_path = str(args.noto_sans) if has_native else str(args.skip_otf)
    label_font_idx = 2 if has_native else 0
    title_font = ImageFont.truetype(label_font_path, 16, index=label_font_idx)
    small_font = ImageFont.truetype(label_font_path, 11, index=label_font_idx)
    ch_font = ImageFont.truetype(label_font_path, 10, index=label_font_idx)

    char_font_native = None
    char_font_v3 = None
    char_font_v4 = None
    if has_native:
        char_font_native = ImageFont.truetype(str(args.noto_sans), cell, index=2)
    if has_v3:
        char_font_v3 = ImageFont.truetype(str(args.v3_missing), cell)
    if has_v4:
        char_font_v4 = ImageFont.truetype(str(args.v4_missing), cell)

    # Title
    title = "CnS v4 vs v3: Global Transformer SDF vs Local U-Net SDF"
    draw.text((pad, 2), title, font=title_font, fill=(0, 0, 0))

    for row_idx, (row_label, row_type) in enumerate(rows):
        y_offset = 28 + row_idx * (cell + label_h + pad)
        draw.text((pad, y_offset), row_label, font=small_font, fill=(80, 80, 80))

        for col, ch in enumerate(sample_chars):
            x = pad + col * (cell + pad)
            y = y_offset + label_h

            draw.rectangle([x, y, x + cell, y + cell], fill=(245, 245, 245), outline=(200, 200, 200))

            if row_type == "native" and char_font_native is not None:
                arr = render_char(char_font_native, ch, cell)
            elif row_type == "v3" and char_font_v3 is not None:
                arr = render_char(char_font_v3, ch, cell)
            elif row_type == "v4" and char_font_v4 is not None:
                arr = render_char(char_font_v4, ch, cell)
            else:
                arr = np.full((cell, cell), 255, dtype=np.uint8)

            cell_img = Image.fromarray(arr, "L").convert("RGB")
            canvas.paste(cell_img, (x, y))

            bbox_ch = draw.textbbox((0, 0), ch, font=ch_font)
            cx = x + cell // 2 - (bbox_ch[2] - bbox_ch[0]) // 2
            draw.text((cx, y + cell + 1), ch, font=ch_font, fill=(100, 100, 100))

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out_png)
    print(f"Showcase saved to {args.out_png}")
    print(f"  Size: {canvas.size[0]}x{canvas.size[1]}, {len(sample_chars)} chars, {n_rows} rows")


if __name__ == "__main__":
    main()
