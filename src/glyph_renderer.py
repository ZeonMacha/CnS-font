"""
Exact 1000-UPEM Baseline-Locked Glyph Renderer & Font Character Set Inspector.

Resolves the 25px vertical offset trap between Noto Sans/Serif CJK (ascender=1160)
and FOT-SkipPro-B.OTF (ascender=880) by anchoring every glyph at OpenType (0, 0)
left-baseline (`anchor="ls"`), ensuring deterministic pixel <-> 1000-UPEM round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class CanvasSpec:
    """Deterministic mapping between OpenType 1000-UPEM space and S x S pixel canvas."""

    canvas_size: int = 256
    em_ratio: float = 0.84  # 1000 UPEM occupies 84% of canvas, leaving 8% margin on each side
    upem: int = 1000
    cjk_ascender: int = 880  # Standard Adobe/CJK ideographic em-box top ([-120, +880])

    @property
    def em_px(self) -> float:
        return self.canvas_size * self.em_ratio

    @property
    def x_origin(self) -> float:
        """Canvas X pixel corresponding to OpenType X = 0."""
        return (self.canvas_size - self.em_px) * 0.5

    @property
    def y_baseline(self) -> float:
        """Canvas Y pixel corresponding to OpenType Y = 0 baseline."""
        top_margin = (self.canvas_size - self.em_px) * 0.5
        return top_margin + (self.cjk_ascender / self.upem) * self.em_px

    def canvas_to_upem(self, px: np.ndarray, py: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert (px, py) pixel coordinates (col, row) back to OpenType (X, Y) in 1000 UPEM."""
        scale = self.upem / self.em_px
        x_upem = (px - self.x_origin) * scale
        y_upem = (self.y_baseline - py) * scale
        return x_upem, y_upem


class AlignedFontRenderer:
    """Renders glyphs with exact 1000-UPEM coordinate locking across CJK fonts."""

    def __init__(
        self,
        font_path: str | Path,
        spec: CanvasSpec,
        ttc_index: int = 0,
        variation_weight: Optional[int] = None,
        supersample: int = 2,
    ) -> None:
        self.font_path = str(font_path)
        self.spec = spec
        self.ttc_index = ttc_index
        self.supersample = supersample

        render_px = int(round(spec.em_px * supersample))
        self.pil_font = ImageFont.truetype(self.font_path, size=render_px, index=ttc_index)
        if variation_weight is not None:
            try:
                self.pil_font.set_variation_by_axes([variation_weight])
            except Exception:
                pass

    def render_coverage(self, char: str) -> np.ndarray:
        """
        Render character as float32 ink coverage in [0.0, 1.0] (1.0 = black ink, 0.0 = background)
        aligned strictly to OpenType (0, 0) baseline.
        """
        ss = self.supersample
        s_hi = self.spec.canvas_size * ss
        img = Image.new("L", (s_hi, s_hi), 0)
        draw = ImageDraw.Draw(img)
        x0 = self.spec.x_origin * ss
        y0 = self.spec.y_baseline * ss
        draw.text((x0, y0), char, fill=255, font=self.pil_font, anchor="ls")
        if ss > 1:
            img = img.resize(
                (self.spec.canvas_size, self.spec.canvas_size),
                resample=Image.Resampling.BOX,
            )
        return np.asarray(img, dtype=np.float32) / 255.0

    def render_binary_hi(self, char: str) -> np.ndarray:
        """Render high-resolution supersampled binary mask (bool) for exact Euclidean SDF."""
        ss = self.supersample
        s_hi = self.spec.canvas_size * ss
        img = Image.new("L", (s_hi, s_hi), 0)
        draw = ImageDraw.Draw(img)
        x0 = self.spec.x_origin * ss
        y0 = self.spec.y_baseline * ss
        draw.text((x0, y0), char, fill=255, font=self.pil_font, anchor="ls")
        return np.asarray(img, dtype=np.uint8) >= 128


def get_font_cjk_codepoints(font_path: str | Path, ttc_index: int = 0) -> Set[int]:
    """Return mapped CJK Unified Ideograph codepoints (U+4E00..U+9FFF) in the font."""
    tt = TTFont(str(font_path), fontNumber=ttc_index if str(font_path).lower().endswith(".ttc") else -1)
    cmap = tt.getBestCmap() or {}
    cjk = {cp for cp in cmap.keys() if 0x4E00 <= cp <= 0x9FFF}
    tt.close()
    return cjk


def get_gb2312_sets() -> Tuple[Set[int], Set[int]]:
    """Return (gb2312_level1_3755, gb2312_all_6763) Unicode codepoint sets."""
    l1: Set[int] = set()
    all_gb: Set[int] = set()
    for b1 in range(0xB0, 0xF8):
        for b2 in range(0xA1, 0xFF):
            try:
                cp = ord(bytes([b1, b2]).decode("gb2312"))
            except Exception:
                continue
            all_gb.add(cp)
            if b1 < 0xD7 or (b1 == 0xD7 and b2 <= 0xF9):
                l1.add(cp)
    return l1, all_gb
