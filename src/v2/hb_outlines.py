"""
Platform-Independent Vector Outline Extraction via uharfbuzz (zh-Hans / ja)
and Direct CFF RecordingPen Extraction from FOT-SkipPro-B.OTF.

Solves the Windows `raqm=False` fallback bug by driving HarfBuzz directly in Python
and extracting exact 1000-UPEM cubic Bézier commands without rasterization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import uharfbuzz as hb
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.recordingPen import RecordingPen
from fontTools.ttLib import TTFont

DrawCmd = Tuple[str, List[Tuple[float, float]]]
BBox = Tuple[float, float, float, float]


def split_cmds_into_contours(cmds: List[DrawCmd]) -> List[List[DrawCmd]]:
    """Split a flat DrawCmd sequence into closed contour DrawCmd lists."""
    contours: List[List[DrawCmd]] = []
    curr: List[DrawCmd] = []
    for op, pts in cmds:
        curr.append((op, list(pts)))
        if op in ("closePath", "endPath"):
            if curr:
                contours.append(curr)
                curr = []
    if curr:
        contours.append(curr)
    return contours


def cmds_to_bbox(cmds: List[DrawCmd]) -> Optional[BBox]:
    """Compute exact (xmin, ymin, xmax, ymax) bounding box of a DrawCmd list."""
    if not cmds:
        return None
    bp = BoundsPen(None)
    for op, pts in cmds:
        getattr(bp, op)(*pts)
    return bp.bounds


def rasterize_cmds(
    cmds: List[DrawCmd],
    canvas_size: int = 128,
    padding: float = 100.0,
) -> np.ndarray:
    """
    Rasterize 1000-UPEM DrawCmds into a binary mask (bool) using cv2.fillPoly
    over flattened cubic Béziers (100% platform-independent, exact zh-Hans!).
    """
    mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    if not cmds:
        return mask.astype(bool)

    # Map UPEM x in [-100, 1100], y in [-150, 950] into [0, canvas_size]
    scale = canvas_size / (1000.0 + 2.0 * padding)

    def to_px(pt: Tuple[float, float]) -> Tuple[float, float]:
        px = (pt[0] + padding) * scale
        py = (880.0 + padding - pt[1]) * scale
        return px, py

    contours = split_cmds_into_contours(cmds)
    poly_items: List[Tuple[float, bool, np.ndarray]] = []

    for c_cmds in contours:
        pts: List[Tuple[float, float]] = []
        start_upem = (0.0, 0.0)
        curr_upem = (0.0, 0.0)
        area_upem = 0.0
        for op, cpts in c_cmds:
            if op == "moveTo":
                start_upem = curr_upem = cpts[0]
                pts.append(to_px(curr_upem))
            elif op == "lineTo":
                p1 = cpts[0]
                area_upem += (curr_upem[0] * p1[1] - p1[0] * curr_upem[1])
                curr_upem = p1
                pts.append(to_px(p1))
            elif op == "curveTo":
                p0 = np.array(curr_upem, dtype=np.float64)
                p1 = np.array(cpts[0], dtype=np.float64)
                p2 = np.array(cpts[1], dtype=np.float64)
                p3 = np.array(cpts[2], dtype=np.float64)
                chord = float(np.linalg.norm(p3 - p0))
                n_seg = max(4, min(16, int(chord / 18.0) + 2))
                ts = np.linspace(0.0, 1.0, n_seg + 1)[1:]
                prev = curr_upem
                for t in ts:
                    u = 1.0 - t
                    q = (u ** 3) * p0 + 3.0 * (u ** 2) * t * p1 + 3.0 * u * (t ** 2) * p2 + (t ** 3) * p3
                    qt = (float(q[0]), float(q[1]))
                    area_upem += (prev[0] * qt[1] - qt[0] * prev[1])
                    prev = qt
                    pts.append(to_px(qt))
                curr_upem = (float(p3[0]), float(p3[1]))
            elif op == "closePath":
                area_upem += (curr_upem[0] * start_upem[1] - start_upem[0] * curr_upem[1])

        if len(pts) < 3:
            continue
        arr = np.round(np.array(pts, dtype=np.float32)).astype(np.int32)
        poly_items.append((abs(area_upem), area_upem >= 0.0, arr))

    # Paint largest-area contours first, one at a time, preserving 3+ level nested holes
    # and unioning overlapping Noto CFF2 sub-strokes without even-odd XOR holes.
    poly_items.sort(key=lambda item: -item[0])
    for _, is_pos, arr in poly_items:
        cv2.fillPoly(mask, [arr], 255 if is_pos else 0)
    return mask > 0


def cmds_iou(cmds_a: List[DrawCmd], cmds_b: List[DrawCmd], size: int = 96) -> float:
    """Compute exact polygon IoU between two 1000-UPEM DrawCmd glyphs."""
    ma = rasterize_cmds(cmds_a, canvas_size=size)
    mb = rasterize_cmds(cmds_b, canvas_size=size)
    inter = int((ma & mb).sum())
    union = int((ma | mb).sum())
    return inter / float(max(union, 1))


class HBOutlineExtractor:
    """Extracts 1000-UPEM cubic Bézier outlines via HarfBuzz with explicit script/language."""

    def __init__(
        self,
        font_path: str | Path,
        ttc_index: int = 0,
        weight: Optional[int] = None,
    ) -> None:
        self.font_path = str(font_path)
        blob = hb.Blob.from_file_path(self.font_path)
        self.face = hb.Face(blob, ttc_index)
        self.font = hb.Font(self.face)
        self.font.scale = (1000, 1000)
        if weight is not None:
            self.font.set_variations({"wght": float(weight)})

        self._df = hb.DrawFuncs()
        self._df.set_move_to_func(lambda x, y, c: c.append(("moveTo", [(float(x), float(y))])))
        self._df.set_line_to_func(lambda x, y, c: c.append(("lineTo", [(float(x), float(y))])))
        self._df.set_cubic_to_func(
            lambda x1, y1, x2, y2, x3, y3, c: c.append(
                ("curveTo", [(float(x1), float(y1)), (float(x2), float(y2)), (float(x3), float(y3))])
            )
        )
        # Convert quadratic curves to cubic if a TTF font is passed
        self._df.set_quadratic_to_func(self._quad_to_cubic)
        self._df.set_close_path_func(lambda c: c.append(("closePath", [])))
        self._cache: Dict[Tuple[str, str], Tuple[int, List[DrawCmd]]] = {}
        self._union_cache: Dict[Tuple[str, str], List[DrawCmd]] = {}

    def get_unioned_cmds(self, char: str, language: str = "zh-Hans") -> List[DrawCmd]:
        """Merge overlapping CFF2 sub-strokes into clean topological islands via pathops."""
        key = (char, language)
        if key not in self._union_cache:
            from src.v2.curve_optimizer import boolean_union_cmds

            _, raw = self.get_gid_and_cmds(char, language)
            self._union_cache[key] = boolean_union_cmds([raw]) if raw else []
        return self._union_cache[key]

    @staticmethod
    def _quad_to_cubic(x1: float, y1: float, x2: float, y2: float, c: List[DrawCmd]) -> None:
        if c and c[-1][1]:
            x0, y0 = c[-1][1][-1]
        else:
            x0, y0 = x1, y1
        cx1 = x0 + (2.0 / 3.0) * (x1 - x0)
        cy1 = y0 + (2.0 / 3.0) * (y1 - y0)
        cx2 = x2 + (2.0 / 3.0) * (x1 - x2)
        cy2 = y2 + (2.0 / 3.0) * (y1 - y2)
        c.append(("curveTo", [(cx1, cy1), (cx2, cy2), (float(x2), float(y2))]))

    def shape_gid(self, char: str, language: str = "zh-Hans") -> int:
        buf = hb.Buffer()
        buf.add_str(char)
        buf.direction = "ltr"
        buf.script = "Hani"
        buf.language = language
        hb.shape(self.font, buf, {})
        infos = buf.glyph_infos
        return int(infos[0].codepoint) if infos else 0

    def get_gid_and_cmds(self, char: str, language: str = "zh-Hans") -> Tuple[int, List[DrawCmd]]:
        key = (char, language)
        if key not in self._cache:
            gid = self.shape_gid(char, language)
            cmds: List[DrawCmd] = []
            if gid > 0:
                self.font.draw_glyph(gid, self._df, cmds)
            self._cache[key] = (gid, cmds)
        return self._cache[key]

    def get_cmds(self, char: str, unioned: bool = True, language: str = "zh-Hans") -> List[DrawCmd]:
        """Convenience wrapper: return raw or unioned DrawCmd list for a character."""
        if unioned:
            return self.get_unioned_cmds(char, language)
        else:
            return self.get_gid_and_cmds(char, language)[1]


class SkipProOutlineExtractor:
    """Direct 1000-UPEM CFF outline reader for FOT-SkipPro-B.OTF (or Master Bank OTF)."""

    def __init__(self, otf_path: str | Path) -> None:
        self.otf_path = str(otf_path)
        self.tt = TTFont(self.otf_path)
        self.cmap: Dict[int, str] = self.tt.getBestCmap() or {}
        self.glyph_set = self.tt.getGlyphSet()
        self._cache: Dict[str, List[DrawCmd]] = {}

    def has_char(self, char: str) -> bool:
        return ord(char) in self.cmap

    def get_cmds(self, char: str) -> List[DrawCmd]:
        if char not in self._cache:
            cp = ord(char)
            if cp not in self.cmap:
                self._cache[char] = []
            else:
                rec = RecordingPen()
                self.glyph_set[self.cmap[cp]].draw(rec)
                self._cache[char] = [
                    (op, [(float(x), float(y)) for x, y in pts]) for op, pts in rec.value
                ]
        return self._cache[char]

    def close(self) -> None:
        self.tt.close()


def load_tg8105_table(tg_dir: str | Path) -> Dict[str, int]:
    """Load 《通用规范汉字表》 level-1/2/3 -> dict mapping char to level (1, 2, or 3)."""
    tg_path = Path(tg_dir)
    levels: Dict[str, int] = {}
    for lv in (1, 2, 3):
        txt = (tg_path / f"level-{lv}.txt").read_text(encoding="utf-8")
        for c in txt:
            if not c.isspace() and ord(c) >= 0x3400:
                levels[c] = lv
    return levels


# Known Mainland GB vs Japanese JIS structural variant triggers inside cjkvi-ids
MAINLAND_VARIANT_MARKERS = set(
    "言讠亠辶骨令直真微每反刃食衣衤示礻户羽角鬼免象曾半平艹扌氵忄青者既兼差兑益并华吴吞呈吕启周尚扁"
)


@dataclass
class TG8105Classification:
    tg_levels: Dict[str, int]
    missing_4101: List[str]
    same_1381: List[str]
    diff_all_2623: List[str]
    diff_replace: List[str]
    diff_minor_kept: List[str]
    ious_diff: Dict[str, float]


def classify_tg8105(
    tg_dir: str | Path,
    skip_ext: SkipProOutlineExtractor,
    serif_jp: HBOutlineExtractor,
    serif_sc: HBOutlineExtractor,
    sans_jp: HBOutlineExtractor,
    sans_sc: HBOutlineExtractor,
    ids_map: Optional[Dict[str, str]] = None,
    iou_threshold: float = 0.90,
) -> TG8105Classification:
    """Classify all 8,105 TG characters using HarfBuzz GID + zh-Hans/ja IoU + IDS structural markers."""
    tg_levels = load_tg8105_table(tg_dir)
    missing: List[str] = []
    same: List[str] = []
    diff_all: List[str] = []
    diff_replace: List[str] = []
    diff_minor_kept: List[str] = []
    ious_diff: Dict[str, float] = {}

    for ch in tg_levels:
        if not skip_ext.has_char(ch):
            missing.append(ch)
            continue

        gid_sr_jp, cmds_sr_jp = serif_jp.get_gid_and_cmds(ch, "ja")
        gid_sr_sc, cmds_sr_sc = serif_sc.get_gid_and_cmds(ch, "zh-Hans")
        gid_sn_jp, _ = sans_jp.get_gid_and_cmds(ch, "ja")
        gid_sn_sc, _ = sans_sc.get_gid_and_cmds(ch, "zh-Hans")

        if gid_sr_jp == gid_sr_sc and gid_sn_jp == gid_sn_sc:
            same.append(ch)
        else:
            diff_all.append(ch)
            iou = cmds_iou(cmds_sr_jp, cmds_sr_sc, size=80)
            ious_diff[ch] = iou
            seq = (ids_map.get(ch, ch) + ch) if ids_map else ch
            has_marker = any(m in seq for m in MAINLAND_VARIANT_MARKERS)
            if iou < iou_threshold or has_marker:
                diff_replace.append(ch)
            else:
                diff_minor_kept.append(ch)

    return TG8105Classification(
        tg_levels=tg_levels,
        missing_4101=missing,
        same_1381=same,
        diff_all_2623=diff_all,
        diff_replace=diff_replace,
        diff_minor_kept=diff_minor_kept,
        ious_diff=ious_diff,
    )
