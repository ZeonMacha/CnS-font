"""
v4.5 Decoupled Structural Gate (`StructuralGateV45`).

Fixes over v3/v4.3 `StructuralGateV3` (from 审计v4.md §2.3 #3 & Phase A #3):
  1. Smooth linear `p95_limit` ramp between `chamfer=26.0` (96.0u) and `chamfer=32.0` (72.3u)
     when `miss_sc <= 0.18` and `miss_cand <= 0.18` — eliminates the sharp `chamfer=28.0` cliff
     that falsely rejected `孖 (ch=28.02, p95=87.0)`.
  2. Completely decoupled from `variant_classifier.py`: classification continues to use
     an unmodified baseline `StructuralGateV3` so `KEEP_ALREADY_SC` never silently drifts.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.v2.hb_outlines import BBox, DrawCmd
from src.v3.hb_light import SansLightExtractor
from src.v3.structural_gate import StructuralGateV3


class StructuralGateV45(StructuralGateV3):
    """Smooth-ramp synthesis acceptance gate decoupled from variant classification."""

    def __init__(
        self,
        sn_jp: SansLightExtractor,
        sn_sc: SansLightExtractor,
        canvas_size: int = 160,
        padding: float = 80.0,
        tau_diff_upem: float = 22.0,
    ) -> None:
        super().__init__(sn_jp, sn_sc, canvas_size=canvas_size, padding=padding, tau_diff_upem=tau_diff_upem)

    @staticmethod
    def smooth_p95_limit(chamfer: float, miss_sc: float, miss_cand: float) -> float:
        """Continuous p95 ceiling avoiding the v4.3 chamfer==28.0 cliff."""
        if miss_sc > 0.18 or miss_cand > 0.18:
            return 72.3
        if chamfer <= 26.0:
            return 96.0
        if chamfer >= 32.0:
            return 72.3
        t = (chamfer - 26.0) / 6.0
        return 96.0 * (1.0 - t) + 72.3 * t

    def check(
        self,
        cand_cmds: List[DrawCmd],
        ch: str,
        exempt_boxes: Optional[Sequence[BBox]] = None,
        replaced_boxes: Optional[Sequence[BBox]] = None,
        check_jp_diff: bool = True,
    ) -> Tuple[bool, dict]:
        ok, gm = super().check(
            cand_cmds, ch,
            exempt_boxes=exempt_boxes,
            replaced_boxes=replaced_boxes,
            check_jp_diff=check_jp_diff,
        )
        if ok or "reasons" not in gm:
            return ok, gm

        # Re-evaluate ONLY p95 failure using the continuous ramp (no cliff at 28.0u)
        reasons = list(gm.get("reasons", []))
        p95_reasons = [r for r in reasons if r.startswith("p95=")]
        if p95_reasons and len(p95_reasons) == len(reasons):
            ch_v = float(gm.get("chamfer", 99.0))
            p95_v = float(gm.get("p95", 999.0))
            ms = float(gm.get("miss_sc", 1.0))
            mc = float(gm.get("miss_cand", 1.0))
            lim = self.smooth_p95_limit(ch_v, ms, mc)
            if p95_v <= lim:
                gm = dict(gm)
                gm["passed"] = True
                gm["reasons"] = []
                gm["p95_limit_used"] = round(lim, 2)
                return True, gm
        return ok, gm
