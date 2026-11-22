"""
v3.0 Variant Classifier & Policy Loader (`Plan3.0.md §3`).

Implements:
  - `data_v3/variant_policy.tsv` loader with `HARD_KEEP_12` forced to `KEEP`.
  - V1 (`竖弯钩 <-> 竖+提`: 衣/衤/𧘇/表/...) & V2 (`ネ <-> 礻`: 礻/示) + policy-keep component exemption.
  - Marker-table demotion (`MAINLAND_VARIANT_MARKERS` nominates candidates only; gate + V1/V2 decides).
  - Full 8,105 TG partition into disjoint sets: `MISSING`, `SC_VARIANT`, and `KEEP_*`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.v2.component_bank import (
    _assign_islands_to_slots,
    cmds_to_islands,
    seq_has_invalid_entity,
    top_split,
)
from src.v2.hb_outlines import (
    MAINLAND_VARIANT_MARKERS,
    BBox,
    DrawCmd,
    SkipProOutlineExtractor,
    cmds_to_bbox,
)
from src.v3 import HARD_KEEP_CHARS
from src.v3.hb_light import SansLightExtractor
from src.v3.structural_gate import StructuralGateV3

V1_EXEMPT_COMPONENTS = set("衣衤𧘇表哀衰衷袁")
V2_EXEMPT_COMPONENTS = set("礻示")
ALL_EXEMPT_COMPONENTS = V1_EXEMPT_COMPONENTS | V2_EXEMPT_COMPONENTS | HARD_KEEP_CHARS


def load_variant_policy(policy_tsv: str | Path) -> Dict[str, str]:
    """Load `data_v3/variant_policy.tsv`, enforcing HARD_KEEP_12 as KEEP."""
    policy: Dict[str, str] = {c: "KEEP" for c in HARD_KEEP_CHARS}
    p = Path(policy_tsv)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("codepoint\t"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                ch, pol = parts[1].strip(), parts[2].strip().upper()
                if ch in HARD_KEEP_CHARS:
                    policy[ch] = "KEEP"
                elif pol in ("KEEP", "REPLACE", "AUTO"):
                    policy[ch] = pol
    return policy


def find_component_slot_boxes(
    ch: str,
    ids_map: Dict[str, str],
    sn_sc: SansLightExtractor,
    target_set: Set[str],
    want_exempt: bool = True,
) -> List[BBox]:
    """
    Estimate 1000-UPEM bounding boxes in `ch` for IDS slots that match (or contain)
    any char in `target_set` (`want_exempt=True`), OR slots that do NOT contain them (`want_exempt=False`).
    """
    seq = ids_map.get(ch)
    if not seq or seq_has_invalid_entity(seq):
        return []
    idc, comps = top_split(seq)
    if idc is None or idc == "⿻" or comps == [ch]:
        return []

    cs_unioned = sn_sc.get_cmds(ch, unioned=True)
    if not cs_unioned:
        return []
    full_bb = cmds_to_bbox(cs_unioned)
    if full_bb is None:
        return []
    gx0, gy0, gx1, gy1 = full_bb
    gw, gh = max(gx1 - gx0, 1.0), max(gy1 - gy0, 1.0)

    isls = cmds_to_islands(cs_unioned)
    slot_boxes: Optional[List[BBox]] = None
    if len(isls) >= len(comps):
        buckets = _assign_islands_to_slots(isls, idc, len(comps))
        if all(len(b) > 0 for b in buckets):
            boxes = []
            for b in buckets:
                boxes.append(
                    (
                        min(i.bbox[0] for i in b),
                        min(i.bbox[1] for i in b),
                        max(i.bbox[2] for i in b),
                        max(i.bbox[3] for i in b),
                    )
                )
            slot_boxes = boxes

    if slot_boxes is None:
        if idc == "⿰" and len(comps) == 2:
            s = gx0 + 0.37 * gw
            slot_boxes = [(gx0, gy0, s, gy1), (s, gy0, gx1, gy1)]
        elif idc == "⿲" and len(comps) == 3:
            w3 = gw / 3.0
            slot_boxes = [(gx0, gy0, gx0 + w3, gy1), (gx0 + w3, gy0, gx0 + 2 * w3, gy1), (gx0 + 2 * w3, gy0, gx1, gy1)]
        elif idc == "⿱" and len(comps) == 2:
            s = gy1 - 0.44 * gh
            slot_boxes = [(gx0, s, gx1, gy1), (gx0, gy0, gx1, s)]
        elif idc == "⿳" and len(comps) == 3:
            h3 = gh / 3.0
            slot_boxes = [(gx0, gy1 - h3, gx1, gy1), (gx0, gy0 + h3, gx1, gy1 - h3), (gx0, gy0, gx1, gy0 + h3)]
        elif idc == "⿺" and len(comps) == 2:
            slot_boxes = [(gx0, gy0, gx1, gy1), (gx0 + 0.25 * gw, gy0 + 0.20 * gh, gx1, gy1)]
        elif idc == "⿸" and len(comps) == 2:
            slot_boxes = [(gx0, gy0, gx1, gy1), (gx0 + 0.22 * gw, gy0, gx1, gy1 - 0.24 * gh)]
        else:
            return []

    out: List[BBox] = []
    for k, comp in enumerate(comps):
        has_hit = any(c in target_set for c in comp)
        if has_hit == want_exempt:
            out.append(slot_boxes[k])
    return out


@dataclass
class ClassificationResultV3:
    missing: List[str]
    sc_variant: List[str]
    keep_policy: List[str]
    keep_same_gid: List[str]
    keep_v1_v2: List[str]
    keep_already_sc: List[str]
    labels: Dict[str, str]
    exempt_boxes_map: Dict[str, List[BBox]]


def classify_tg8105_v3(
    tg_levels: Dict[str, int],
    skip_ext: SkipProOutlineExtractor,
    sn_jp: SansLightExtractor,
    sn_sc: SansLightExtractor,
    gate: StructuralGateV3,
    ids_map: Dict[str, str],
    policy_map: Dict[str, str],
) -> ClassificationResultV3:
    """
    Classify all 8,105 TG characters cleanly under Plan 3.0 rules:
      - `MISSING`: absent from `SkipPro-B`
      - `KEEP_*`: kept verbatim as Tier0_Native (excludes from BOTH Missing.otf and SCVariants.otf!)
      - `SC_VARIANT`: present in `SkipPro-B` with genuine non-exempt JP!=SC structural diff -> `SCVariants.otf`
    """
    missing: List[str] = []
    sc_variant: List[str] = []
    keep_policy: List[str] = []
    keep_same_gid: List[str] = []
    keep_v1_v2: List[str] = []
    keep_already_sc: List[str] = []
    labels: Dict[str, str] = {}
    exempt_boxes_map: Dict[str, List[BBox]] = {}

    for ch in tg_levels:
        if not skip_ext.has_char(ch):
            missing.append(ch)
            labels[ch] = "MISSING"
            continue

        pol = policy_map.get(ch, "AUTO")
        if pol == "KEEP":
            keep_policy.append(ch)
            labels[ch] = "KEEP_POLICY"
            continue
        if pol == "REPLACE":
            sc_variant.append(ch)
            labels[ch] = "SC_VARIANT"
            continue

        gj = sn_jp.shape_gid(ch, "ja")
        gs = sn_sc.shape_gid(ch, "zh-Hans")
        if gj == gs and gj > 0:
            keep_same_gid.append(ch)
            labels[ch] = "KEEP_SAME_GID"
            continue

        # Check V1 / V2 / HARD_KEEP_12 component exemptions
        seq = ids_map.get(ch, ch) + ch
        has_exempt_comp = any(c in ALL_EXEMPT_COMPONENTS for c in seq)
        ex_boxes = find_component_slot_boxes(ch, ids_map, sn_sc, set(ALL_EXEMPT_COMPONENTS), want_exempt=True) if has_exempt_comp else []

        if has_exempt_comp and ex_boxes:
            di_out = gate.analyze_jp_sc_diff(ch, exempt_boxes=ex_boxes)
            exempt_boxes_map[ch] = ex_boxes
            # Exempt only if no substantive diff remains outside the exempt component boxes
            if di_out["has_ref"] and di_out["dsc_px"] < 8 and di_out["djp_px"] < 8:
                keep_v1_v2.append(ch)
                labels[ch] = "KEEP_V1_V2"
                continue
        elif has_exempt_comp and all((c in ALL_EXEMPT_COMPONENTS or c in "⿰⿱⿲⿳") for c in ids_map.get(ch, ch)):
            keep_v1_v2.append(ch)
            labels[ch] = "KEEP_V1_V2"
            continue

        # Keep native if JP vs SC has zero structural skeleton diff outside exempt component boxes
        di = gate.analyze_jp_sc_diff(ch, exempt_boxes=ex_boxes or None)
        if di["has_ref"] and di["dsc_px"] < 8 and di["djp_px"] < 8:
            keep_v1_v2.append(ch)
            labels[ch] = "KEEP_V1_V2"
            continue

        # Even if `MAINLAND_VARIANT_MARKERS` nominates this char, only mark SC_VARIANT if
        # SkipPro-B's own glyph fails `StructuralGateV3` (and is not a trivial <12% callig shift).
        ok_orig, m_orig = gate.check(skip_ext.get_cmds(ch), ch, exempt_boxes=ex_boxes or None, check_jp_diff=True)
        if ok_orig or (
            di["has_ref"]
            and di["diff_ratio"] < 0.12
            and m_orig.get("chamfer", 99.0) <= 16.5
            and m_orig.get("p95", 99.0) <= 42.0
        ):
            keep_already_sc.append(ch)
            labels[ch] = "KEEP_SKIPPRO_ALREADY_SC"
        else:
            sc_variant.append(ch)
            labels[ch] = "SC_VARIANT"

    return ClassificationResultV3(
        missing=missing,
        sc_variant=sc_variant,
        keep_policy=keep_policy,
        keep_same_gid=keep_same_gid,
        keep_v1_v2=keep_v1_v2,
        keep_already_sc=keep_already_sc,
        labels=labels,
        exempt_boxes_map=exempt_boxes_map,
    )
