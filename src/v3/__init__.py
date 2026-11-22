"""CnS v3.0: Deskinned Sans Light reference, Structural Gate, Local Inpainting, and Split Packaging."""

from __future__ import annotations

from typing import Dict

HARD_KEEP_12: Dict[int, str] = {
    0x4E10: "丐",
    0x4E30: "丰",
    0x514D: "免",
    0x51AB: "冫",
    0x534A: "半",
    0x6BCF: "每",
    0x793B: "礻",
    0x793E: "社",
    0x795E: "神",
    0x8279: "艹",
    0x8863: "衣",
    0x8864: "衤",
}

HARD_KEEP_CHARS = frozenset(HARD_KEEP_12.values())


def is_hard_keep(ch: str) -> bool:
    """Return True if `ch` is one of the 12 human-verified characters forced native."""
    return ch in HARD_KEEP_CHARS
