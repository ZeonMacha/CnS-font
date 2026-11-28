"""CnS v4.6: Dual-dataset training, local-attention+uncertainty architecture, 384px QA gate & multi-sample selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.v46.vectorize_v46 import (
    VecParams,
    VecStats,
    cmds_to_polylines,
    polylines_to_cmds,
    sdf_iso_contours_upem,
    sdf_to_cmds_v46,
)

if TYPE_CHECKING:
    from src.v46.dataset_v46 import DualGlyphDatasetV46, create_v46_dataloaders
    from src.v46.glyph_qa_v46 import GlyphQAGateV46, QAConfig, QAResult, repair_phantom_islands_v46
    from src.v46.losses_v46 import GlobalGlyphLossV46, compute_sdf_fog_ratio
    from src.v46.model_v46 import GlobalGlyphTransformerV46, load_v45_weights_into_v46
    from src.v46.sampler_v46 import NeuralSDFSamplerV46, select_best_candidate_v46
    from src.v46.synthesizer_v46 import ZHSDFSynthesizerV46

_LAZY_IMPORTS = {
    "DualGlyphDatasetV46": ("src.v46.dataset_v46", "DualGlyphDatasetV46"),
    "create_v46_dataloaders": ("src.v46.dataset_v46", "create_v46_dataloaders"),
    "GlyphQAGateV46": ("src.v46.glyph_qa_v46", "GlyphQAGateV46"),
    "QAConfig": ("src.v46.glyph_qa_v46", "QAConfig"),
    "QAResult": ("src.v46.glyph_qa_v46", "QAResult"),
    "repair_phantom_islands_v46": ("src.v46.glyph_qa_v46", "repair_phantom_islands_v46"),
    "GlobalGlyphLossV46": ("src.v46.losses_v46", "GlobalGlyphLossV46"),
    "compute_sdf_fog_ratio": ("src.v46.losses_v46", "compute_sdf_fog_ratio"),
    "GlobalGlyphTransformerV46": ("src.v46.model_v46", "GlobalGlyphTransformerV46"),
    "load_v45_weights_into_v46": ("src.v46.model_v46", "load_v45_weights_into_v46"),
    "NeuralSDFSamplerV46": ("src.v46.sampler_v46", "NeuralSDFSamplerV46"),
    "select_best_candidate_v46": ("src.v46.sampler_v46", "select_best_candidate_v46"),
    "ZHSDFSynthesizerV46": ("src.v46.synthesizer_v46", "ZHSDFSynthesizerV46"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        mod_name, attr_name = _LAZY_IMPORTS[name]
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "VecParams",
    "VecStats",
    "sdf_to_cmds_v46",
    "sdf_iso_contours_upem",
    "polylines_to_cmds",
    "cmds_to_polylines",
    *list(_LAZY_IMPORTS.keys()),
]
