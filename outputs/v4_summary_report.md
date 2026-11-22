# CnS v4.0: Global-Context Transformer SDF Font Synthesis — Final Report

## Overview

v4.0 replaces the v3 local Conv U-Net (~2.8M params) with a **CNN + Transformer hybrid architecture** (~36M params) that captures full 256×256 glyph context via self-attention. The model is trained on 7,898 TG-8105 characters using a two-stage curriculum (freeze CNN → unfreeze all) and achieves significant quality improvements over v3.

## Architecture

**GlobalGlyphTransformer** (v4-lite):
- **CNN Encoder**: 4-stage downsampling (4→48@128→96@64→192@32→384@16) with GroupNorm + SiLU
- **Patchify**: 1×1 conv (384→768) producing 16×16 = 256 tokens
- **Positional Encoding**: 2D Sincos (fixed, non-learnable)
- **Transformer**: 4 layers, d_model=768, n_heads=12, pre-norm
- **Unpatchify**: 1×1 conv (768→384)
- **CNN Decoder**: 4-stage upsampling with skip connections
- **SDF Head**: Bounded flow warp + convex blend + active gate (same as v3)

**Parameters**: 35,988,868 (36.0M)
**VRAM**: 3.8 GB (batch=16, 256×256 canvas)

## Training

| Stage | Epochs | LR | Strategy | Duration |
|-------|--------|----|----------|----------|
| Stage 1 | 30 | 5e-5 (cosine decay) | Freeze CNN, train Transformer | ~25 min |
| Stage 2 | 15 | 2e-5 (cosine decay) | Unfreeze all, fine-tune | ~18 min |

**Total training time**: ~43 min on RTX 4090 (24GB)
**Dataset**: 7,898 chars (TG-8105 minus 14 holdout), 256×256 full glyphs
**Augmentation**: dw jitter, r_u jitter, ±2° rotation, 0.95–1.05 scale

### Loss Progression

| Metric | Stage 1 Final | Stage 2 Final | Improvement |
|--------|--------------|--------------|-------------|
| SDF L1 | 0.8541 | 0.3098 | -64% |
| Soft IoU | 0.6004 | 0.7496 | +25% |
| clDice | 0.1775 | 0.0447 | -75% |
| Grad L1 | 0.0219 | 0.0135 | -38% |

## Holdout Evaluation (14 chars)

| Metric | v4 | v3 (baseline) | Change |
|--------|-----|--------------|--------|
| SDF L1 (mean) | 0.0736 | ~0.35 (est.) | -79% |
| Soft IoU (mean) | 0.8227 | ~0.74 (est.) | +11% |
| Skeleton Chamfer (UPEM) | 9.25 | 14.52 | -36% |
| Contour Hausdorff (UPEM) | 126.63 | 137.71 | -8% |

**Note**: v3 baseline numbers from `v3_m2_ablation_report.json` (80-char eval). v4 eval is on 14 holdout chars (subset not in training).

### Per-Char Highlights

| Char | SDF L1 | IoU | Chamfer | Hausdorff |
|------|--------|-----|---------|-----------|
| 栅 | 0.0562 | 0.899 | 5.81 | 120.78 |
| 菜 | 0.0607 | 0.868 | 5.83 | 77.00 |
| 洳 | 0.0710 | 0.864 | 7.01 | 164.41 |
| 言 | 0.0935 | 0.694 | 14.95 | 135.55 |

## TG-8105 Build Results

| Metric | Value |
|--------|-------|
| Total TG-8105 chars | 8,105 |
| Missing targets | 4,101 |
| Missing packed | 4,100 (99.98%) |
| Variants packed | 0 |
| Blocked to review | 1 |
| Gate pass rate | 99.98% |
| Tier 2 (Atomic) | 1 |
| Tier 3 (Global SDF) | 4,099 |
| Build time | 699.7s (~12 min) |

### Invariants (all passed)
- [x] Missing/variants disjoint
- [x] Missing/base disjoint
- [x] Variants subset of base
- [x] Hard-keep 12 point-exact

## Output Files

| File | Description |
|------|-------------|
| `outputs/CnS-SkipPro-B-v4-Missing.otf` | 4,100 synthesized missing glyphs |
| `outputs/CnS-SkipPro-B-v4-SCVariants.otf` | SC variant glyphs (0 in this run) |
| `outputs/v4_glyph_manifest.json` | Per-glyph manifest with tier & metrics |
| `outputs/v4_metrics.json` | Build summary metrics |
| `outputs/v4_eval_report.json` | Holdout evaluation details |
| `checkpoints/v4_zh_sdf_global.pt` | Final model checkpoint (36M params) |

## Known Limitations

1. **ids.txt missing**: The CJK decomposition table was not available, so Tier 1 (compose from components) had no data. Only Tier 2 (1 char) and Tier 3 (4,099 chars) were used.
2. **No leak ratio**: Noto Sans CJK is non-variable (no wght=620), so Sans 620 leak ratio could not be measured.
3. **SC variants = 0**: The variant classifier found 0 SC variants in this run (possibly due to missing ids.txt affecting classification).
4. **1 char blocked**: 1 character failed the structural gate and was routed to manual review.

## Comparison: v4 vs v3 Architecture

| Aspect | v3 (Local U-Net) | v4 (Global Transformer) |
|--------|-----------------|------------------------|
| Parameters | 2.8M | 36.0M |
| Context | 96×96 random crops | 256×256 full glyph |
| Attention | None (local conv only) | 4-layer self-attention (256 tokens) |
| VRAM | ~1.5 GB | ~3.8 GB |
| SDF L1 (holdout) | ~0.35 (est.) | 0.0736 |
| IoU (holdout) | ~0.74 (est.) | 0.8227 |
| Chamfer (UPEM) | 14.52 | 9.25 |
| Build time | ~10 min | ~12 min |

## Files Modified/Created

### New (v4)
- `src/v4/__init__.py`
- `src/v4/model_v4.py` — GlobalGlyphTransformer
- `src/v4/dataset_v4.py` — FullGlyphDatasetV4
- `src/v4/losses_v4.py` — Composite loss + OCR head
- `src/v4/synthesizer_v4.py` — ZHSDFSynthesizerV4
- `scripts/v4_train.py` — Two-stage training
- `scripts/v4_deploy.py` — SCP-based deployment
- `scripts/v4_vram_test.py` — VRAM benchmark
- `scripts/v4_scale_test.py`, `v4_scale_test2.py` — Scale testing
- `scripts/v4_evaluate.py` — Holdout evaluation
- `scripts/v4_build_tg8105.py` — Full TG8105 build
- `scripts/v4_check_font.py` — Font inspection
- `scripts/v4_test_dataset.py` — Dataset test
- `scripts/v4_debug_pe.py`, `v4_debug_full.py` — Debug scripts
- `scripts/v4_download_ids.py` — ids.txt downloader

### Modified
- `src/v2/hb_outlines.py` — Added `get_cmds()` convenience method to `HBOutlineExtractor`

## Next Steps (Optional)

1. **Scale to v4-full (8L, d=1024, ~79M params)**: VRAM headroom confirmed (6.8 GB at batch=16). Would improve quality further.
2. **OCR feedback loop**: Deploy PaddleOCR PP-OCRv4 on RTX 3060 for automated glyph quality verification.
3. **Re-run with proper ids.txt**: Download the full CJK decomposition table to enable Tier 1 composition.
4. **Generate showcase image**: Side-by-side v3 vs v4 rendering of sample glyphs for visual comparison.
