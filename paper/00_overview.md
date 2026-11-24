# Glyph Style Transfer via Global-Context Signed Distance Field Generation for CJK Font Localization

**CnS v4.4 — Technical Report**

> **Status**: Draft — arXiv-style technical report
> **Code**: [`CnS-font`](../) repository, branch `new-achi-v4-t`
> **Model**: `GlobalGlyphTransformer` (36,009,267 parameters), `decoder_version = v44`
> **Checkpoint**: [`checkpoints/v44_zh_sdf_global.pt`](../checkpoints/v44_zh_sdf_global.pt)

---

## Document Structure

This report is organized into eight independently-writable section files. Each file can be read standalone or combined into a single arXiv-format manuscript.

| File | Section | Content |
|:---|:---|:---|
| `00_overview.md` | — | Title, abstract, document map, notation |
| `01_intro.md` | 1 | Introduction, motivation (game font localization), related work |
| `02_problem.md` | 2 | Problem formulation: CJK glyph coverage, variant classification, data pipeline |
| `03_method.md` | 3 | Method — `GlobalGlyphTransformer` architecture (v4.4 direct SDF head) |
| `04_losses.md` | 4 | Method — composite loss function design |
| `05_pipeline.md` | 5 | Method — inference pipeline, structural gate, vectorization & post-processing |
| `06_experiments.md` | 6 | Experiments: training protocol, full-library results, per-version comparison |
| `07_analysis.md` | 7 | Analysis: "子"-family case study, ablation, failure taxonomy |
| `08_conclusion.md` | 8 | Discussion, limitations, future work, conclusion |

---

## Abstract

We present a neural network system for batch glyph style transfer that localizes a Japanese CJK font (FOT-SkipPro-B) into its Simplified Chinese counterpart while preserving the original artistic design language. The core component, **GlobalGlyphTransformer**, is a 36-parameter CNN–Transformer–CNN hybrid that predicts a complete 256×256 Signed Distance Field (SDF) for each target glyph given four-channel "deskinned" input channels derived from a geometrically neutral reference font (Noto Sans CJK Light). Unlike prior local inpainting approaches that suffered from seam artifacts and boundary truncation at radical junctions, v4.4 employs a **direct SDF prediction head** that eliminates the piecewise `base_blend` composition and its associated hard-clip discontinuities. Trained end-to-end with a composite objective combining boundary-weighted SDF L1, surface-normal gradient consistency, straight-stem normal alignment, soft-clDice topological fidelity, soft IoU, flow total-variation regularization, narrow-band Eikonal enforcement, and curvature alignment with curvature TV smoothing, the model achieves a final training SDF L1 of 0.131 and soft IoU of 0.786. On the full 8,105-character 《通用规范汉字表》 (Tongui-8105) target set, the system packs 4,091 missing glyphs and 952 simplified-variant redraws into two independent OpenType CFF font files, achieving 8,095/8,105 (99.88%) coverage. A structural gate with Chamfer-distance, p95 offset, differential-zone, and topology checks intercepts 378 (6.97%) low-confidence candidates for manual review. We provide a detailed case study on the challenging "子" (zi) radical family — where the original font's extreme rightward-arching hook exceeds the model's physical deformation budget — and a systematic failure taxonomy that informs the v4.5 design roadmap.

---

## Notation

| Symbol | Meaning |
|:---|:---|
| $S = 256$ | Canvas side length (pixels) |
| $\Delta = 96$ | Padding in UPEM units (canvas covers 1192 UPEM) |
| $\text{u/px} = (1000 + 2\Delta)/S = 4.65625$ | UPEM per pixel |
| $T_{\text{trunc}} = 16$ px $\approx 74.5$ UPEM | SDF truncation distance |
| $\mathbf{x} \in \mathbb{R}^{4 \times S \times S}$ | Input: $[\phi_{\text{light}}, \phi_{\text{skel}}, \cos 2\theta, \sin 2\theta]$ |
| $\mathbf{y} \in \mathbb{R}^{1 \times S \times S}$ | Target: normalized SDF of SkipPro glyph |
| $\phi(\mathbf{p})$ | Signed distance from point $\mathbf{p}$ to glyph contour (positive outside, negative inside), truncated at $\pm T_{\text{trunc}}$ |
| $\mathbf{f} \in \mathbb{R}^{2 \times S \times S}$ | Optical-flow displacement field in pixels, bounded by $\pm 8.0$ px |
| $d = 768$ | Transformer hidden dimension |
| $L = 4$ | Number of Transformer encoder layers |
| $H = 12$ | Number of attention heads |
| $P = 16 \times 16 = 256$ | Number of patch tokens |
| $r_u$ | Uniform half-width parameter (UPEM) for skeleton SDF channel |
| $\text{UPEM}$ | Units Per EM (OpenType design unit, 1000 for this font) |
