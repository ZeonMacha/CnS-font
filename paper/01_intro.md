# 1. Introduction

## 1.1 Motivation: Font Localization for Game Commissioning Work

In professional game localization — particularly for titles originally produced in Japan — the art team must extend a proprietary font's glyph repertoire to cover the target language's character set **without commissioning a full re-design**. The Japanese source font in this work, **FOT-SkipPro-B** (7,912 CJK glyphs, 1000 UPEM, CFF outlines), was designed with a distinctive artistic character: exaggerated stroke terminals, a pronounced rightward-arching "竖弯钩" (vertical-then-hook) in the "子" radical, and overall generous whitespace. When localizing to Simplified Chinese, two categories of glyphs are affected:

1. **Missing glyphs** (4,101 characters in the 《通用规范汉字表》 Tongui-8105 target set): Simplified characters that do not exist in the Japanese font at all (e.g., `语`, `饮`, `汤`, `绸`). These must be **synthesized from scratch** while matching the source font's stroke weight, curvature, and terminal styling.

2. **Simplified-variant glyphs** (1,320 characters): Characters that exist in the Japanese font but whose Simplified Chinese form differs in radical structure (e.g., `學 → 学`, `語 → 语`). These require **selective radical replacement** while preserving the non-target components' original design.

The central challenge is **artistic consistency at scale**: a single human calligrapher would need weeks to redraw 5,000+ glyphs. The system must produce, in a single batch run, glyphs that a viewer cannot distinguish from hand-drawn extensions of the original font.

## 1.2 Why Signed Distance Fields?

Glyph outlines are naturally represented as piecewise-algebraic curves (cubic Béziers in OpenType/CFF). Directly regressing Bézier control points from an input glyph is ill-posed for three reasons: (a) the number of control points varies per glyph, (b) point correspondence between source and target outlines is non-trivial for radical replacements, and (c) topological changes (hole creation/annihilation) cannot be expressed as point-wise regression.

**Signed Distance Fields (SDFs)** sidestep all three. An SDF $\phi: \Omega \to \mathbb{R}$ assigns to each pixel the signed distance to the nearest contour, positive outside and negative inside, truncated at a narrow band. This gives:
- A **fixed-size** representation ($S \times S$ grid regardless of glyph complexity),
- **Topology-aware** level-set extraction (holes are simply negative regions surrounded by positive),
- **Smooth gradients** that enable differentiable contour operations (Eikonal, curvature),
- **Natural composition** via $\max$, $\min$, and arithmetic on distance values.

The v4.4 system predicts a full 256×256 SDF per glyph, then vectorizes the zero-level set into cubic Béziers and applies curvature-masked line-segment snapping to produce production-ready OpenType CFF outlines.

## 1.3 Key Design Decisions in v4.4

The v4.4 iteration (this report) addresses the most critical architectural limitation identified in the v4.3 audit [`审计v4.md`](../审计v4.md):

> **v4.3's piecewise SDF composition** — `pred_sdf = (1-α)·warped_light + α·warped_skel + gate·δ` — introduced a **hard-clip discontinuity** at the truncation boundary ($T_{\text{trunc}} = 16$ px). The `active_gate` sigmoid, computed from a dilated union SDF, forced the output to background ($-1.0$) outside a $\sim$107 UPEM radius, creating a physical ceiling on how far the predicted contour could deviate from the input. This was the primary cause of the "子"-family hook truncation documented in the audit.

**v4.4's core change**: replace the piecewise composition with a **direct SDF prediction head** (a two-layer convolutional network) that takes the decoder's spatial features and the warped input channels as joint input and outputs an unbounded (then tanh-squashed) SDF. This eliminates the `active_gate` hard-clip and the `base_blend` interpolation, giving the network full physical expressiveness across the entire canvas.

Additional v4.4 refinements:
- **Stage 1 gradient fix**: The v4.1/v4.3 Stage 1 (frozen CNN, trainable Transformer) was mathematically dead — the zero-initialized `flow_head` and `refine_head` made $\partial\mathcal{L}/\partial\theta_{\text{transformer}} \equiv \mathbf{0}$. v4.4's direct `sdf_head` is **not** zero-initialized, so Stage 1 now produces meaningful gradients from epoch 1.
- **Smoother convergence**: The direct SDF head converges faster in Stage 2 (SDF L1: 0.5998 → 0.1313 over 35 epochs vs. v4.3's 0.6214 → 0.1309) and achieves a lower flow-TV endpoint (0.00001 vs. 0.04457), indicating a smoother deformation field.

## 1.4 Related Work

### CJK Font Generation and Extension

Traditional CJK font production relies on **stroke-order-based vector editing** (e.g., FontLab, AFDKO) or **component-based composition** (e.g., Adobe's Kana/CJK radical systems). These approaches scale linearly with the number of new glyphs and require expert intervention per glyph.

Neural approaches to glyph generation have focused primarily on **image-domain** models: GANs (e.g., FontDiff [Zhang et al. 2023], FontStruct [Yin et al. 2023]) that generate glyph images and then trace them to vectors. These suffer from (a) topology errors in the tracing step, (b) inability to preserve exact stroke-level style consistency, and (c) difficulty with the structured radical composition that CJK glyphs require.

**SDF-based glyph generation** has been explored in the graphics community for shape interpolation (O SDF [Lipman et al. 2006], Unsigned Distance Functions for shape matching). However, no prior work has applied SDF regression specifically to **CJK font localization with radical-level style transfer**.

### Style Transfer for Vector Graphics

Vector style transfer (e.g., [Bertails et al. 2008], [Yamashita et al. 2020]) operates on Bézier control points and is limited to **same-topology** pairs. The CJK localization problem is fundamentally different: the target glyph may have a **different radical structure** (e.g., replacing the Japanese `學` top component with the Simplified `⺍`), which changes the outline topology.

### Structural Gate and Quality Assurance

The `StructuralGateV3` used in this work is a rule-based geometric validator inspired by **font QA tools** (e.g., AFDKO's `ttx` validation, Font Baker's overlap checks) but extended with: (a) Chamfer distance between candidate and reference contours, (b) p95 percentile offset analysis for detecting systematic radical displacement, (c) differential-zone coverage for verifying that the intended replacement region was actually modified while non-target regions remain intact, and (d) topology invariants (hole count, endpoint count, junction count). No prior public system combines all four checks in a CJK font production pipeline.

## 1.5 Contributions

1. **GlobalGlyphTransformer v4.4**: A 36M-parameter CNN–Transformer–CNN architecture with a direct SDF prediction head that eliminates the piecewise composition artifacts of v4.3, trained end-to-end with a nine-term composite loss.

2. **Deskinned four-channel input representation**: A novel input encoding that strips weight/style information from a geometrically neutral reference font (Noto Sans CJK Light), providing the network with pure geometric context (SDF, skeleton distance, orientation field) while decoupling style transfer from weight variation.

3. **Full 8,105-character production pipeline**: An end-to-end system that classifies, synthesizes, gates, vectorizes, and packages CJK glyphs into production-ready OpenType CFF fonts, with SHA-256-verified artifact lineage and hard invariant checks.

4. **Systematic failure taxonomy and "子"-family case study**: A detailed quantitative analysis of the physical reachability limits of the flow-field architecture, the training-set imbalance problem for extreme-curvature radicals, and the structural gate's reverse-selection effect — providing a concrete roadmap for v4.5.

5. **Open artifacts**: All model weights, per-glyph manifests (5,421 entries with full geometric diagnostics), training logs, and comparison images are included in the repository for reproducibility.
