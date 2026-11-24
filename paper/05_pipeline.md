# 5. Method: Inference Pipeline, Structural Gate, and Vectorization

## 5.1 End-to-End Inference Pipeline

The full inference pipeline for each target character proceeds through five stages:

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Input Construction                                              │
│   - Extract SC reference outline (Noto Sans CJK, zh-Hans, wght=None)     │
│   - Build 4-channel deskinned tensor (SDF, skel SDF, cos2θ, sin2θ)       │
│   - Apply inference-time jitter: (dw, r_u) from retry schedule           │
├──────────────────────────────────────────────────────────────────────────┤
│ Stage 2: Neural SDF Prediction                                           │
│   - GlobalGlyphTransformer forward pass → (pred_sdf, flow_px)            │
│   - BF16 mixed precision on CUDA, FP32 fallback on CPU                   │
├──────────────────────────────────────────────────────────────────────────┤
│ Stage 3: SDF → Contour Vectorization                                     │
│   - 3× upsample SDF (832×832) for sub-pixel contour resolution           │
│   - Marching squares → raw polygonal contours                            │
│   - Convert to cubic Bézier segments (CFF commands)                      │
├──────────────────────────────────────────────────────────────────────────┤
│ Stage 4: Contour Post-Processing                                         │
│   - Taubin arc-length smoothing (endpoint-preserving)                    │
│   - Curvature-masked line-segment snapping (no_snap_mask)                │
│   - Hausdorff-guarded revert (max_hausdorff=4.5 UPEM)                    │
│   - Island cleanup (< 400 UPEM² fragments)  [v4.5: still missing]        │
├──────────────────────────────────────────────────────────────────────────┤
│ Stage 5: Structural Gate Validation                                      │
│   - Chamfer distance check                                               │
│   - p95 percentile offset check                                          │
│   - Differential zone coverage/residual check                            │
│   - Topology invariants (holes, endpoints, junctions)                    │
│   - PASS → accept into output font                                       │
│   - FAIL → 3 retry attempts with jittered (dw, r_u)                      │
│   - All 3 FAIL → BlockedToReview (isolated, not in output font)          │
└──────────────────────────────────────────────────────────────────────────┘
```

## 5.2 Retry Schedule

When the structural gate rejects a candidate, the synthesizer retries with **jittered input parameters** to explore the neighborhood of the initial prediction:

| Attempt | $\delta_w$ (weight shift) | $r_u$ (half-width, UPEM) |
|:---:|:---:|:---:|
| 0 | 0.00 | 40.0 |
| 1 | −0.04 | 43.0 |
| 2 | +0.05 | 37.0 |

The negative $\delta_w$ (attempt 1) makes the input SDF "thinner," encouraging the network to predict slightly bolder strokes. The positive $\delta_w$ (attempt 2) does the opposite. The $r_u$ variation adjusts the skeleton half-width, indirectly controlling perceived stroke weight.

**v4.4 statistics**: Of 5,421 total candidates, 5,043 (93.03%) passed on at least one attempt. The remaining 378 (6.97%) failed all three attempts and were routed to `BlockedToReview`.

## 5.3 SDF to Contour Vectorization

The predicted SDF is converted to OpenType CFF cubic Bézier outlines via:

1. **3× Upsampling**: The 256×256 SDF is upsampled to 768×768 (effective resolution: 1192 UPEM / 768 ≈ 1.55 UPEM/px) for sub-pixel contour resolution.

2. **Marching Squares**: The zero-level set ($\phi = 0$) is extracted as a set of polygonal contours. Each contour is a closed loop of vertices.

3. **Bézier Fitting**: Each polygonal contour is fitted with cubic Béziers using a least-squares approach with maximum deviation $1.8$ UPEM. The fitter (`sdf_to_cff_commands`) produces `DrawCmd` objects (move-to, line-to, cubic-to, close) compatible with the CFF table.

4. **Hole Detection**: Contours are classified as outer boundaries or holes via the even-odd rule. Hole count is a gate invariant.

## 5.4 Contour Post-Processing

### 5.4.1 Taubin Arc-Length Smoothing

Before line-segment snapping, each contour undergoes **Taubin low-pass smoothing** (5 iterations, λ=0.5, μ=−0.53) that reduces high-frequency noise while preserving overall shape. Endpoints are fixed to prevent contour closure drift.

### 5.4.2 Curvature-Masked Line-Segment Snapping

The `optimize_synthesized_contours_v3` function identifies **straight segments** in the fitted Bézier contours and snaps them to exact lines (removing the Bézier control points, producing `line-to` commands instead of `curve-to`).

**No-snap mask**: A mask is computed from the reference SC contour's curvature. Pixels where the reference has curvature in the range [6°, 42°] (i.e., genuinely curved segments) are dilated by 28 UPEM and marked as **no-snap** — the fitter must not force these into straight lines.

**Line-segment test**: A Bézier segment is eligible for snapping if:
- Length $L \geq 52$ UPEM
- Maximum perpendicular deviation from the chord $\leq 4.5$ UPEM
- Not in the no-snap mask

### 5.4.3 Hausdorff-Guarded Revert

After snapping, each modified contour is compared to the original fitted contour via **bidirectional Hausdorff distance**. If $d_H > 4.5$ UPEM or the relative area change $> 1.8\%$, the **entire contour is reverted** to the unconstrained Bézier fit.

> **Known limitation**: The 4.5 UPEM Hausdorff threshold is **smaller than one pixel** at 256 resolution (1 px = 4.656 UPEM). This means that a single-pixel perturbation at any point on a contour triggers a full revert. The result is a **98.13% revert rate** (26,942 reverts out of 27,413 total contours in v4.4), which destroys the line-segment snapping benefit and inflates the `on_curve_nodes` count. This is the primary cause of the `zero_mid` metric (fraction of glyphs with zero mid-stem Bézier control points) being only 22.08% in v4.4 vs. 67.99% in v3.
>
> **v4.5 fix**: Segment-level (not whole-contour) Hausdorff validation, and threshold increase to 5.5–6.0 UPEM.

## 5.5 Structural Gate (StructuralGateV3)

The structural gate is a **rule-based geometric validator** that checks a candidate glyph against multiple criteria before accepting it into the output font. All checks must pass simultaneously.

### 5.5.1 Check 1: Chamfer Distance

$$d_{\text{chamfer}} = \max\left(\max_{\mathbf{p} \in C_{\text{cand}}} \min_{\mathbf{q} \in C_{\text{ref}}} |\mathbf{p} - \mathbf{q}|,\ \max_{\mathbf{q} \in C_{\text{ref}}} \min_{\mathbf{p} \in C_{\text{cand}}} |\mathbf{q} - \mathbf{p}|\right)$$

The candidate's Chamfer distance to the SC reference must be $\leq 28.0$ UPEM (relaxed to $\leq 30.0$ for Missing glyphs). This bounds the maximum contour deviation.

### 5.5.2 Check 2: p95 Percentile Offset

The **p95 percentile** of the point-wise distance between candidate and reference contours must satisfy:
$$p_{95} \leq \begin{cases} 92.0 & \text{if } d_{\text{chamfer}} \leq 28.0 \text{ and } \text{miss}_{\text{sc}} \leq 0.15 \text{ and } \text{miss}_{\text{cand}} \leq 0.15 \\ 72.3 & \text{otherwise} \end{cases}$$

> **Known limitation (step-cliff)**: The threshold has a **discontinuity at $d_{\text{chamfer}} = 28.0$ UPEM**. A glyph with Chamfer = 28.02 (exceeding by 0.02 UPEM) drops from the 92.0 limit to 72.3, potentially causing a false rejection. In v4.4, the character `孖` (Chamfer = 29.49) is affected by this cliff. **v4.5 fix**: Smooth transition or per-radical exemption.

### 5.5.3 Check 3: Differential Zone Coverage

For SC-Variant glyphs, the gate verifies that the **intended replacement region** was actually modified while non-target regions remain intact:

- $\text{cov}_{\text{dsc}} \geq 0.62$: At least 62% of the SC-differential zone (pixels where JP and SC differ) is covered by the candidate's modification
- $\text{res}_{\text{djp}} \leq 0.35$: At most 35% residual of the JP-differential zone remains unmodified

This prevents the "passive pass" where the candidate is essentially identical to the input (no actual radical replacement occurred).

### 5.5.4 Check 4: Topology Invariants

- **Hole count**: $|n_{\text{holes}}^{\text{cand}} - n_{\text{holes}}^{\text{ref}}| \leq 1$
- **Endpoint count**: $|n_{\text{endpoints}}^{\text{cand}} - n_{\text{endpoints}}^{\text{ref}}| \leq 2$
- **Junction count**: $|n_{\text{junctions}}^{\text{cand}} - n_{\text{junctions}}^{\text{ref}}| \leq 2$

### 5.5.5 Check 5: Strip Maximum Chamfer

The maximum Chamfer distance within any **vertical strip** (100 UPEM wide) must be $\leq 48.8$ UPEM. This catches localized extreme deviations that the global Chamfer might average out.

## 5.6 Output Packaging

Accepted glyphs are packaged into two independent OpenType CFF fonts:

| File | Content | Characters |
|:---|:---|:---:|
| `CnS-SkipPro-B-v44-Missing.otf` | Missing glyphs only | 4,091 |
| `CnS-SkipPro-B-v44-SCVariants.otf` | SC-Variant redraws only | 952 |
| `CnS-SkipPro-B-v44-Full.otf` | Merged: base + Missing + SCVariants | 8,095 |

**Hard invariants** (asserted at build time):
1. `Missing ∩ SCVariants = ∅` (disjoint codepoint sets)
2. `Missing ∩ Base = ∅` (no overwrite of original glyphs)
3. `SCVariants ⊆ Base` (variants only replace existing codepoints)
4. `HARD_KEEP_12` codepoints are point-exact with the original font

Blocked glyphs are isolated in `manual_review/v44_needs_review.otf` with per-glyph diagnostic JSON (`v44_needs_review.json`).

## 5.7 Inference Performance

| Metric | Value |
|:---|:---:|
| Model size | 36.0M parameters (144 MB BF16) |
| Peak VRAM (batch=16) | ~4.2 GB (RTX 5070 Ti) |
| Inference time per glyph | ~12 ms (BF16, batch=16) |
| Full library (5,421 glyphs) | ~65 s inference + ~15 min vectorization/gating |
| Total build time (v4.4) | 958.6 s (~16 min) |
