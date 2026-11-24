# 7. Analysis: Failure Taxonomy, Ablation, and the "子"-Family Deep Dive

## 7.1 Physical Reachability Analysis

The most fundamental constraint on the GlobalGlyphTransformer's ability to reproduce the SkipPro-B artistic style is the **physical reachability bound** imposed by the flow-field architecture.

### 7.1.1 Reachability Model

The predicted SDF is constructed from two information sources:
1. **Warped input channels**: The input SDF and skeleton SDF are displaced by at most $\tau_{\max} = 8.0$ px ($\approx$37.25 UPEM). Any target contour feature more than 37 UPEM from the nearest input feature **cannot be reached by warping**.
2. **Spatial features $\mathbf{u}_0$**: The decoder's 24-channel feature map at 256×256 resolution. The SDF head can generate new geometry from these features, but the effective receptive field is limited by the CNN kernel sizes (3×3) and the 16×16 Transformer bottleneck.

The **combined reachability radius** is therefore approximately:
$$R_{\text{reach}} \approx \tau_{\max} \cdot \text{u/px} + R_{\text{head}} \approx 37.25 + \sim 30 \approx 67 \text{ UPEM}$$

where $R_{\text{head}}$ is an empirical estimate of the SDF head's effective generation radius.

### 7.1.2 Measured Reachability Gaps

For the "子"-family characters, the SkipPro-B original's hook displacement relative to the Noto Sans reference (measured as p99 skeleton displacement) far exceeds the reachability bound:

| Character | Target ink pixels | p99 hook displacement (UPEM) | % ink beyond 8 px flow | Gate-out unreachable px |
|:---:|:---:|:---:|:---:|:---:|
| **孑** | 13,711 | **209** | **23.8%** | **1,529** (11.2%) |
| **子** | 12,240 | **182** | **23.1%** | **1,065** (8.7%) |
| **好** | 17,863 | **154** | **14.0%** | **795** (4.5%) |
| **仔** | 15,907 | **149** | **15.1%** | **710** (4.5%) |
| **孚** | 15,337 | **140** | **11.4%** | **507** (3.3%) |
| **孔** | 16,187 | **112** | **16.9%** | **287** (1.8%) |

**Interpretation**: For `孑`, 23.8% of the target ink lies more than 8 px from the nearest input contour. The model **cannot** generate this ink by warping; it must synthesize it from spatial features alone. In practice, the network learns a **compromise**: it partially extends the hook but pulls it back toward the input to maintain contour continuity within the reachable zone. This produces the "flattened hook" artifact visible in the comparison images.

### 7.1.3 VF Weight Mismatch Amplification

The training and inference pipeline uses `weight=None` (default `wght=100` Thin) for the Noto Sans CJK Variable Font, not the intended `wght=300` (Light) or `wght=400` (Regular). The Thin weight produces **narrower strokes**, which increases the distance between the input skeleton and the target SkipPro-B skeleton, **worsening the reachability gap**:

| Character | `wght=400` (Regular): % beyond 8 px | `wght=100` (Thin): % beyond 8 px | Degradation |
|:---:|:---:|:---:|:---:|
| 子 | 23.1% | **31.9%** | +8.8 pp |
| 孑 | 23.8% | **35.5%** | +11.7 pp |
| 孔 | 16.9% | **36.0%** | +19.1 pp |
| 孚 | 11.4% | **26.0%** | +14.6 pp |

This is a **P0 configuration bug** identified in the v4.3 audit. The v4.5 roadmap mandates explicit `--noto_weight 300` for all training and inference.

## 7.2 Training Set Imbalance: The "直柱" Dilution Problem

The training set contains an inherently imbalanced distribution of "子"-radical characters:

| Structural category | Count | Examples | Hook curvature |
|:---|:---:|:---|:---|
| Bottom `*子` (上下 structure) | 14 | 享 孕 字 孚 孛 季 孥 学 孳 孶 學 孿 斈 李 | Mild (near-vertical, p99 = 102–126 UPEM) |
| Left `⿰子` (left radical) | 8 | 孔 孖 孜 孤 孩 孫 孺 孾 | Extreme (rightward arc, p99 = 112–209 UPEM) |
| Right `⿰*子` (right component) | 4 | 仔 好 孖 耔 | Extreme |
| Missing (need synthesis) | 7 | 孙 孢 孪 孽 籽 虸 觃 | Unknown |

The 14 "bottom 子" characters have **mild, near-vertical hooks** that are easily satisfied by the flow field. The 12 "extreme arc" characters (8 left + 4 right) require the full 37 UPEM flow budget plus SDF head generation. In a global average-regression training regime, the model learns a **"safe compromise"**: when it detects a "子" pattern in the input, it predicts a mildly curved hook that is closer to the 14 mild examples than the 12 extreme ones. This is the **"向思源直柱塌缩"** (collapse toward the straight-column baseline) identified in the audit.

**v4.5 fix**: 3×–4× oversampling of the 12 extreme-arc characters, or position-sensitive weighting that upweights the loss contribution of pixels in the hook region.

## 7.3 Structural Gate: Reverse Selection and the p95 Cliff

### 7.3.1 The "Straight-Body Police" Problem

The StructuralGateV3 uses **Noto Sans Light (zh-Hans)** as the geometric reference for all validation checks. This creates a fundamental tension:

- The **target** is to reproduce the SkipPro-B artistic style (extreme hooks, bold terminals)
- The **gate reference** is a geometrically neutral Noto Sans Light (straight stems, moderate hooks)
- The gate's p95 check measures the candidate's deviation from the **Noto reference**, not from the **SkipPro original**

Consequently, a candidate that faithfully reproduces the SkipPro hook (p95 ≈ 100–135 relative to Noto) **fails** the p95 ≤ 92.0 check, while a candidate that flattens the hook toward the Noto reference (p95 ≈ 50–70) **passes**. The gate is **selecting against fidelity to the source font's artistic style**.

This is quantified by the fact that **SkipPro's own original glyphs fail the gate**:
- `子`: p95 = 137.8 → FAIL
- `孑`: p95 = 135.0 → FAIL
- `仔`: p95 = 116.0 → FAIL
- `好`: p95 = 95.3 → FAIL
- `孔`: p95 = 94.2 → FAIL (borderline)

### 7.3.2 The 28.0 UPEM Step-Cliff

The p95 limit has a discontinuity at Chamfer = 28.0 UPEM:
```
p95_limit = 92.0  if chamfer ≤ 28.0 and miss_sc ≤ 0.15 and miss_cand ≤ 0.15
            72.3  otherwise
```

A glyph with Chamfer = 28.02 (exceeding by 0.02 UPEM) drops from the 92.0 limit to 72.3. In v4.3, `孖` (Chamfer = 28.02, p95 = 87.0) was **false-rejected** by this cliff: its p95 of 87.0 would have passed the 92.0 limit but fails the 72.3 limit.

In v4.4, `孖` has Chamfer = 29.49 and p95 = 106.1, failing both the Chamfer ≤ 28.0 check and the p95 check. The cliff is no longer the binding constraint for `孖`, but the mechanism remains a latent risk for other characters.

### 7.3.3 Gate–Classifier Coupling

The same `StructuralGateV3.check()` is called by both:
1. The **inference pipeline** (validating synthesized candidates)
2. The **variant classifier** (deciding whether an original SkipPro glyph already matches SC → `KEEP_ALREADY_SC`)

This coupling means that any change to the gate parameters (e.g., raising `p95_limit` from 72.3 to 92.0 in commit `47593f5`) **silently changes the classifier's output set**. In practice, raising the p95 limit caused `耔`, `禘`, and `次` to flip from `SC_VARIANT` (redraw) to `KEEP_ALREADY_SC` (keep original), reducing the redraw queue by 3 characters without any explicit decision.

**v4.5 fix**: Decouple the classifier from the gate by giving the classifier an independent, frozen `ClassifierConfig` that is not affected by gate parameter tuning.

## 7.4 Contour Post-Processing: The 98.13% Revert Problem

### 7.4.1 Root Cause

The `optimize_synthesized_contours_v3` function's Hausdorff-guarded revert operates at the **whole-contour** level: if **any** single point on a closed contour deviates by more than 4.5 UPEM from the original fit, the **entire contour** is reverted to the unconstrained Bézier representation.

The critical numerical issue: at 256×256 resolution, **1 pixel = 4.656 UPEM**. The Hausdorff threshold of 4.5 UPEM is therefore **smaller than one pixel**. Any single-pixel perturbation (from the 3× upsample, the marching squares discretization, or the Taubin smoothing) is guaranteed to exceed 4.5 UPEM at some point, triggering a full revert.

### 7.4.2 Consequences

| Consequence | Measurement |
|:---|:---|
| Contour revert rate | 98.13% (26,942 / 27,413) |
| Line-segment snap retention | 1.72% (471 / 27,413) |
| `zero_mid` (clean straight stems) | 22.08% (vs. 67.99% in v3) |
| Average `on_curve_nodes` per glyph | 129.80 (vs. 192.16 in v3) |

The revert **destroys the benefit of line-segment snapping**: even when the curvature mask correctly identifies a straight segment and the fitter produces a clean line, the Hausdorff check reverts the entire contour back to a Bézier representation with redundant control points.

### 7.4.3 v4.5 Fix

1. **Segment-level validation**: Check Hausdorff per-segment, not per-contour. Only revert segments that exceed the threshold.
2. **Threshold increase**: Raise `max_hausdorff_upem` from 4.5 to 5.5–6.0 UPEM (or validate at 3× upsampled resolution where 1 px = 1.55 UPEM).
3. **Island cleanup**: Add the `< 400 UPEM²` fragment filter (present in `local_inpainter.py` but **missing** from `synthesizer_v4.py`) to eliminate floating specks like `孩` (106.5 UPEM²) and `孓` (686.1 UPEM²).

## 7.5 Ablation: The Role of the Direct SDF Head

While a full ablation study (v4.3 piecewise vs. v4.4 direct, same training data and schedule) was not explicitly run, the training logs provide indirect evidence:

| Metric | v4.3 (piecewise) | v4.4 (direct) | Interpretation |
|:---|:---:|:---:|:---|
| Stage 1 SDF L1 (Ep 1→5) | 0.808 → 0.807 (flat) | 0.965 → 0.753 (decreasing) | v4.4 Stage 1 is **not** a dead zone |
| Stage 2 SDF L1 (final) | 0.1309 | 0.1313 | Equivalent final fit quality |
| Stage 2 Flow TV (final) | **0.0446** | **0.00001** | v4.4 uses 4,460× less warping |
| Stage 2 Flow TV (trend) | **Increasing** (0.023→0.045) | **Decreasing** (0.0035→0.00001) | v4.3 compensates for gate dead-zone |
| Soft IoU (final) | 0.7852 | 0.7858 | Negligible difference |
| Eikonal (final) | 0.0162 | 0.0146 | v4.4 slightly more SDF-consistent |

The flow TV trend is the most diagnostic: v4.3's increasing flow TV indicates the network is **progressively relying more on warping** as training proceeds, because the `active_gate` dead-zone prevents the SDF head from generating geometry outside ~107 UPEM. The network learns to "stretch" the flow field to reach farther, at the cost of flow smoothness. v4.4's decreasing flow TV confirms that the direct SDF head **takes over** the generation task, freeing the flow field to remain smooth.

## 7.6 Failure Taxonomy Summary

| Failure mode | Root cause | Affected chars | v4.5 fix |
|:---|:---|:---|:---|
| Hook truncation / flattening | 8 px flow bound + training imbalance | 孑 仔 好 孖 孔 孚 | Increase `max_shift_px` to 14–16; oversample extreme-arc chars |
| Gate reverse-selection | p95 measured vs. Noto, not SkipPro | 孑 仔 好 孖 子 | Decouple gate reference; use SkipPro-family envelope as p95 upper bound |
| p95 step-cliff | Discontinuous threshold at Chamfer=28.0 | 孖 (v4.3), others latent | Smooth transition or per-radical exemption |
| `diff_zone` rejection | Incomplete radical replacement | 365/378 blocked | Component-level conditioning; selective Stage 0a vector surgery |
| Island/speck artifacts | Missing `< 400 UPEM²` filter in Tier 3 | 孩 (106.5), 孓 (686.1), 塆 询 谌 馕 | Add island cleanup to `synthesizer_v4.py` |
| `zero_mid` regression | Hausdorff revert at 4.5 UPEM < 1 px | All 5,421 glyphs | Segment-level validation; threshold → 5.5–6.0 UPEM |
| VF weight mismatch | `weight=None` → wght=100 (Thin) | All chars (worsens reachability) | Explicit `--noto_weight 300` |
| Cache hash gap | Tag lacks font weight/hash | All (stale cache risk) | Embed weight + font SHA256 in cache tag |
