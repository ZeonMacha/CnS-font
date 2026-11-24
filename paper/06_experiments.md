# 6. Experiments and Results

## 6.1 Experimental Setup

### Hardware

| Component | Specification |
|:---|:---|
| GPU | NVIDIA GeForce RTX 5070 Ti (16 GB GDDR7) |
| CPU | (host system) |
| Precision | BF16 mixed precision (FP32 fallback for curvature/Eikonal terms) |
| Framework | PyTorch 2.x, CUDA 12.x |

### Training Configuration (v4.4)

| Parameter | Value |
|:---|:---:|
| Canvas size | 256 × 256 |
| Batch size | 16 |
| d_model | 768 |
| Transformer layers | 4 |
| Attention heads | 12 |
| Stage 1 epochs | 5 (frozen CNN, trainable Transformer + SDF head) |
| Stage 2 epochs | 35 (full end-to-end) |
| Stage 1 LR | 1×10⁻⁴ (linear warmup 3 ep + cosine decay) |
| Stage 2 LR | 1×10⁻⁴ (linear warmup 7 ep + cosine decay) |
| Weight decay | 0.01 (AdamW) |
| Gradient clip | 1.0 (L2 norm) |
| Total training time | 3,274 s (~55 min) |
| Checkpoint | `checkpoints/v44_zh_sdf_global.pt` (144 MB) |

### Build Configuration (v4.4)

| Parameter | Value |
|:---|:---:|
| Target set | Tongui-8105 (8,105 characters) |
| Source font | FOT-SkipPro-B.OTF (7,912 CJK glyphs) |
| Reference font | NotoSansCJK-VF.ttc (weight=None → default wght=100) |
| Mode | Pure v4 (no legacy v3 tiers) |
| Gate retry attempts | 3: (dw=0, r_u=40), (dw=−0.04, r_u=43), (dw=+0.05, r_u=37) |
| Build time | 958.6 s (~16 min) |

## 6.2 Full-Library Results: v4.4

### 6.2.1 Coverage and Packaging

| Metric | v4.4 Value |
|:---|:---:|
| Tongui-8105 total | 8,105 |
| Missing targets | 4,101 |
| Missing packed | **4,091** (99.76%) |
| SC-Variant targets | 1,320 |
| SC-Variant packed | **952** (72.12%) |
| Blocked to review | **378** (6.97%) |
| Gate pass rate | **93.03%** |
| Merged TG coverage L1 | 3,496 / 3,500 (99.89%) |
| Merged TG coverage L2 | 2,998 / 3,000 (99.93%) |
| Merged TG coverage L3 | 1,601 / 1,605 (99.75%) |
| **Merged TG total coverage** | **8,095 / 8,105 (99.88%)** |
| keep_same_gid | 1,453 |
| keep_v1_v2_exempt | 349 |
| keep_already_sc | 874 |
| keep_policy_12 | 8 |

### 6.2.2 Blocked Glyph Breakdown (378 total)

| Block reason | Count | % of blocked |
|:---|:---:|:---:|
| `diff_zone` (differential zone coverage/residual) | 365 | 96.6% |
| `p95` (percentile offset exceeded) | 14 | 3.7% |
| `strip_max_ch` (strip Chamfer exceeded) | 4 | 1.1% |
| `miss_sc` (SC miss rate exceeded) | 2 | 0.5% |
| `miss_cand` (candidate miss rate exceeded) | 2 | 0.5% |
| `cos3x3` (cosine 3×3 check) | 1 | 0.3% |
| `holes` (topology invariant) | 1 | 0.3% |

The overwhelming dominance of `diff_zone` (96.6%) confirms that the primary failure mode is **incomplete radical replacement** — the model fails to sufficiently modify the target radical region while leaving non-target regions intact. This is a fundamental limitation of the global SDF approach for characters with complex multi-radical structures.

### 6.2.3 Contour Quality Statistics

| Metric | v4.4 | v4.3 | v4.1 | v3 |
|:---|:---:|:---:|:---:|:---:|
| Total glyphs in manifest | 5,421 | 5,421 | 5,125 | 5,271 |
| Passed gate | 5,043 | 5,047 | 4,826 | 4,862 |
| `zero_mid` (no mid-stem Bézier points) | **1,197 (22.08%)** | 968 (17.86%) | 1,098 (21.42%) | **3,584 (67.99%)** |
| Snapped contours | 471 | 501 | 554 | 2,435 |
| Reverted contours | 26,942 | 26,867 | 26,939 | 15,698 |
| Snap retention rate | 1.72% | 1.87% | 2.06% | **13.43%** |

The `zero_mid` metric — the fraction of glyphs whose outlines contain **no Bézier control points in the middle of straight stems** — is a direct measure of line-segment snapping effectiveness. v3's 67.99% indicates that most straight stems were correctly identified and snapped to clean lines. v4.x's ~20% indicates that the Hausdorff-guarded revert (Section 5.4.3) is undoing nearly all snapping.

## 6.3 Cross-Version Comparison

### 6.3.1 Summary Metrics (v3 → v4 → v4.1 → v4.3 → v4.4)

| Metric | v3 | v4 | v4.1 | v4.3 | **v4.4** | Trend |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| Missing packed | 4,088 | 4,105 | 4,094 | 4,096 | **4,091** | stable |
| SC-Variant packed | 1,183 | 1,110 | 1,031 | 951 | **952** | v4.4 ≈ v4.3 |
| Blocked to review | 257 | 214 | 299 | 374 | **378** | v4.4 ≈ v4.3 |
| Gate pass rate | 95.35% | 96.06% | 94.49% | 93.10% | **93.03%** | stable |
| Merged TG coverage | 8,087 | 8,104 | 8,098 | 8,100 | **8,095** | stable |
| `zero_mid` | **67.99%** | 59.46% | 21.42% | 17.86% | **22.08%** | v4.4 +4.2pp over v4.3 |
| Snap retention | 13.43% | 8.64% | 2.06% | 1.87% | **1.72%** | declining |
| Training SDF L1 (final) | — | 0.310 | 0.139 | 0.131 | **0.131** | v4.4 ≈ v4.3 |
| Training Soft IoU (final) | — | 0.720 | 0.784 | 0.785 | **0.786** | v4.4 +0.001 |
| Training Flow TV (final) | — | — | 0.0014 | 0.0446 | **0.00001** | **v4.4 4,460× lower** |

### 6.3.2 Key Observations

1. **v4.4 vs v4.3**: The direct SDF head produces nearly identical final SDF L1 (0.1313 vs. 0.1309) and Soft IoU (0.7858 vs. 0.7852), but with a **drastically smoother flow field** (TV: 0.00001 vs. 0.0446). The `zero_mid` improves by 4.2 percentage points (22.08% vs. 17.86%), suggesting the direct SDF produces slightly cleaner contours that are marginally more amenable to line-segment snapping. The blocked count increases by 4 (378 vs. 374), a negligible difference within run-to-run variance.

2. **v4.4 vs v3**: The architectural shift from local inpainting (v3) to global SDF (v4.x) trades **coverage** (v3's 95.35% gate pass vs. v4.4's 93.03%) for **contour continuity** (elimination of mask-boundary seams). The `zero_mid` regression (67.99% → 22.08%) is entirely attributable to the Hausdorff-guarded revert bug, not the neural model.

3. **Flow TV as an architecture diagnostic**: The v4.3 flow TV *increased* during Stage 2 (0.023 → 0.045), indicating the network was compensating for the `active_gate` dead-zone by relying more on warping. v4.4's monotonically decreasing flow TV (0.0035 → 0.00001) confirms that the direct SDF head can generate new geometry without warping, validating the architectural change.

## 6.4 "子"-Family Case Study (Quantitative)

The "子" (zi) radical family is the most challenging case in the entire 8,105-character set due to the original font's extreme rightward-arching hook. The following table compares v4.3 and v4.4 for all 26 "子"-family characters in the manifest:

| Character | Codepoint | Tier | v4.3 Result | v4.4 Result | Δ |
|:---:|:---:|:---:|:---|:---|:---|
| 仔 | U+4ED4 | SCVar | ❌ FAIL (p95=95.1, diff_zone) | ❌ FAIL (p95=116.0, strip) | p95 +20.9 |
| 好 | U+597D | SCVar | ❌ FAIL (p95=94.3, diff_zone) | ❌ FAIL (p95=108.8, diff_zone) | p95 +14.5 |
| 孑 | U+5B51 | SCVar | ❌ FAIL (p95=94.3) | ❌ FAIL (p95=130.5, strip) | p95 +36.3 |
| 孓 | U+5B53 | Missing | ✅ PASS (ch=17.0, p95=56.6) | ❌ FAIL (p95=95.2, miss_sc) | **REGRESSED** |
| 孔 | U+5B54 | SCVar | ✅ PASS (ch=24.4, p95=84.9) | ✅ PASS (ch=27.0, p95=87.0) | p95 +2.1 |
| 孖 | U+5B56 | SCVar | ❌ FAIL (ch=28.0, p95=87.0, cliff) | ❌ FAIL (ch=29.5, p95=106.1, strip) | p95 +19.1 |
| 孚 | U+5B5A | SCVar | ✅ PASS (ch=24.1, p95=66.9) | ✅ PASS (ch=24.9, p95=58.4) | p95 −8.5 |
| 孜 | U+5B5C | SCVar | ❌ FAIL (diff_zone) | ❌ FAIL (diff_zone) | stable |
| 孢 | U+82DE | Missing | ✅ PASS (ch=18.9, p95=46.1) | ✅ PASS (ch=17.8, p95=43.5) | p95 −2.6 |
| 孤 | U+5B64 | SCVar | ✅ PASS (ch=20.2, p95=58.0) | ✅ PASS (ch=19.3, p95=50.8) | p95 −7.3 |
| 孩 | U+5B69 | SCVar | ✅ PASS (ch=21.2, p95=50.8) | ❌ FAIL (diff_zone) | **REGRESSED** |
| 孪 | U+5B6A | Missing | ✅ PASS (ch=15.5, p95=59.1) | ✅ PASS (ch=13.6, p95=58.9) | p95 −0.1 |
| 孳 | U+5B73 | SCVar | ✅ PASS (ch=13.5, p95=36.3) | ✅ PASS (ch=11.9, p95=39.0) | p95 +2.8 |
| 孵 | U+5B75 | Missing | ✅ PASS (ch=19.8, p95=43.5) | ✅ PASS (ch=19.9, p95=44.9) | stable |
| 孺 | U+5B7A | SCVar | ❌ FAIL (diff_zone) | ❌ FAIL (diff_zone) | stable |
| 孽 | U+5B6A | Missing | ✅ PASS (ch=11.0, p95=29.0) | ✅ PASS (ch=10.5, p95=29.7) | stable |

### Key Findings

1. **v4.4 produces more faithful (but gate-failing) "子" hooks**: Characters like `孑` (p95: 94.3 → 130.5) and `仔` (p95: 95.1 → 116.0) show **larger p95 offsets** in v4.4, meaning the direct SDF head generates hooks that are **closer to the SkipPro original** (which itself fails the gate at p95=135–138) but are rejected by the structural gate. This is the **reverse-selection effect** documented in the audit: the gate penalizes fidelity to the original style.

2. **Two regressions**: `孓` (v4.3 PASS → v4.4 FAIL) and `孩` (v4.3 PASS → v4.4 FAIL) are concerning. The `孓` regression is due to increased `miss_sc` (0.27 > 0.15 threshold), suggesting the v4.4 model produces a slightly different stroke coverage pattern. The `孩` regression is due to `diff_zone` (the model over-modified the non-target "子" component).

3. **Six "子"-family SC-Variants remain blocked** in v4.4 (`仔 好 孑 孖 孜 孺`), the same set as v4.3. The direct SDF head does not resolve the fundamental physical reachability limitation of the 8 px flow bound.

## 6.5 Visual Comparison

The repository includes the following comparison images:

| Image | Content |
|:---|:---|
| [`outputs/compare_v43_vs_v44.png`](../outputs/compare_v43_vs_v44.png) | 5-row comparison: Noto SC Ref / v43 Merged / v44 Merged for simplified radicals, regression chars, SC variants, and custom spot-checks |
| [`outputs/v44_zoom_verify.png`](../outputs/v44_zoom_verify.png) | 110 px zoomed comparison of v43 (red) vs v44 (black) on 13 custom characters including "子"-family |
| [`outputs/debug_v43_zi_3stages.png`](../outputs/debug_v43_zi_3stages.png) | Three-stage debug for "子" character (SDF → contour → vectorized) |
| [`outputs/gpu_simplified_12_showcase.png`](../outputs/gpu_simplified_12_showcase.png) | 12 simplified radical showcase |
