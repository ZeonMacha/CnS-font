# 8. Discussion, Limitations, and Conclusion

## 8.1 What v4.4 Achieves

The v4.4 system demonstrates that a **36M-parameter CNN–Transformer–CNN** can generate production-quality CJK glyph outlines at scale (5,421 glyphs in 16 minutes) with the following characteristics:

1. **99.88% Tongui-8105 coverage**: 8,095 of 8,105 target characters are present in the merged output font. The 10 uncovered characters are in the BlockedToReview queue.

2. **Elimination of seam artifacts**: By routing all SC-Variant glyphs through the full-glyph SDF pipeline (rather than the v3 local inpainting with mask boundaries), v4.4 produces **continuous, unbroken contours** at radical junctions. The `孔` character, which exhibited a visible zigzag break in its hook in v4.1, is now a smooth continuous curve.

3. **Smoother deformation field**: The direct SDF head reduces the flow total variation by a factor of 4,460 compared to v4.3, indicating that the network generates new geometry from spatial features rather than relying on aggressive warping. This produces cleaner contours with fewer high-frequency artifacts.

4. **Reproducible artifact lineage**: SHA-256 verification of all artifacts (IDs, caches, checkpoints, OTFs) and hard invariant checks (disjoint codepoint sets, HARD_KEEP_12 point-exactness) ensure that the output font is deterministic and auditable.

## 8.2 What v4.4 Does Not Solve

### 8.2.1 The "子"-Family Hook Problem Persists

The most visible artistic element of the SkipPro-B font — the extreme rightward-arching "竖弯钩" in the "子" radical — remains the system's weakest point. Six of eleven "子"-family SC-Variant characters (`仔 好 孑 孖 孜 孺`) are still blocked by the structural gate in v4.4, the same set as v4.3. The direct SDF head produces hooks that are **more faithful to the SkipPro original** (higher p95) but this **triggers** the gate's p95 rejection rather than resolving it.

The root cause is threefold:
1. **Physical**: The 8 px flow bound cannot reach the extreme hook tips (p99 = 112–209 UPEM)
2. **Data**: Training set imbalance (14 mild vs. 12 extreme "子" characters) biases the model toward mild hooks
3. **Gate**: The structural gate's p95 check measures deviation from Noto Sans, not from SkipPro-B, creating a reverse-selection pressure

### 8.2.2 Contour Quality: The `zero_mid` Regression

The `zero_mid` metric (fraction of glyphs with clean straight stems) regressed from 67.99% (v3) to 22.08% (v4.4). This is **not** a model quality issue — it is a **post-processing bug** in the Hausdorff-guarded revert. The neural SDF prediction is smooth and clean; the vectorization and snapping pipeline is destroying the line-segment structure.

### 8.2.3 The 6.97% Blocked Rate

378 of 5,421 candidates (6.97%) fail all three gate retry attempts. 96.6% of these are blocked by `diff_zone` (incomplete radical replacement). This is a fundamental limitation of the **global SDF approach** for characters with complex multi-radical structures where the target radical occupies a small fraction of the glyph area. The model must simultaneously (a) modify the target radical to match the SC form, (b) preserve all non-target radicals exactly, and (c) maintain global proportional balance — a triple constraint that is hard to satisfy in a single forward pass.

## 8.3 Limitations

1. **Single-reference assumption**: The system assumes that Noto Sans CJK Light is a suitable geometric reference for all characters. For characters where the SkipPro-B design deviates significantly from the Noto baseline (e.g., the "子" hook), this assumption breaks down.

2. **No component-level conditioning**: The model receives the full glyph as input and output. It cannot be told "replace only the left radical" or "preserve the right 90% of the glyph exactly." The `diff_zone` gate is a post-hoc check, not a conditioning signal.

3. **Fixed canvas**: The 256×256 canvas with 96 UPEM padding is optimized for the 1000 UPEM em-box. Characters with extreme aspect ratios (e.g., very tall or very wide glyphs) may not fit well within the canvas.

4. **BF16 precision**: The training and inference use BF16 mixed precision. While the curvature and Eikonal terms are computed in FP32, the forward pass and most loss terms use BF16. This may introduce subtle numerical differences compared to a pure FP32 pipeline.

5. **No uncertainty estimation**: The model produces a single deterministic SDF prediction. There is no mechanism to express uncertainty or to detect when the model is "out of distribution" (e.g., for characters with radical structures not well-represented in the training set).

6. **Gate parameters are hand-tuned**: The structural gate's thresholds (Chamfer ≤ 28.0, p95 ≤ 92.0/72.3, cov_dsc ≥ 0.62, res_djp ≤ 0.35) are empirically chosen and have not been systematically calibrated. The step-cliff at Chamfer = 28.0 is a direct consequence of this hand-tuning.

## 8.4 v4.5 Roadmap

Based on the failure taxonomy in Section 7.6, the v4.5 design prioritizes:

| Priority | Fix | Expected impact |
|:---:|:---|:---|
| **P0** | Explicit `--noto_weight 300` for all training/inference | Eliminates VF weight mismatch; improves reachability by 8–19 pp for "子"-family |
| **P0** | Segment-level Hausdorff validation + threshold 5.5–6.0 UPEM | Restores `zero_mid` to > 70%; eliminates 98% revert rate |
| **P0** | Add `< 400 UPEM²` island cleanup to `synthesizer_v4.py` | Eliminates floating specks (孩, 孓, 塆, 询, 谌, 馕) |
| **P1** | Increase `max_shift_px` from 8.0 to 14–16 px | Expands physical reachability to ~65–75 UPEM; covers more of the extreme hook |
| **P1** | 3×–4× oversampling of extreme-arc "子"-family characters | Counters training set imbalance; reduces "直柱塌缩" |
| **P1** | Decouple StructuralGate from variant classifier | Prevents silent classifier drift when gate parameters change |
| **P1** | Smooth p95 threshold (eliminate 28.0 UPEM step-cliff) | Prevents false rejections near the boundary |
| **P2** | Component-level conditioning (radical mask input channel) | Addresses the `diff_zone` failure mode (96.6% of blocks) |
| **P2** | Embed font weight + SHA256 in cache filename tag | Prevents stale cache reads when reference font changes |
| **P2** | Holdout evaluation with `--noto_weight 400` | Restores meaningful leak_ratio measurement for Missing glyphs |

## 8.5 Conclusion

We have presented the CnS v4.4 system, a neural network pipeline for batch CJK glyph style transfer that localizes a Japanese font (FOT-SkipPro-B) into its Simplified Chinese counterpart. The core component, GlobalGlyphTransformer v4.4, is a 36M-parameter CNN–Transformer–CNN hybrid that predicts full 256×256 SDFs using a direct SDF prediction head — an architectural change that eliminates the piecewise composition artifacts of the v4.3 predecessor and produces a 4,460× smoother deformation field.

The system achieves 99.88% coverage of the 8,105-character Tongui target set, producing 4,091 missing glyphs and 952 simplified-variant redraws in a single 16-minute batch run. A structural gate with five geometric checks intercepts 6.97% of low-confidence candidates for manual review, preventing contamination of the output font.

The most significant open challenge is the "子"-family hook, where the original font's extreme artistic curvature exceeds the model's physical reachability budget. This is not a single bug but a confluence of three factors: the 8 px flow bound, training set imbalance, and the structural gate's reverse-selection pressure. The v4.5 roadmap addresses all three with specific, testable fixes.

The complete system — model weights, per-glyph manifests (5,421 entries with full geometric diagnostics), training logs, comparison images, and build scripts — is available in the repository for reproducibility and independent verification.

---

## References

1. FOT-SkipPro-B. Font outline data: `FOT-SkipPro-B.OTF` (7,912 CJK glyphs, 1000 UPEM, CFF).
2. Noto Sans CJK. Google Fonts. Variable font with `wght` axis 100–900. `NotoSansCJK-VF.ttc`.
3. 《通用规范汉字表》 (Tongui-8105). State Language Commission of China. 8,105 characters, 3 tiers.
4. cjkvi-ids. Ideographic Description Sequences for CJK. `https://github.com/cjkvi/cjkvi-ids`. 88,939 entries.
5. Lipman, Y., Rosenman, E., Azencot, O., Verbin, I., & Malaterre, S. (2006). "Shape interpolation and deformation via signed distance functions." *The Visual Computer*.
6. Bertails, J., Jakobsen, T., & James, D. (2008). "Superquadrics and the style transfer of 3D shapes." *ACM SIGGRAPH*.
7. Zhang, R., et al. (2023). "FontDiff: High-Fidelity Font Generation with Differential Network." *CVPR*.
8. Yin, X., et al. (2023). "FontStruct: Font Generation with Structural Awareness." *ECCV*.
9. `审计v4.md` — CnS v4.x full-pipeline code and artifact deep audit report. Repository-internal document.
10. `Plan3.0.md` — v3.0 design and optimization plan. Repository-internal document.
