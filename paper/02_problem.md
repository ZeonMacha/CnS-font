# 2. Problem Formulation and Data Pipeline

## 2.1 Target Character Set: Tongui-8105

The localization target is the **《通用规范汉字表》** (Tongui-8105, "Standard Table of Regularly Used Chinese Characters"), published by the State Language Commission of China. It defines 8,105 characters in three frequency tiers:

| Tier | Count | Description |
|:---|:---:|:---|
| Level 1 | 3,500 | Most frequent (covers ~99% of printed text) |
| Level 2 | 3,000 | Secondary frequency |
| Level 3 | 1,605 | Extended (names, rare characters) |

The source font `FOT-SkipPro-B.OTF` covers 7,912 CJK codepoints in the URO range `0x4E00–0x9FFF`. The delta between the target set and the source font yields two disjoint work queues:

- **Missing**: Characters in Tongui-8105 with no glyph in the source font → 4,101 targets
- **SC-Variant**: Characters present in the source font whose Japanese form differs from the Simplified Chinese form → 1,320 targets

## 2.2 Glyph Classification Pipeline

Before any neural synthesis, every Tongui-8105 character passes through a **five-stage classifier** (`variant_classifier.py`) that assigns it to one of six mutually exclusive categories:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Tongui-8105 (8,105 chars)                       │
├─────────────────────────────────────────────────────────────────────┤
│ 1. HARD_KEEP_12     (12 chars)                                     │
│    Policy-locked characters that must retain the original glyph.    │
│    Point-exact verification against source OTF.                    │
├─────────────────────────────────────────────────────────────────────┤
│ 2. KEEP_SAME_GID    (1,453 chars)                                  │
│    JP and SC forms share the same glyph ID in Noto Sans CJK.       │
│    No difference → keep original SkipPro glyph.                    │
├─────────────────────────────────────────────────────────────────────┤
│ 3. KEEP_V1_V2       (349 chars)                                    │
│    Minor JP↔SC differences (e.g., 竖弯钩 ↔ 竖+提, ネ ↔ 礻).      │
│    Below replacement threshold → keep original.                    │
├─────────────────────────────────────────────────────────────────────┤
│ 4. KEEP_ALREADY_SC  (874 chars)                                    │
│    SkipPro glyph already matches SC form (verified via gate).      │
│    No redraw needed.                                               │
├─────────────────────────────────────────────────────────────────────┤
│ 5. MISSING          (4,101 chars)                                  │
│    No glyph in source font → full neural synthesis (Tier 3).       │
├─────────────────────────────────────────────────────────────────────┤
│ 6. SC_VARIANT       (1,320 chars)                                  │
│    JP form differs from SC form → radical replacement (Tier 3).    │
└─────────────────────────────────────────────────────────────────────┘
```

**v4.4 dispatch**: In pure-v4 mode (default since `47593f5`), both `MISSING` and `SC_VARIANT` characters are routed **entirely to Tier 3** (GlobalGlyphTransformer full-glyph SDF synthesis). The legacy v3 tiers (Tier 0a vector surgery, Tier 0b local inpainting, Tier 1 component splicing, Tier 2 atomic assembly) are available via `--use_legacy_v3_tiers` but disabled by default to eliminate mask-boundary seam artifacts.

## 2.3 Input Representation: Four-Channel Deskinned Encoding

The core design choice for the neural input is a **four-channel "deskinned" tensor** that strips stylistic weight information from a geometrically neutral reference font, leaving only the structural skeleton.

### Reference Font Selection

**Noto Sans CJK** (Variable Font, `NotoSansCJK-VF.ttc`) is used as the geometric reference. Two sub-families are extracted:
- `sn_jp`: Japanese language variants (`language="ja"`, `ttc_index=0`)
- `sn_sc`: Simplified Chinese variants (`language="zh-Hans"`, `ttc_index=2`)

The reference is rendered at **Sans Light weight** (`wght=300`), deliberately lighter than the target SkipPro-B (Bold), to create a **style gap** that the network must bridge. This "deskinned" input prevents the network from simply copying the reference's weight and forces it to learn the SkipPro-B-specific stroke thickening, terminal flare, and hook curvature.

> **Important**: The v4.x training and inference pipeline uses `weight=None` (the font's default `wght=100` Thin for the VF), which creates an even larger style gap. The v4.5 roadmap (Section 8) recommends explicit `--noto_weight 300` to align with the Plan 3.0 design intent.

### Channel Construction

For each character, the reference font's outline (list of cubic Bézier segments) is rasterized at $2S = 512$ pixels (for anti-aliasing) and downsampled to $S = 256$. The four channels are:

**Channel 0 — $\phi_{\text{light}}$**: Normalized SDF of the reference glyph.
$$\phi_{\text{light}}(\mathbf{p}) = \text{clip}\left(\frac{d(\mathbf{p}, \mathcal{C}_{\text{ref}})}{T_{\text{trunc}} \cdot \text{u/px}},\ -1,\ +1\right)$$
where $d(\mathbf{p}, \mathcal{C}_{\text{ref}})$ is the signed Euclidean distance (positive outside) and $T_{\text{trunc}} = 16$ px.

**Channel 1 — $\phi_{\text{skel}}$**: Skeleton distance field with uniform half-width.
$$\phi_{\text{skel}}(\mathbf{p}) = \text{clip}\left(\frac{r_u - d(\mathbf{p}, \mathcal{S}_{\text{ref}})}{T_{\text{trunc}} \cdot \text{u/px}},\ -1,\ +1\right)$$
where $\mathcal{S}_{\text{ref}}$ is the medial-axis skeleton of the reference glyph (computed via `skimage.morphology.skeletonize`) and $r_u = 40$ UPEM is a uniform half-width parameter. This channel encodes **stroke width** independently of the outer contour.

**Channels 2–3 — $(\cos 2\theta, \sin 2\theta)$**: Orientation field derived from the skeleton distance gradient.
$$\nabla d(\mathbf{p}, \mathcal{S}_{\text{ref}}) = (g_x, g_y), \quad \cos 2\theta = \frac{g_y^2 - g_x^2}{g_x^2 + g_y^2} \cdot e^{-d/65}, \quad \sin 2\theta = \frac{-2g_x g_y}{g_x^2 + g_y^2} \cdot e^{-d/65}$$
The double-angle encoding makes the field **invariant to 180° rotation** (stroke direction is undirected). The exponential decay $e^{-d/65}$ localizes the orientation to the skeleton neighborhood.

### Training Augmentation

On-the-fly augmentation is applied to both input and target:
- **Weight jitter**: $\phi_{\text{light}} \leftarrow \text{clip}(\phi_{\text{light}} + \delta_w)$, $\delta_w \sim \mathcal{U}[-0.08, +0.10]$
- **Half-width jitter**: $\phi_{\text{skel}} \leftarrow \text{clip}(\phi_{\text{skel}} + (r_u' - r_u)/T_{\text{trunc}})$, $r_u' \sim \mathcal{U}[32, 48]$
- **Rotation**: $\theta \sim \mathcal{U}[-2°, +2°]$ (50% probability), applied synchronously to $\mathbf{x}$ and $\mathbf{y}$; orientation channels rotated by $2\theta$
- **Scale**: $s \sim \mathcal{U}[0.95, 1.05]$ (50% probability), applied synchronously

All augmentations use bilinear interpolation with explicit background fill ($-1.0$ for SDF channels, $0.0$ for orientation channels).

## 2.4 Target SDF Construction

The target $\mathbf{y}$ is the normalized SDF of the **SkipPro-B original glyph** (for SC-Variant characters) or a **zero-filled background** (for Missing characters with no SkipPro glyph). For Missing characters, the target is the Noto SC glyph's SDF — the network must learn to generate the SC form from the JP reference geometry.

The target SDF is computed at $2S = 512$ resolution for sub-pixel accuracy, then downsampled to $S = 256$ with bilinear filtering.

## 2.5 Training Data Statistics

| Property | Value |
|:---|:---|
| Total training characters | ~6,000+ (all SkipPro CJK minus 26-char holdout) |
| Canvas size | 256 × 256 px |
| Input channels | 4 (SDF, skeleton SDF, cos2θ, sin2θ) |
| Target channels | 1 (SDF) |
| Cache format | int8 (×127 quantization) |
| Holdout set | 26 characters (simplified radicals for generalization testing) |
| Batch size | 16 |
| Training steps/epoch | ~375 |

## 2.6 Data Lineage and Hash Verification

Per the engineering standard (`AGENTS.md` §4.4), all artifacts in the pipeline carry SHA-256 fingerprints:

1. **Tokenizer/ID tables**: `data_v2/ids.txt` (88,939 lines from `cjkvi-ids`) — SHA-256 verified on every sync
2. **SDF cache**: `datasets/v4_inputs_*.npy` + `v4_targets_*.npy` — cached with tag encoding holdout size and canvas size
3. **Model checkpoint**: `checkpoints/v44_zh_sdf_global.pt` — SHA-256 verified on remote→local transfer
4. **Output OTFs**: Each `CnS-SkipPro-B-v44-*.otf` — SHA-256 verified post-build
5. **Glyph manifest**: `outputs/v44_glyph_manifest.json` (5,421 entries) — per-character geometric diagnostics

**Artifact invalidation rule**: If the reference font (Noto Sans CJK) or its weight setting changes, all SDF caches must be regenerated. The cache filename tag (`v4_full_n{holdout}_s{canvas}`) currently does **not** encode the font weight or hash — this is a known v4.5 fix item.
