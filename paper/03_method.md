# 3. Method: GlobalGlyphTransformer v4.4 Architecture

## 3.1 Design Philosophy

The GlobalGlyphTransformer is a **hybrid CNN–Transformer–CNN** encoder-decoder that predicts a full 256×256 SDF for a CJK glyph given four-channel deskinned input. The design follows three principles:

1. **Global context**: CJK glyph style is determined by the **whole-character structure** (radical arrangement, proportional balance, whitespace distribution), not just local stroke neighborhoods. A 4-layer Transformer operating on 256 patch tokens captures inter-radical relationships that a pure CNN with 3×3 kernels cannot.

2. **Multi-scale feature fusion**: The CNN encoder produces features at four resolutions (256→128→64→32→16). The Transformer operates at the coarsest level (16×16 = 256 tokens, each 768-dim). The CNN decoder upsampled with **skip connections from all encoder levels**, including a critical **f0 (128×128) skip** added in v4.1 that preserves fine stroke detail lost in the 16×16 bottleneck.

3. **Direct SDF prediction (v4.4)**: The decoder's spatial features and the warped input channels are jointly fed into a **two-layer convolutional SDF head** that directly outputs the predicted SDF. This replaces v4.3's piecewise composition:
   ```
   v4.3: pred = (1-α)·warped_light + α·warped_skel + gate·δ_sdf
   v4.4: pred = tanh(sdf_head([spatial, warped_bases]))
   ```
   The elimination of `α`, `gate`, and `δ_sdf` removes the hard-clip discontinuity at the truncation boundary and the `active_gate` dead-zone that truncated extreme-curvature hooks.

## 3.2 CNN Encoder

The encoder downsamples the 256×256 input through four stages:

```
Input:  (B, 4, 256, 256)
  │
  ├─ Stem:    Conv(4→48, 7×7, s2) + GN + SiLU + ResBlock(48)  → 48 @ 128×128
  ├─ Down1:   Conv(48→96, 3×3, s2) + GN + SiLU + ResBlock(96) → 96 @ 64×64
  ├─ Down2:   Conv(96→192, 3×3, s2) + GN + SiLU + ResBlock(192)→ 192 @ 32×32
  └─ Down3:   Conv(192→384, 3×3, s2) + GN + SiLU + ResBlock(384)→ 384 @ 16×16
```

Each **ResBlock** is a standard translation-equivariant 3×3 residual block:
```
ResBlock(x) = SiLU(x + Conv3×3(GN(SiLU(Conv3×3(x)))))
```
GroupNorm (G=8) is used instead of BatchNorm for stability with small batch sizes (B=16) and mixed precision (BF16).

**Total encoder parameters**: ~4.2M

## 3.3 Patchify and Transformer Encoder

At the 16×16 spatial level, features are projected to the Transformer's hidden dimension:

```
f3: (B, 384, 16, 16)
  → Conv1×1(384→768):  (B, 768, 16, 16)
  → Reshape:            (B, 256, 768)     [256 tokens, each 768-dim]
  → + Sincos PE:        (B, 256, 768)     [fixed 2D positional encoding]
  → Transformer × 4:    (B, 256, 768)
```

### Positional Encoding

A **fixed 2D Sincos positional encoding** is used (not learned). For grid position $(r, c)$, the encoding is:
$$\text{PE}(r,c) = [\sin(rf_0), \cos(rf_0), \ldots, \sin(rf_{d/4-1}), \cos(rf_{d/4-1}), \sin(cf_0), \ldots, \sin(cf_{d/4-1}), \cos(cf_{d/4-1})]$$
with frequencies $f_k = 10000^{-2k/d}$. The fixed encoding provides inductive bias for spatial locality without consuming parameters.

### Transformer Blocks

Each of the 4 layers uses **Pre-Norm** (LayerNorm before attention/FFN):
```
PreNorm MHA:  x → LN → MultiHeadAttention(12 heads, d=768) → +residual
FFN:          x → LN → Linear(768→3072) → GELU → Linear(3072→768) → +residual
```
Dropout rate: 0.1. Activation: GELU. `batch_first=True`, `enable_nested_tensor=False`.

**Total Transformer parameters**: ~23.3M (4 layers × (MHA ~2.2M + FFN ~3.6M + LN ~0.01M))

## 3.4 CNN Decoder with Skip Connections

The Transformer output is unpatchified and upsampled through four stages with **skip connections from every encoder level**:

```
spatial: (B, 384, 16, 16)  [from unpatch_conv]
  │
  ├─ Up3: Conv(384+384→192, 3×3) + Upsample2× + ResBlock(192)  → 192 @ 32×32
  │       └── skip: f3 (384 @ 16×16) projected via concat
  ├─ Up2: Conv(192+192→96, 3×3) + Upsample2× + ResBlock(96)    → 96 @ 64×64
  │       └── skip: f2 (96 @ 32×32)
  ├─ Up1: Conv(96+96→48, 3×3) + Upsample2× + ResBlock(48)      → 48 @ 128×128
  │       └── skip: f1 (96 @ 64×64)
  └─ Up0 (v4.1+): Conv(48+48→24, 3×3) + Upsample2× + ResBlock(24) → 24 @ 256×256
          └── skip: f0 (48 @ 128×128)  ← CRITICAL: preserves fine detail
```

### The f0 Skip Connection (v4.1 Key Fix)

In v4.0, `up0` was a stride-2 downsample followed by 4× upsample:
```python
# v4.0 (legacy):
up0 = Conv(48→24, 3×3, s2) + GN + SiLU + Upsample4×
```
This **halved the spatial resolution** before the final upsample, creating a 128→256 bottleneck that lost stroke-level detail. v4.1 replaced it with:
```python
# v4.1+:
up0 = Conv(48+48→24, 3×3) + GN + SiLU + Upsample2× + ResBlock(24)
#      └── cat(u1, f0) ──┘
```
The **f0 skip** (48 channels at 128×128) carries the stem's direct view of the input, preserving sub-pixel stroke edges that the 16×16 Transformer bottleneck cannot resolve.

## 3.5 Flow Head and Tensor Warping

The **flow head** predicts a 2-channel displacement field:
$$\mathbf{f}(\mathbf{p}) = \tau_{\max} \cdot \tanh(\text{Conv}_{3\times3}^{24\to2}(\mathbf{u}_0(\mathbf{p})))$$
where $\tau_{\max} = 8.0$ px ($\approx 37.25$ UPEM). The flow field is applied to the **first two input channels** (SDF and skeleton SDF) via bilinear grid sampling:
$$\text{warped}(\mathbf{p}) = \text{interp}([\phi_{\text{light}}, \phi_{\text{skel}}],\ \mathbf{p} + \mathbf{f}(\mathbf{p}))$$

This warping allows the network to **reposition** input contours (e.g., shifting a "子" radical's hook rightward) before the SDF head makes its final prediction. The flow is **zero-initialized** in all versions, so at initialization the warped input equals the original input.

> **Physical reachability limit**: The 8 px flow bound means any target contour feature more than 8 px ($\approx$37 UPEM) from the nearest input contour feature **cannot be reached by warping alone**. The SDF head must generate such features from the spatial features $\mathbf{u}_0$ alone. For the "子" family's extreme hooks (p99 displacement 112–209 UPEM), this is a significant constraint — see Section 7.

## 3.6 Direct SDF Head (v4.4)

The v4.4 SDF head is a two-layer convolutional network:
```python
sdf_head = Sequential(
    CNNBlock(24 + 2 → 48),           # Conv(26→48, 3×3) + GN + SiLU
    Conv2d(48 → 1, 3×3, padding=1),  # No activation (linear output)
)
pred_sdf = tanh(sdf_head(cat([u0, warped_bases], dim=1)))
#         └── (B, 1, 256, 256) ──┘
```

**Input**: Concatenation of the decoder's spatial features $\mathbf{u}_0$ (24 channels) and the warped input channels (2 channels: $\phi_{\text{light}}^{\text{warped}}$, $\phi_{\text{skel}}^{\text{warped}}$), totaling 26 channels.

**Output**: A single-channel SDF in $[-1, +1]$ (tanh activation ensures bounded output compatible with the SDF truncation convention).

### Why Direct Prediction?

The v4.3 piecewise composition had three structural problems:

| Problem | v4.3 Mechanism | v4.4 Fix |
|:---|:---|:---|
| **Hard-clip at truncation** | `pred = clamp(base_blend + gate·δ, -1, +1)` creates a 90° step at $\pm T_{\text{trunc}}$ | `tanh` is smooth everywhere; no hard clamp |
| **Active gate dead-zone** | `active_gate = sigmoid((union_dil + 0.96)·15)` → 0 outside ~107 UPEM radius, forcing `pred → base_blend` | No gate; the SDF head has full canvas access |
| **Blend interpolation** | `base_blend = (1-α)·light + α·skel` is a linear interpolation between two SDFs, which is not a valid SDF operation | The head learns a non-linear mapping from features + warped input to SDF |

## 3.7 Parameter Count Breakdown

| Component | Parameters |
|:---|:---:|
| CNN Encoder (Stem + Down1–3) | ~4.2M |
| Patch Conv (1×1) | 0.295M |
| Transformer (4 layers) | ~23.3M |
| Unpatch Conv (1×1) | 0.295M |
| CNN Decoder (Up3–Up0) | ~3.5M |
| Flow Head (3×3 conv) | 0.173M |
| **SDF Head (v4.4)** | **~0.71M** |
| **Total** | **36,009,267** |

The v4.4 SDF head (0.71M) is slightly larger than the v4.3 refine head (0.68M) due to the additional input channel (26 vs. 26 — same, but the output is 1 channel vs. 2 channels with different internal widths).

## 3.8 Forward Pass Summary

```
x (B, 4, 256, 256)
  │
  ├─ f0 = stem(x)          (B, 48, 128, 128)
  ├─ f1 = down1(f0)        (B, 96, 64, 64)
  ├─ f2 = down2(f1)        (B, 192, 32, 32)
  ├─ f3 = down3(f2)        (B, 384, 16, 16)
  │
  ├─ tokens = patch_conv(f3).reshape(B, 256, 768) + sincos_pe
  ├─ tokens = transformer(tokens)              (B, 256, 768)
  ├─ spatial = unpatch_conv(tokens.reshape(B, 768, 16, 16))  (B, 384, 16, 16)
  │
  ├─ u3 = up3(cat([spatial, f3]))   (B, 192, 32, 32)
  ├─ u2 = up2(cat([u3, f2]))        (B, 96, 64, 64)
  ├─ u1 = up1(cat([u2, f1]))        (B, 48, 128, 128)
  ├─ u0 = up0(cat([u1, f0]))        (B, 24, 256, 256)
  │
  ├─ flow = 8.0 · tanh(flow_head(u0))          (B, 2, 256, 256)
  ├─ warped = grid_sample(cat([x0, x1]), flow)  (B, 2, 256, 256)
  │
  └─ pred_sdf = tanh(sdf_head(cat([u0, warped])))  (B, 1, 256, 256)

Returns: (pred_sdf, flow)
```

**Inference latency**: ~12 ms/glyph on RTX 5070 Ti (BF16, batch=16). Full library (5,421 glyphs) inference: ~65 s.
