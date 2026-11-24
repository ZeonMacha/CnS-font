# 4. Method: Composite Loss Function

## 4.1 Design Principles

The loss function $\mathcal{L}$ is a weighted sum of nine terms, each targeting a specific geometric property of the predicted SDF. The design follows three principles:

1. **Multi-scale supervision**: The loss operates on the full 256×256 SDF (trimmed by 4 px border for gradient computations), combining pixel-wise terms (SDF L1, gradient) with region-wise terms (IoU, clDice) and integral terms (curvature, flow TV).

2. **Topological awareness**: The soft-clDice term explicitly penalizes topological errors (missing strokes, spurious holes) that pixel-wise L1 loss cannot detect.

3. **Physical consistency**: The Eikonal and curvature terms enforce that the predicted SDF is a valid distance function (unit gradient magnitude) with smooth contours, preventing the wobbly, multi-inflection boundaries that plague naive SDF regression.

## 4.2 Loss Terms

All terms are computed on the trimmed SDF ($S-8 = 248$ px) to avoid boundary artifacts in convolution-based gradient estimates.

### Term 1: Boundary-Weighted SDF L1

$$\mathcal{L}_{\text{sdf}} = \frac{1}{N} \sum_{\mathbf{p}} w(\mathbf{p}) \cdot |\hat{\phi}(\mathbf{p}) - \phi^*(\mathbf{p})|$$

where $w(\mathbf{p}) = 1 + \lambda_b \cdot \mathbb{1}[|\phi^*(\mathbf{p})| < 0.35]$ and $\lambda_b = 8.0$. The boundary band ($|\phi^*| < 0.35$, approximately ±0.35 × $T_{\text{trunc}}$ ≈ ±26 UPEM) receives 9× weighting, focusing supervision on the contour where geometric accuracy matters most.

**v4.4 final value**: 0.1313

### Term 2: Surface-Normal Gradient Consistency

$$\mathcal{L}_{\text{grad}} = \frac{1}{N} \sum_{\mathbf{p}} m_g(\mathbf{p}) \cdot \left(|\nabla_x \hat{\phi} - \nabla_x \phi^*| + |\nabla_y \hat{\phi} - \nabla_y \phi^*|\right)$$

where $\nabla$ is the Sobel operator (3×3 kernel, /8 normalization) and $m_g(\mathbf{p}) = \mathbb{1}[|\phi^*(\mathbf{p})| < 0.45] + 0.1$. The edge band ($|\phi^*| < 0.45$) is weighted; the 0.1 floor ensures weak supervision in background regions.

Weight: $\lambda_g = 2.0$.

**v4.4 final value**: 0.00507

### Term 3: Straight-Stem Normal Alignment

$$\mathcal{L}_{\text{stem}} = \frac{\sum_{\mathbf{p} \in \mathcal{M}_{\text{stem}}} \left(|\hat{n}_x - n_x^*| + |\hat{n}_y - n_y^*|\right)}{|\mathcal{M}_{\text{stem}}|}$$

where $\mathcal{M}_{\text{stem}} = \{\mathbf{p} : |\phi^*(\mathbf{p})| < 0.30 \text{ and } |\nabla^2 \phi^*(\mathbf{p})| < 0.16\}$ is the set of near-contour pixels with low curvature (i.e., straight stems). $\hat{n} = \nabla\hat{\phi}/|\nabla\hat{\phi}|$ and $n^* = \nabla\phi^*/|\nabla\phi^*|$ are unit surface normals.

This term enforces that **straight strokes remain straight** — critical for the horizontal/vertical stroke geometry that defines CJK calligraphic style.

Weight: $\lambda_s = 1.6$.

**v4.4 final value**: 0.1017

### Term 4: Soft-clDice Topological Fidelity

The **soft-clDice** (corner-loss Dice) loss measures topological overlap between predicted and target skeletons:

$$\mathcal{L}_{\text{cldice}} = 1 - \frac{2 \cdot \mathcal{P} \cdot \mathcal{S}}{\mathcal{P} + \mathcal{S}}$$

where:
- $\hat{I} = \sigma(16\hat{\phi})$, $I^* = \sigma(16\phi^*)$ are soft binary ink masks (sigmoid with sharpness 16)
- $\hat{S} = \text{Skel}(\hat{I})$, $S^* = \text{Skel}(I^*)$ are **differentiable soft skeletons** (5-iteration morphological thinning via soft erosion/dilation)
- $\mathcal{P} = \frac{\sum \hat{S} \cdot I^*}{\sum \hat{S}}$ (precision: fraction of predicted skeleton on target ink)
- $\mathcal{S} = \frac{\sum S^* \cdot \hat{I}}{\sum S^*}$ (sensitivity: fraction of target skeleton on predicted ink)

The soft skeleton is computed as:
```
skel = 0
img1 = soft_open(img)
skel = ReLU(img - img1)
for i in 1..4:
    img = soft_erode(img)
    img1 = soft_open(img)
    delta = ReLU(img - img1)
    skel = skel + ReLU(delta - skel * delta)
```
All operations are differentiable (soft max/min via `F.max_pool2d` on negated inputs).

**FP32 computation**: The clDice term is computed in FP32 (not BF16) to avoid numerical instability in the iterative thinning.

Weight: $\lambda_c = 0.8$.

**v4.4 final value**: 0.02109

### Term 5: Soft IoU

$$\mathcal{L}_{\text{iou}} = 1 - \frac{\sum \hat{I} \cdot I^*}{\sum (\hat{I} + I^* - \hat{I} \cdot I^*)}$$

A global overlap measure that complements clDice: clDice detects **local** topological errors (missing a stroke), while IoU detects **global** shape mismatch (too thick/thin overall).

Weight: $\lambda_{\text{iou}} = 0.6$.

**v4.4 final value (IoU)**: 0.7858

### Term 6: Flow Total Variation + L2

$$\mathcal{L}_{\text{flow}} = \lambda_{\text{tv}} \cdot \left(\frac{1}{N}\sum_{\mathbf{p}} |\mathbf{f}(\mathbf{p}+\hat{x}) - \mathbf{f}(\mathbf{p})| + \frac{1}{N}\sum_{\mathbf{p}} |\mathbf{f}(\mathbf{p}+\hat{y}) - \mathbf{f}(\mathbf{p})|\right) + \lambda_{\text{l2}} \cdot \frac{1}{N}\sum_{\mathbf{p}} |\mathbf{f}(\mathbf{p})|^2$$

with $\lambda_{\text{tv}} = 0.02$, $\lambda_{\text{l2}} = 0.002$.

The TV term encourages a **smooth deformation field** (no high-frequency warping artifacts). The L2 term regularizes the magnitude of the flow, preventing the network from relying excessively on warping when the SDF head could handle the deformation.

**v4.4 final TV value**: 0.00001 (vs. v4.3's 0.04457 — a 4,457× reduction, indicating the direct SDF head requires far less warping)

### Term 7: Narrow-Band Eikonal

$$\mathcal{L}_{\text{eik}} = \frac{\sum_{\mathbf{p} \in \mathcal{M}_{\text{eik}}} \left(\frac{|\nabla\hat{\phi}(\mathbf{p})| - 1/T_{\text{trunc}}}{1/T_{\text{trunc}}}\right)^2}{|\mathcal{M}_{\text{eik}}|}$$

where $\mathcal{M}_{\text{eik}} = \{\mathbf{p} : |\phi^*(\mathbf{p})| < 0.35 \text{ and } |\nabla\phi^*(\mathbf{p})| > 0.5/T_{\text{trunc}}\}$.

The **Eikonal equation** $|\nabla\phi| = 1$ (in normalized SDF space, $|\nabla\phi| = 1/T_{\text{trunc}}$) is the defining property of a distance function. Violations indicate the predicted SDF is not a valid distance field, leading to contour extraction artifacts.

The mask excludes medial-axis ridges where $|\nabla\phi^*|$ naturally drops to zero.

**FP32 computation**.

Weight: $\lambda_e = 0.25$.

**v4.4 final value**: 0.01455

### Term 8: Mean Curvature Alignment

$$\mathcal{L}_{\text{curv}} = \frac{\sum_{\mathbf{p} \in \mathcal{M}_{\text{curv}}} |\kappa_{\hat{\phi}}(\mathbf{p}) - \kappa_{\phi^*}(\mathbf{p})|}{|\mathcal{M}_{\text{curv}}|}$$

where the mean curvature is computed as $\kappa = \nabla \cdot (\nabla\phi/|\nabla\phi|)$ (divergence of the unit normal), and $\mathcal{M}_{\text{curv}} = \{\mathbf{p} : |\phi^*| < 0.30, |\nabla\phi^*| > 0.5/T_{\text{trunc}}, 0.04 \leq |\nabla^2\phi^*| < 0.55\}$ selects smooth curved strokes (excluding both straight stems and sharp 90° corners).

Curvature values are clamped to $[-2, +2]$ for numerical stability.

**FP32 computation**.

Weight: $\lambda_k = 0.50$.

**v4.4 final value**: 0.1805

### Term 9: Curvature Total Variation

$$\mathcal{L}_{\text{curv\_tv}} = \frac{\sum_{\text{smooth band}} |\kappa(\mathbf{p}+\hat{x}) - \kappa(\mathbf{p})| + |\kappa(\mathbf{p}+\hat{y}) - \kappa(\mathbf{p})|}{|\text{smooth band}|}$$

This penalizes **rapid curvature changes** along smooth arcs and straight stems, suppressing the multi-inflection wobbles that appear when the SDF head overfits to noise. The smooth band mask is $|\phi^*| < 0.30$ and $|\nabla^2\phi^*| < 0.55$.

Weight: $\lambda_{kt} = 0.15$.

**v4.4 final value**: 0.07431

## 4.3 Total Loss

$$\mathcal{L} = \mathcal{L}_{\text{sdf}} + 2.0\mathcal{L}_{\text{grad}} + 1.6\mathcal{L}_{\text{stem}} + 0.8\mathcal{L}_{\text{cldice}} + 0.6\mathcal{L}_{\text{iou}} + \mathcal{L}_{\text{flow}} + 0.25\mathcal{L}_{\text{eik}} + 0.50\mathcal{L}_{\text{curv}} + 0.15\mathcal{L}_{\text{curv\_tv}}$$

## 4.4 Convergence Behavior

### Stage 1 (5 epochs, frozen CNN, trainable Transformer + SDF head)

| Epoch | SDF L1 | IoU | Flow TV | Eikonal | Curv L1 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.9651 | 0.4120 | 0.000 | 0.4381 | 0.7198 |
| 2 | 0.8282 | 0.5190 | 0.000 | 0.2960 | 0.6673 |
| 3 | 0.7956 | 0.5400 | 0.000 | 0.2607 | 0.6499 |
| 4 | 0.7718 | 0.5547 | 0.000 | 0.2436 | 0.6357 |
| 5 | 0.7527 | 0.5658 | 0.000 | 0.2329 | 0.6261 |

**Key observation**: Unlike v4.1/v4.3 where Stage 1 was a **zero-gradient dead zone** (SDF L1 stuck at 0.807, Flow TV = 0.0), v4.4's Stage 1 shows **monotonic SDF L1 decrease** (0.965 → 0.753) because the direct `sdf_head` is not zero-initialized. The Transformer receives meaningful gradients from epoch 1.

### Stage 2 (35 epochs, full end-to-end)

| Epoch | SDF L1 | IoU | Flow TV | Eikonal | Curv L1 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.5998 | 0.6281 | 0.0035 | 0.1542 | 0.5493 |
| 5 | 0.4031 | 0.7210 | 0.0019 | 0.0803 | 0.4584 |
| 10 | 0.3479 | 0.7385 | 0.0018 | 0.0539 | 0.4115 |
| 15 | 0.2827 | 0.7565 | 0.0011 | 0.0370 | 0.3525 |
| 20 | 0.2203 | 0.7711 | 0.0008 | 0.0259 | 0.2886 |
| 25 | 0.1716 | 0.7801 | 0.0004 | 0.0191 | 0.2324 |
| 30 | 0.1408 | 0.7846 | 0.0001 | 0.0156 | 0.1933 |
| 35 | **0.1313** | **0.7858** | **0.00001** | **0.0146** | **0.1805** |

**Convergence profile**: SDF L1 decreases monotonically with a characteristic "fast then slow" curve. The first 10 epochs account for 74% of the total improvement (0.60 → 0.35). The Flow TV drops by 350× over Stage 2, indicating the deformation field becomes increasingly smooth as the SDF head takes on more of the generation task.

**Training time**: 3,274 s (~55 min) on RTX 5070 Ti, BF16 mixed precision, batch=16.

## 4.5 Comparison with v4.3 Loss Trajectory

| Metric | v4.3 Ep1 (S2) | v4.3 Ep35 (S2) | v4.4 Ep1 (S2) | v4.4 Ep35 (S2) | v4.4 vs v4.3 |
|:---|:---:|:---:|:---:|:---:|:---:|
| SDF L1 | 0.6214 | 0.1309 | 0.5998 | 0.1313 | ≈ equal |
| Soft IoU | 0.6582 | 0.7852 | 0.6281 | 0.7858 | v4.4 +0.0006 |
| Flow TV | 0.0233 | **0.0446** | 0.0035 | **0.00001** | **v4.4 4,460× lower** |
| Eikonal | 0.1187 | 0.0162 | 0.1542 | 0.0146 | v4.4 −10% |
| Curv L1 | 0.5047 | 0.1825 | 0.5493 | 0.1805 | ≈ equal |

The most striking difference is the **Flow TV endpoint**: v4.3's flow TV *increased* during Stage 2 (0.023 → 0.045) as the network leaned more on warping to compensate for the `active_gate` dead-zone. v4.4's flow TV *decreases* monotonically (0.0035 → 0.00001) because the direct SDF head can generate new geometry without warping.
