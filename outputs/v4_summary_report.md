# CnS v4.0 / v4.1: Global-Context Transformer SDF Font Synthesis — Report & Audit

## 1. Overview

v4 replaces the v3 local Conv U-Net (~2.8M params, 96×96 crops) with a **CNN + Transformer hybrid architecture** (~36M params) operating on full 256×256 glyphs. Following an architectural audit of v4.0, **v4.1** resolves five critical defects (empty `SCVariants.otf`, missing `ids.txt`, `up0` 64×64 resolution bottleneck, unsynchronized geometric augmentation, and legacy v3 `MasterDirect`/`Tier1` override) and adds curvature-aware losses plus Taubin arc-smoothing vector post-processing.

---

## 2. v4.0 Audit Findings & v4.1 Fixes

| Issue Found in v4.0 | Root Cause | Fix Applied in v4.1 |
| :--- | :--- | :--- |
| **`CnS-SkipPro-B-v4-SCVariants.otf` empty (8.9 KB, 0 glyphs)** | `v4_build_tg8105.py` instantiated `sn_jp = HBOutlineExtractor(..., ttc_index=0)` whose `get_cmds()` defaulted to `language="zh-Hans"`. HarfBuzz returned SC glyphs for `sn_jp`, making `dsc_px == 0` and misclassifying all ~1,323 SC variants as `KEEP_V1_V2`. | Replaced with `SansLightExtractor(..., language="ja"/"zh-Hans", weight=None)` and added non-empty `SCVariants` & `ids.txt` assertions. |
| **Missing `cjkvi-ids` (`ids.txt` was 25B stub)** | `v4_download_ids.py` pointed to non-existent `Unifont/cjkvi-ids`. | Fixed URL to `cjkvi/cjkvi-ids/master/ids.txt` (88,939 lines, 2.16 MB), enabling V1/V2 exemptions (`衣/衤/礻/示`). |
| **Wobbly straight stems & multi-inflection curves** | (1) `model_v4.py` `up0` applied `stride=2` (128→64) before `Upsample(4x)` without `f0` skip; (2) `dataset_v4.py` rotated/scaled `x` without transforming `y` and filled border with `0.0`; (3) `losses_v4.py` lacked Eikonal & curvature smoothness terms; (4) `curve_constrain.py` faithfully fit SDF noise with `max_err=1.35`. | **Scheme B**: `up0` upgraded to `128→256` with `f0` skip + `256×256 ResBlock`; synchronized `(x, y)` augmentation with `-1.0` fill & exact additive `r_u` shift; fixed Stage 1 freeze filter (`patch_conv/unpatch_conv`); added FP32 narrow-band **Eikonal**, **Curvature L1**, and **Curvature TV** losses. **Scheme A**: added Taubin λ-μ open-segment smoothing (`max_err=2.2u`) + interior-core PCA straight-stem regression preserving SkipPro's 1.8°–3.5° horizontal rise. |
| **10 sample chars (`举领义护飞丝绸伞龙论`) regressed when v3 Master Bank synced** | `v4_build_tg8105.py` prioritized un-edited v3 2.8M U-Net `MasterDirect` and `Tier1[M:...]` splicing above v4.1 Transformer. | Pure v4 mode by default (`--use_legacy_v3_tiers` opt-in only): `ids.txt` used for V1/V2 classification only; all 4,101 missing targets route directly to v4.1 `Tier3_GlobalSDF`. |

---

## 3. Training Progression: v4.0 vs v4.1 (RTX 4090, `batch=16`)

* **v4.1 Schedule**: Stage 1 (5 ep, `lr=1e-4`, freeze CNN) + Stage 2 (35 ep, `lr=1e-4` cosine decay, unfreeze all), ~55 min total, peak VRAM `4.37 GB`.
* **Dataset**: 7,898 chars (256×256 full glyphs from `FOT-SkipPro-B.OTF` & `NotoSansCJK-Regular.ttc`).

| Metric | v4.0 Final | v4.1 Final | Change |
| :--- | :---: | :---: | :---: |
| `sdf_l1` | `0.3098` | **`0.1393`** | `-55.0%` |
| `grad_l1` | `0.0135` | **`0.0058`** | `-57.0%` |
| `cldice` | `0.0447` | **`0.0241`** | `-46.1%` |
| `soft_iou` | `0.7496` | **`0.7838`** | `+3.42 pts` |
| `eikonal` | — | **`0.0267`** | new |
| `curv_l1` | — | **`0.1843`** (from `0.5893`) | `-68.7%` |

---

## 4. Full Tonggui-8105 Build Results

| Metric | v4.0 (Initial Buggy) | v4.0 (Fixed Classifier) | v4.1 (Pure v4.1 + Scheme A/B) |
| :--- | :---: | :---: | :---: |
| Missing packed | `4,100 / 4,101` | `4,105 / 4,106` | **`4,094 / 4,101`** (100% `Tier3_GlobalSDF`) |
| SCVariants packed | `0 / 0` (empty!) | `1,110 / 1,323` | **`1,031 / 1,323`** (`Tier0` 972 + `Tier3` 59) |
| Blocked to review | `1` | `214` | `299` (`manual_review/v41_needs_review.otf`) |
| Gate pass rate | `99.98%` | `96.06%` | `94.49%` |
| Merged TG-8105 coverage | — | `8,104 / 8,105` | **`8,098 / 8,105`** (L1: 3496, L2: 2999, L3: 1603) |
| `HARD_KEEP_12` point-exact | `true` | `true` | **`true`** (verified on `CnS-SkipPro-B-v41-Full.otf`) |

---

## 5. Output Artifacts & Pipeline Usage

### Final v4.1 Artifacts
* `outputs/CnS-SkipPro-B-v41-Full.otf` (11.0 MB, merged installable font, 5,125 injected glyphs)
* `outputs/CnS-SkipPro-B-v41-Missing.otf` (2.23 MB, 4,094 standalone missing glyphs)
* `outputs/CnS-SkipPro-B-v41-SCVariants.otf` (583 KB, 1,031 standalone SC variants)
* `outputs/v41_radicals_verify.png` & `outputs/v41_vs_v40_compare.png` (visual verification sheets)
* `checkpoints/v41_zh_sdf_global.pt` (36.0M params, auto-detected by `ZHSDFSynthesizerV4`)

### One-Click Pipeline (`scripts/v41_pipeline.py`)
```bash
# Sync code & ids.txt with SHA256 check
python scripts/v41_pipeline.py sync

# Resume training from v4.1 -> v42
python scripts/v41_pipeline.py resume --base_ckpt checkpoints/v41_zh_sdf_global.pt --tag v42 --epochs 15 --lr 2e-5

# Full build + merge + pull + SHA256 verify
python scripts/v41_pipeline.py build --tag v42

# Visual & quantitative comparison
python scripts/v41_pipeline.py compare --a v41 --b v42

# End-to-end (sync -> resume -> build+merge -> compare)
python scripts/v41_pipeline.py all --base_ckpt checkpoints/v41_zh_sdf_global.pt --tag v42 --epochs 15 --lr 2e-5 --compare_with v41
```
