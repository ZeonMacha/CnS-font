"""
v4.6 Dual-Dataset Pipeline: Primary Font Dataset + Optional Extended Simplified-Radical Dataset.

Addresses 审计v46_进度.md R6, R7 & §3:
  1. Primary dataset (`SkipPro` + `Noto Sans JP`), keeping full cache lineage (font stem, weight,
     split, holdout signature, canvas size, radical-mask flag).
  2. Optional extended simplified-radical dataset from either:
       - an OTF/TTF target font (`extra_target_otf`, paired with Noto Sans SC `zh-Hans`), OR
       - a precomputed NPZ archive (`extra_npz`, e.g. self-distilled consensus SDFs from `v46_select.py`
         or vector-surgery pseudo-labels) with optional per-sample confidence weights.
  3. Unified `DualGlyphDatasetV46` returning `(x, y, char, sample_weight, source_id)` with
     synchronized augmentation and optional 5th channel (`radical_mask`).
  4. Stratified holdout builder (`build_stratified_holdout_v46`, >= 200 chars grouped by IDS radical)
     and IDS-aware `WeightedRandomSampler` (`build_stratified_weights_v46`) supporting
     configurable `--extra_sample_boost` or `--extra_mix_ratio`.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from src.sdf_utils import mask_to_normalized_sdf
from src.v2.component_bank import parse_cjkvi_ids
from src.v2.hb_outlines import HBOutlineExtractor, SkipProOutlineExtractor, rasterize_cmds
from src.v45 import UNIFIED_HOLDOUT_28
from src.v45.dataset_v45 import (
    OVERSAMPLE_CHARS,
    FullGlyphDatasetV45,
    build_deskinned_channels,
    pack_input_tensor,
)

# Simplified-radical families that are zero-shot / under-represented in Japanese SkipPro (R6)
SIMPLIFIED_RADICAL_FAMILIES: Tuple[str, ...] = (
    "纟", "讠", "钅", "饣", "车", "贝", "见", "马", "鸟", "鱼", "门", "页", "风", "韦",
)

# Traditional donor counterparts in SkipPro used for stratified validation grouping
VAL_RADICAL_GROUPS: Tuple[str, ...] = (
    "糸", "糹", "纟", "言", "讠", "金", "钅", "食", "飠", "饣", "木", "氵", "扌", "口", "子",
)


def classify_ids_radical_group(ch: str, ids_map: Dict[str, str]) -> str:
    """Map a character to a coarse IDS radical family label for stratified sampling & val reporting."""
    seq = ids_map.get(ch, "")
    if not seq:
        return "other"
    if seq.startswith("⿰") and len(seq) >= 2:
        left = seq[1]
        if left in ("纟", "糹", "糸"):
            return "si_radical"
        if left in ("讠", "言"):
            return "yan_radical"
        if left in ("钅", "金", "釒"):
            return "jin_radical"
        if left in ("饣", "食", "飠"):
            return "shi_radical"
        if left in ("车", "贝", "见", "马", "鸟", "鱼", "门", "页", "风", "韦"):
            return "simp_other"
        if left in ("木", "氵", "扌", "口", "子"):
            return f"left_{left}"
    if "子" in seq:
        return "has_zi"
    for r in SIMPLIFIED_RADICAL_FAMILIES:
        if r in seq:
            return "simp_other"
    return "other"


def build_radical_mask_channel(
    ch: str,
    ids_map: Dict[str, str],
    canvas_size: int = 256,
) -> np.ndarray:
    """
    Build coarse 5th conditioning channel `[S, S]` in [0, 1] highlighting the expected spatial region
    of a left-side simplified/target radical (`⿰` left 42% band) or enclosing radical (`⿵` outer frame).
    Returns zeros for characters without a targeted radical structure.
    """
    mask = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    seq = ids_map.get(ch, "")
    if not seq or len(seq) < 2:
        return mask

    op = seq[0]
    comp = seq[1]
    is_target = (comp in SIMPLIFIED_RADICAL_FAMILIES) or (comp in ("糸", "糹", "言", "金", "釒", "食", "飠", "子"))
    if not is_target:
        return mask

    if op == "⿰":
        # Smooth horizontal ramp covering left ~44% of glyph canvas
        x = np.linspace(0.0, 1.0, canvas_size, dtype=np.float32)
        col = 1.0 / (1.0 + np.exp((x - 0.44) * 20.0))
        mask[:] = col[None, :]
    elif op == "⿱":
        y = np.linspace(0.0, 1.0, canvas_size, dtype=np.float32)
        row = 1.0 / (1.0 + np.exp((y - 0.44) * 20.0))
        mask[:] = row[:, None]
    elif op in ("⿵", "⿴", "⿺", "⿹"):
        # Soft border highlight
        yy, xx = np.meshgrid(
            np.linspace(-1.0, 1.0, canvas_size, dtype=np.float32),
            np.linspace(-1.0, 1.0, canvas_size, dtype=np.float32),
            indexing="ij",
        )
        r = np.maximum(np.abs(xx), np.abs(yy))
        mask[:] = np.clip((r - 0.55) / 0.25, 0.0, 1.0)
    return mask


def build_stratified_holdout_v46(
    all_cjk: Sequence[str],
    ids_map: Dict[str, str],
    target_size: int = 200,
    base_holdout: Optional[Set[str]] = None,
) -> Set[str]:
    """
    Deterministically construct a >= `target_size` stratified holdout set over `all_cjk`
    starting from `UNIFIED_HOLDOUT_28` and sampling proportionally across IDS radical groups.
    """
    avail = set(all_cjk)
    selected: Set[str] = set(c for c in (base_holdout or UNIFIED_HOLDOUT_28) if c in avail)
    if len(selected) >= target_size:
        return selected

    buckets: Dict[str, List[str]] = {}
    for ch in sorted(all_cjk):
        if ch in selected:
            continue
        grp = classify_ids_radical_group(ch, ids_map)
        buckets.setdefault(grp, []).append(ch)

    # Deterministic per-bucket stride sampling
    need = target_size - len(selected)
    group_names = sorted(buckets.keys())
    if not group_names:
        return selected

    # Guarantee minimum quota for key radical groups first, then fill proportionally
    min_per_group = max(4, need // max(len(group_names) * 2, 1))
    picked: List[str] = []
    for grp in group_names:
        pool = buckets[grp]
        take = min(len(pool), min_per_group)
        if take > 0:
            idxs = np.linspace(0, len(pool) - 1, take, dtype=int)
            for i in idxs:
                picked.append(pool[int(i)])

    for ch in picked:
        if len(selected) >= target_size:
            break
        selected.add(ch)

    if len(selected) < target_size:
        rem = [c for c in sorted(all_cjk) if c not in selected]
        still = min(len(rem), target_size - len(selected))
        if still > 0:
            idxs = np.linspace(0, len(rem) - 1, still, dtype=int)
            for i in idxs:
                selected.add(rem[int(i)])

    return selected


def _short_set_hash(chars: Set[str]) -> str:
    h = hashlib.sha256("".join(sorted(chars)).encode("utf-8")).hexdigest()
    return h[:8]


class ExtraRadicalDatasetV46:
    """
    Optional extended simplified-radical dataset loader.
    Loads either:
      - an OTF/TTF target font paired with Noto Sans (`extra_target_otf`), cached to `.npy`, OR
      - a precomputed `.npz` archive (`extra_npz`) containing `chars`, `inputs_i8`/`inputs`,
        `targets_i8`/`targets`, and optional `weights`.
    """

    def __init__(
        self,
        *,
        extra_target_otf: Optional[Path] = None,
        extra_npz: Optional[Path] = None,
        noto_sans: Optional[Path] = None,
        noto_lang: str = "zh-Hans",
        noto_weight: Optional[int] = None,
        cache_dir: Path = Path("datasets"),
        ids_map: Optional[Dict[str, str]] = None,
        radical_filter: Optional[Set[str]] = None,
        canvas_size: int = 256,
        use_cache: bool = True,
    ) -> None:
        self.canvas_size = canvas_size
        self.chars: List[str] = []
        self.inputs_i8: np.ndarray = np.zeros((0, 4, canvas_size, canvas_size), dtype=np.int8)
        self.targets_i8: np.ndarray = np.zeros((0, 1, canvas_size, canvas_size), dtype=np.int8)
        self.conf_weights: np.ndarray = np.zeros((0,), dtype=np.float32)

        if extra_npz is not None:
            self._load_npz(Path(extra_npz), ids_map=ids_map, radical_filter=radical_filter)
        elif extra_target_otf is not None:
            if noto_sans is None or not Path(noto_sans).exists():
                raise ValueError("extra_target_otf requires a valid noto_sans path.")
            self._load_or_build_otf(
                Path(extra_target_otf),
                Path(noto_sans),
                noto_lang=noto_lang,
                noto_weight=noto_weight,
                cache_dir=Path(cache_dir),
                ids_map=ids_map,
                radical_filter=radical_filter,
                use_cache=use_cache,
            )

    def _load_npz(
        self,
        npz_path: Path,
        ids_map: Optional[Dict[str, str]],
        radical_filter: Optional[Set[str]],
    ) -> None:
        if not npz_path.exists():
            raise FileNotFoundError(f"Extra NPZ dataset not found: {npz_path}")
        data = np.load(npz_path, allow_pickle=False)
        chars_raw = [str(c) for c in data["chars"].tolist()]

        if "inputs_i8" in data:
            inp = data["inputs_i8"].astype(np.int8)
        else:
            inp = np.clip(np.round(data["inputs"].astype(np.float32) * 127.0), -127, 127).astype(np.int8)

        if "targets_i8" in data:
            trg = data["targets_i8"].astype(np.int8)
        else:
            t_f = data["targets"].astype(np.float32)
            if t_f.ndim == 3:
                t_f = t_f[:, None, :, :]
            trg = np.clip(np.round(t_f * 127.0), -127, 127).astype(np.int8)

        if "weights" in data:
            wts = data["weights"].astype(np.float32)
        else:
            wts = np.ones(len(chars_raw), dtype=np.float32)

        if radical_filter and ids_map:
            keep_idx = [
                i for i, ch in enumerate(chars_raw)
                if any(r in ids_map.get(ch, "") for r in radical_filter)
            ]
        else:
            keep_idx = list(range(len(chars_raw)))

        self.chars = [chars_raw[i] for i in keep_idx]
        self.inputs_i8 = inp[keep_idx, :4] if keep_idx else np.zeros((0, 4, self.canvas_size, self.canvas_size), dtype=np.int8)
        self.targets_i8 = trg[keep_idx] if keep_idx else np.zeros((0, 1, self.canvas_size, self.canvas_size), dtype=np.int8)
        self.conf_weights = wts[keep_idx] if keep_idx else np.zeros((0,), dtype=np.float32)
        print(f"[v46-ExtraDS] Loaded NPZ {npz_path.name}: {len(self.chars)} chars", flush=True)

    def _load_or_build_otf(
        self,
        target_otf: Path,
        noto_sans: Path,
        *,
        noto_lang: str,
        noto_weight: Optional[int],
        cache_dir: Path,
        ids_map: Optional[Dict[str, str]],
        radical_filter: Optional[Set[str]],
        use_cache: bool,
    ) -> None:
        if not target_otf.exists():
            raise FileNotFoundError(f"Extra target OTF not found: {target_otf}")

        cache_dir.mkdir(parents=True, exist_ok=True)
        w_tag = f"w{noto_weight}" if noto_weight is not None else "wstatic"
        f_tag = f"rf{len(radical_filter)}_{_short_set_hash(radical_filter)}" if radical_filter else "rfall"
        tag = f"v46_extra_{target_otf.stem}_{noto_sans.stem}_{noto_lang}_{w_tag}_{f_tag}_s{self.canvas_size}"
        in_p = cache_dir / f"inputs_{tag}.npy"
        tg_p = cache_dir / f"targets_{tag}.npy"
        meta_p = cache_dir / f"meta_{tag}.json"
        expected_rf = sorted(radical_filter) if radical_filter else None

        if use_cache and in_p.exists() and tg_p.exists() and meta_p.exists():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            if meta.get("radical_filter") == expected_rf and meta.get("noto_weight") == noto_weight:
                self.inputs_i8 = np.load(in_p)
                self.targets_i8 = np.load(tg_p)
                self.chars = meta["chars"]
                self.conf_weights = np.ones(len(self.chars), dtype=np.float32)
                print(f"[v46-ExtraDS] Loaded OTF cache {in_p.name}: {len(self.chars)} chars", flush=True)
                return

        ext = SkipProOutlineExtractor(target_otf)
        all_cjk = sorted(chr(cp) for cp in ext.cmap if 0x4E00 <= cp <= 0x9FFF)
        ext.close()

        if radical_filter and ids_map:
            target_chars = [c for c in all_cjk if any(r in ids_map.get(c, "") for r in radical_filter)]
        else:
            target_chars = all_cjk

        n = len(target_chars)
        print(
            f"[v46-ExtraDS] Building OTF cache: {n} chars "
            f"(target={target_otf.name}, ref={noto_sans.name} [{noto_lang}, w={noto_weight}])...",
            flush=True,
        )
        inputs_i8 = np.empty((n, 4, self.canvas_size, self.canvas_size), dtype=np.int8)
        targets_i8 = np.empty((n, 1, self.canvas_size, self.canvas_size), dtype=np.int8)
        ttc_idx = 0 if noto_lang == "ja" else 2
        trunc_px = 16.0
        tls = threading.local()

        def _worker(item: Tuple[int, str]) -> Tuple[int, np.ndarray, np.ndarray]:
            i, ch = item
            if not hasattr(tls, "sn"):
                tls.sn = HBOutlineExtractor(noto_sans, ttc_index=ttc_idx, weight=noto_weight)
                tls.tg = SkipProOutlineExtractor(target_otf)

            c_ref = tls.sn.get_gid_and_cmds(ch, noto_lang)[1]
            sl, dtu, c2, s2 = build_deskinned_channels(
                c_ref, canvas_size=self.canvas_size, padding=96.0, trunc_px=trunc_px
            )
            x_f = pack_input_tensor(
                sl, dtu, c2, s2,
                uniform_half_width_u=40.0,
                canvas_size=self.canvas_size,
                padding=96.0,
                trunc_px=trunc_px,
            )
            c_tg = tls.tg.get_cmds(ch)
            if not c_tg:
                y_f = np.full((1, self.canvas_size, self.canvas_size), -1.0, dtype=np.float32)
            else:
                m_tg = rasterize_cmds(c_tg, canvas_size=self.canvas_size * 2, padding=96.0)
                y_f = mask_to_normalized_sdf(m_tg, self.canvas_size, trunc_px=trunc_px)[None, :, :]
            return (
                i,
                np.clip(np.round(x_f * 127.0), -127, 127).astype(np.int8),
                np.clip(np.round(y_f * 127.0), -127, 127).astype(np.int8),
            )

        workers = min(10, max(4, os.cpu_count() or 8))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, xi, yi in tqdm(
                ex.map(_worker, enumerate(target_chars)),
                total=n,
                desc="v46 Extra Cache",
                mininterval=5.0,
            ):
                inputs_i8[i] = xi
                targets_i8[i] = yi

        np.save(in_p, inputs_i8)
        np.save(tg_p, targets_i8)
        meta = {
            "target_otf": target_otf.name,
            "noto_sans": noto_sans.name,
            "noto_lang": noto_lang,
            "noto_weight": noto_weight,
            "radical_filter": expected_rf,
            "n_chars": n,
            "canvas_size": self.canvas_size,
            "chars": target_chars,
        }
        meta_p.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        self.chars = target_chars
        self.inputs_i8 = inputs_i8
        self.targets_i8 = targets_i8
        self.conf_weights = np.ones(n, dtype=np.float32)


class DualGlyphDatasetV46(Dataset):
    """
    Unified dual-source dataset combining:
      - source 0: Primary font dataset (SkipPro + Noto JP)
      - source 1: Optional extended simplified-radical dataset (OTF or NPZ)

    Each item returns: `(x [C, S, S], y [1, S, S], ch, sample_weight, source_id)`.
    """

    def __init__(
        self,
        primary_chars: List[str],
        primary_inputs_i8: np.ndarray,
        primary_targets_i8: np.ndarray,
        *,
        extra_chars: Optional[List[str]] = None,
        extra_inputs_i8: Optional[np.ndarray] = None,
        extra_targets_i8: Optional[np.ndarray] = None,
        extra_conf_weights: Optional[np.ndarray] = None,
        extra_loss_weight: float = 0.85,
        ids_map: Optional[Dict[str, str]] = None,
        use_radical_mask: bool = False,
        canvas_size: int = 256,
        augment: bool = True,
    ) -> None:
        self.canvas_size = canvas_size
        self.augment = augment
        self.use_radical_mask = use_radical_mask
        self.ids_map = ids_map or {}

        self.primary_chars = list(primary_chars)
        self.primary_in = primary_inputs_i8
        self.primary_tg = primary_targets_i8
        self.n_primary = len(self.primary_chars)

        if extra_chars and extra_inputs_i8 is not None and extra_targets_i8 is not None and len(extra_chars) > 0:
            self.extra_chars = list(extra_chars)
            self.extra_in = extra_inputs_i8
            self.extra_tg = extra_targets_i8
            if extra_conf_weights is not None and len(extra_conf_weights) == len(self.extra_chars):
                self.extra_w = (extra_conf_weights.astype(np.float32) * float(extra_loss_weight))
            else:
                self.extra_w = np.full(len(self.extra_chars), float(extra_loss_weight), dtype=np.float32)
        else:
            self.extra_chars = []
            self.extra_in = np.zeros((0, 4, canvas_size, canvas_size), dtype=np.int8)
            self.extra_tg = np.zeros((0, 1, canvas_size, canvas_size), dtype=np.int8)
            self.extra_w = np.zeros((0,), dtype=np.float32)
        self.n_extra = len(self.extra_chars)

        self.chars = self.primary_chars + self.extra_chars
        self.source_ids = [0] * self.n_primary + [1] * self.n_extra

    def __len__(self) -> int:
        return self.n_primary + self.n_extra

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, float, int]:
        if idx < self.n_primary:
            ch = self.primary_chars[idx]
            x = torch.from_numpy(self.primary_in[idx]).float().mul_(1.0 / 127.0)
            y = torch.from_numpy(self.primary_tg[idx]).float().mul_(1.0 / 127.0)
            sw = 1.0
            src_id = 0
        else:
            j = idx - self.n_primary
            ch = self.extra_chars[j]
            x = torch.from_numpy(self.extra_in[j]).float().mul_(1.0 / 127.0)
            y = torch.from_numpy(self.extra_tg[j]).float().mul_(1.0 / 127.0)
            sw = float(self.extra_w[j])
            src_id = 1

        if self.use_radical_mask:
            rmask = torch.from_numpy(build_radical_mask_channel(ch, self.ids_map, self.canvas_size)).unsqueeze(0)
            x = torch.cat([x, rmask], dim=0)

        if not self.augment:
            return x, y, ch, sw, src_id

        # 1. Weight jitter on ch0
        dw = float(np.random.uniform(-0.08, 0.10))
        x[0] = torch.clamp(x[0] + dw, -1.0, 1.0)

        # 2. Exact additive uniform half-width jitter on ch1
        upem_per_px = (1000.0 + 2.0 * 96.0) / float(self.canvas_size)
        trunc_u = 16.0 * upem_per_px
        r_u_new = float(np.random.uniform(32.0, 48.0))
        x[1] = torch.clamp(x[1] + (r_u_new - 40.0) / trunc_u, -1.0, 1.0)

        # 3. Synchronized rotation (±2 deg) on (x, y)
        if np.random.random() < 0.5:
            angle = float(np.random.uniform(-2.0, 2.0))
            x_sdf = FullGlyphDatasetV45._rotate_tensor(x[0:2], angle, fill=-1.0)
            x_ori = FullGlyphDatasetV45._rotate_tensor(x[2:4], angle, fill=0.0)
            rad2 = float(np.deg2rad(-2.0 * angle))
            c2a, s2a = float(np.cos(rad2)), float(np.sin(rad2))
            ori_c = x_ori[0] * c2a - x_ori[1] * s2a
            ori_s = x_ori[0] * s2a + x_ori[1] * c2a
            parts = [x_sdf, torch.stack([ori_c, ori_s], dim=0)]
            if x.size(0) > 4:
                parts.append(FullGlyphDatasetV45._rotate_tensor(x[4:], angle, fill=0.0))
            x = torch.cat(parts, dim=0)
            y = FullGlyphDatasetV45._rotate_tensor(y, angle, fill=-1.0)

        # 4. Synchronized scale ([0.95, 1.05]) on (x, y)
        if np.random.random() < 0.5:
            s = float(np.random.uniform(0.95, 1.05))
            x_sdf = FullGlyphDatasetV45._scale_tensor(x[0:2], s, fill=-1.0)
            x_ori = FullGlyphDatasetV45._scale_tensor(x[2:4], s, fill=0.0)
            parts = [x_sdf, x_ori]
            if x.size(0) > 4:
                parts.append(FullGlyphDatasetV45._scale_tensor(x[4:], s, fill=0.0))
            x = torch.cat(parts, dim=0)
            y = FullGlyphDatasetV45._scale_tensor(y, s, fill=-1.0)

        return x, y, ch, sw, src_id


def build_stratified_weights_v46(
    dataset: DualGlyphDatasetV46,
    ids_map: Dict[str, str],
    *,
    radical_boost: float = 3.0,
    extra_sample_boost: float = 3.5,
    extra_mix_ratio: Optional[float] = None,
) -> List[float]:
    """
    Build per-sample weights for `WeightedRandomSampler`:
      - Primary glyphs with simplified/target radicals or rare `子`/sparse archetypes get `radical_boost`.
      - Extra simplified-radical dataset glyphs get `extra_sample_boost` (or auto-calibrated to hit
        `extra_mix_ratio` fraction of expected draws per epoch when `extra_mix_ratio` is set).
    """
    primary_w: List[float] = []
    n_p_boost = 0
    for ch in dataset.primary_chars:
        seq = ids_map.get(ch, "")
        grp = classify_ids_radical_group(ch, ids_map)
        is_boosted = (
            (ch in OVERSAMPLE_CHARS)
            or (grp in ("si_radical", "yan_radical", "jin_radical", "shi_radical", "simp_other", "has_zi"))
            or (seq.startswith("⿰") and "子" in seq)
        )
        if is_boosted:
            primary_w.append(float(radical_boost))
            n_p_boost += 1
        else:
            primary_w.append(1.0)

    if dataset.n_extra == 0:
        print(
            f"[v46-Dataset] Sampler (primary only): boosted {n_p_boost}/{dataset.n_primary} "
            f"radical/rare chars by {radical_boost:.1f}x",
            flush=True,
        )
        return primary_w

    if extra_mix_ratio is not None and 0.0 < extra_mix_ratio < 0.9:
        sum_p = float(sum(primary_w))
        target_extra_sum = sum_p * (extra_mix_ratio / (1.0 - extra_mix_ratio))
        eff_extra_w = target_extra_sum / max(dataset.n_extra, 1)
    else:
        eff_extra_w = float(extra_sample_boost)

    extra_w = [eff_extra_w] * dataset.n_extra
    tot_p = float(sum(primary_w))
    tot_e = float(sum(extra_w))
    share_e = tot_e / max(tot_p + tot_e, 1e-6)
    print(
        f"[v46-Dataset] Dual Sampler: primary={dataset.n_primary} (boosted {n_p_boost}x{radical_boost:.1f}) | "
        f"extra={dataset.n_extra} (w={eff_extra_w:.2f}, epoch_share={share_e*100:.1f}%)",
        flush=True,
    )
    return primary_w + extra_w


def create_v46_dataloaders(
    skip_otf: Path,
    noto_sans: Path,
    cache_dir: Path,
    *,
    noto_weight: Optional[int] = None,
    ids_path: Optional[Path] = None,
    canvas_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 4,
    val_target_size: int = 200,
    use_radical_mask: bool = False,
    radical_boost: float = 3.0,
    # Optional extended simplified-radical dataset args
    extra_target_otf: Optional[Path] = None,
    extra_npz: Optional[Path] = None,
    extra_noto_sans: Optional[Path] = None,
    extra_noto_lang: str = "zh-Hans",
    extra_noto_weight: Optional[int] = None,
    extra_radicals: Optional[Set[str]] = None,
    extra_loss_weight: float = 0.85,
    extra_sample_boost: float = 3.5,
    extra_mix_ratio: Optional[float] = None,
    extra_val_ratio: float = 0.10,
) -> Tuple[
    DualGlyphDatasetV46,
    DataLoader,
    DualGlyphDatasetV46,
    DataLoader,
    Optional[DataLoader],
    Dict[str, str],
]:
    """
    Create v4.6 train/val DataLoaders (plus optional `val_extra_loader` when an extra dataset is supplied).
    """
    ids_map: Dict[str, str] = parse_cjkvi_ids(ids_path) if (ids_path and ids_path.exists()) else {}

    # 1. Build deterministic stratified holdout set over SkipPro CJK chars
    skip_ext = SkipProOutlineExtractor(skip_otf)
    all_skip_cjk = sorted(chr(cp) for cp in skip_ext.cmap if 0x4E00 <= cp <= 0x9FFF)
    skip_ext.close()
    holdout = build_stratified_holdout_v46(
        all_skip_cjk, ids_map, target_size=val_target_size, base_holdout=UNIFIED_HOLDOUT_28
    )

    # 2. Primary train & val caches: verify exact holdout character-set lineage before reusing
    # an existing `h{N}` cache; isolate into a hash-tagged subdirectory if holdout composition differs
    # (prevents silent train/val leakage without overwriting existing caches, per AGENTS.md §4.4).
    font_stem = Path(noto_sans).stem.replace(" ", "_")
    w_tag = f"w{noto_weight}" if noto_weight is not None else "wstatic"
    sorted_holdout = sorted(holdout)
    primary_cache_dir = Path(cache_dir)
    for split_chk in ("train", "val"):
        legacy_meta = primary_cache_dir / f"meta_v45_{font_stem}_{w_tag}_{split_chk}_h{len(holdout)}_s{canvas_size}.json"
        if legacy_meta.exists():
            m_info = json.loads(legacy_meta.read_text(encoding="utf-8"))
            if m_info.get("holdout") != sorted_holdout:
                primary_cache_dir = Path(cache_dir) / f"holdout_{_short_set_hash(holdout)}"
                break

    primary_train = FullGlyphDatasetV45(
        skip_otf, noto_sans, primary_cache_dir,
        noto_weight=noto_weight, canvas_size=canvas_size,
        holdout_chars=holdout, split="train", augment=False,
    )
    primary_val = FullGlyphDatasetV45(
        skip_otf, noto_sans, primary_cache_dir,
        noto_weight=noto_weight, canvas_size=canvas_size,
        holdout_chars=holdout, split="val", augment=False,
    )

    # 3. Optional extended simplified-radical dataset (split into train & holdout with zero leakage)
    ex_tr_chars: List[str] = []
    ex_tr_in: Optional[np.ndarray] = None
    ex_tr_tg: Optional[np.ndarray] = None
    ex_tr_w: Optional[np.ndarray] = None
    val_extra_loader: Optional[DataLoader] = None

    if extra_target_otf is not None or extra_npz is not None:
        extra_raw = ExtraRadicalDatasetV46(
            extra_target_otf=extra_target_otf,
            extra_npz=extra_npz,
            noto_sans=extra_noto_sans or noto_sans,
            noto_lang=extra_noto_lang,
            noto_weight=extra_noto_weight if extra_noto_weight is not None else noto_weight,
            cache_dir=cache_dir,
            ids_map=ids_map,
            radical_filter=extra_radicals,
            canvas_size=canvas_size,
        )
        n_ex = len(extra_raw.chars)
        if n_ex > 0:
            # Exclude primary holdout chars from extra train, plus deterministic extra_val_ratio split
            val_idx: List[int] = []
            tr_idx: List[int] = []
            stride = max(2, int(round(1.0 / max(extra_val_ratio, 1e-3)))) if extra_val_ratio > 0 else 0
            for i, ch in enumerate(extra_raw.chars):
                if ch in holdout or (stride > 0 and n_ex >= 20 and (i % stride == 0)):
                    val_idx.append(i)
                else:
                    tr_idx.append(i)

            ex_tr_chars = [extra_raw.chars[i] for i in tr_idx]
            ex_tr_in = extra_raw.inputs_i8[tr_idx]
            ex_tr_tg = extra_raw.targets_i8[tr_idx]
            ex_tr_w = extra_raw.conf_weights[tr_idx]

            if val_idx:
                val_extra_ds = DualGlyphDatasetV46(
                    primary_chars=[],
                    primary_inputs_i8=np.zeros((0, 4, canvas_size, canvas_size), dtype=np.int8),
                    primary_targets_i8=np.zeros((0, 1, canvas_size, canvas_size), dtype=np.int8),
                    extra_chars=[extra_raw.chars[i] for i in val_idx],
                    extra_inputs_i8=extra_raw.inputs_i8[val_idx],
                    extra_targets_i8=extra_raw.targets_i8[val_idx],
                    extra_conf_weights=extra_raw.conf_weights[val_idx],
                    extra_loss_weight=1.0,
                    ids_map=ids_map,
                    use_radical_mask=use_radical_mask,
                    canvas_size=canvas_size,
                    augment=False,
                )
                val_extra_loader = DataLoader(
                    val_extra_ds,
                    batch_size=min(batch_size, max(len(val_extra_ds), 1)),
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                )
            print(
                f"[v46-ExtraDS] Split: extra_train={len(ex_tr_chars)} | extra_val={len(val_idx)}",
                flush=True,
            )

    train_ds = DualGlyphDatasetV46(
        primary_train.chars,
        primary_train.inputs_i8,
        primary_train.targets_i8,
        extra_chars=ex_tr_chars,
        extra_inputs_i8=ex_tr_in,
        extra_targets_i8=ex_tr_tg,
        extra_conf_weights=ex_tr_w,
        extra_loss_weight=extra_loss_weight,
        ids_map=ids_map,
        use_radical_mask=use_radical_mask,
        canvas_size=canvas_size,
        augment=True,
    )
    val_ds = DualGlyphDatasetV46(
        primary_val.chars,
        primary_val.inputs_i8,
        primary_val.targets_i8,
        ids_map=ids_map,
        use_radical_mask=use_radical_mask,
        canvas_size=canvas_size,
        augment=False,
    )

    weights = build_stratified_weights_v46(
        train_ds,
        ids_map,
        radical_boost=radical_boost,
        extra_sample_boost=extra_sample_boost,
        extra_mix_ratio=extra_mix_ratio,
    )
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(batch_size, max(len(val_ds), 1)),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_ds, train_loader, val_ds, val_loader, val_extra_loader, ids_map
