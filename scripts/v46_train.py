#!/usr/bin/env python3
"""
v4.6 Dual-Dataset Direct-SDF + Uncertainty Training Script (`scripts/v46_train.py`).

Supports:
  1. Primary dataset (SkipPro + Noto JP) + optional extended simplified-radical dataset
     via `--extra_target_otf` (OTF + Noto SC) or `--extra_npz` (self-distilled / pseudo-label NPZ).
  2. `GlobalGlyphTransformerV46` (32x32 local window attention, heteroscedastic uncertainty head,
     optional 5th radical-mask channel, MC-dropout for sampling diversity).
  3. Seamless fine-tuning from a v4.5 checkpoint (`--resume_v45`) or resuming v4.6 (`--resume`).
  4. Expanded stratified validation (>= 200 chars, grouped by IDS radical family) + separate
     simplified-radical validation metrics (`val_extra_*`) when the extra dataset is enabled.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v46.dataset_v46 import classify_ids_radical_group, create_v46_dataloaders
from src.v46.losses_v46 import GlobalGlyphLossV46, compute_sdf_fog_ratio
from src.v46.model_v46 import GlobalGlyphTransformerV46, load_v45_weights_into_v46


@torch.no_grad()
def evaluate_holdout_v46(
    model: GlobalGlyphTransformerV46,
    val_loader: torch.utils.data.DataLoader,
    ids_map: Dict[str, str],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    trunc_px: float = 16.0,
) -> Dict[str, object]:
    """
    Evaluate SDF L1, binary IoU, zero-level-set fog ratio, and mean predicted uncertainty
    both overall and broken down by IDS radical family.
    """
    model.eval()
    l1_sum, iou_sum, fog_sum, unc_sum, n_tot = 0.0, 0.0, 0.0, 0.0, 0
    grp_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"l1": 0.0, "iou": 0.0, "fog": 0.0, "n": 0.0})

    for batch in val_loader:
        x, y, chars, _, _ = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred, _, logvar = model(x, return_uncertainty=True)
        p = pred.float()
        t = y.float()
        lv = logvar.float()

        l1_per = (p - t).abs().mean(dim=(1, 2, 3))
        pm = p > 0.0
        tm = t > 0.0
        inter = (pm & tm).float().sum(dim=(1, 2, 3))
        union = (pm | tm).float().sum(dim=(1, 2, 3)).clamp_min(1.0)
        iou_per = inter / union
        fog_per = compute_sdf_fog_ratio(p, trunc_px=trunc_px)
        near_zero = (t.abs() < 0.25).float()
        unc_per = (near_zero * torch.exp(lv)).sum(dim=(1, 2, 3)) / near_zero.sum(dim=(1, 2, 3)).clamp_min(1.0)

        for i, ch in enumerate(chars):
            l1_v = float(l1_per[i])
            iou_v = float(iou_per[i])
            fog_v = float(fog_per[i])
            unc_v = float(unc_per[i])
            l1_sum += l1_v
            iou_sum += iou_v
            fog_sum += fog_v
            unc_sum += unc_v
            n_tot += 1

            g = classify_ids_radical_group(ch, ids_map)
            grp_stats[g]["l1"] += l1_v
            grp_stats[g]["iou"] += iou_v
            grp_stats[g]["fog"] += fog_v
            grp_stats[g]["n"] += 1.0

    model.train()
    if n_tot == 0:
        return {"val_sdf_l1": 0.0, "val_iou": 0.0, "val_fog": 0.0, "val_unc": 0.0, "val_n": 0, "val_by_group": {}}

    by_group = {
        g: {
            "n": int(s["n"]),
            "l1": round(s["l1"] / s["n"], 5),
            "iou": round(s["iou"] / s["n"], 5),
            "fog": round(s["fog"] / s["n"], 4),
        }
        for g, s in sorted(grp_stats.items())
        if s["n"] > 0
    }
    return {
        "val_sdf_l1": round(l1_sum / n_tot, 5),
        "val_iou": round(iou_sum / n_tot, 5),
        "val_fog": round(fog_sum / n_tot, 4),
        "val_unc": round(unc_sum / n_tot, 4),
        "val_n": n_tot,
        "val_by_group": by_group,
    }


def test_vram(
    canvas_size: int = 256,
    batch_size: int = 16,
    use_radical_mask: bool = False,
    use_local_attn_32: bool = True,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    in_ch = 5 if use_radical_mask else 4
    model = GlobalGlyphTransformerV46(
        in_channels=in_ch,
        use_radical_mask=use_radical_mask,
        canvas_size=canvas_size,
        use_local_attn_32=use_local_attn_32,
    ).to(device)
    criterion = GlobalGlyphLossV46().to(device)
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(
        f"[v46-VRAM] Model: {model.count_parameters():,} params ({model.count_parameters()/1e6:.2f}M) | "
        f"in_ch={in_ch} local_attn_32={use_local_attn_32} | Device: {gpu_name}"
    )

    x = torch.randn(batch_size, in_ch, canvas_size, canvas_size, device=device)
    y = torch.randn(batch_size, 1, canvas_size, canvas_size, device=device).clamp(-1.0, 1.0)
    sw = torch.ones(batch_size, device=device)

    # Verify Stage 1 gradient flow (encoder frozen, transformer + local_attn_32 + decoder + heads trainable)
    for name, param in model.named_parameters():
        enc = name.startswith(("stem.", "down1.", "down2.", "down3."))
        param.requires_grad = not enc
    model.train()
    pred, flow, logvar = model(x, return_uncertainty=True)
    loss, m = criterion(pred, y, flow, pred_logvar=logvar, sample_weight=sw)
    loss.backward()
    t_grad_norm = sum(
        float(p.grad.detach().norm()) for n, p in model.named_parameters()
        if ("transformer" in n or "local_attn_32" in n) and p.grad is not None
    )
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    print(
        f"[v46-VRAM] batch={batch_size} peak={peak_mb:.0f} MB | "
        f"Stage1 attn grad_norm={t_grad_norm:.4f} | fog_loss={m['fog_loss']:.4f} | uncert_nll={m['uncert_nll']:.4f}"
    )
    assert t_grad_norm > 0.0, "Stage 1 zero-gradient deadlock detected!"


def _run_epoch(
    model: GlobalGlyphTransformerV46,
    train_loader: torch.utils.data.DataLoader,
    criterion: GlobalGlyphLossV46,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    trainable_params: List[torch.nn.Parameter],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    acc: Dict[str, float] = {}
    steps = len(train_loader)
    n_extra_draws = 0
    n_total_draws = 0
    for x, y, _, sw, src_id in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        sw_t = torch.as_tensor(sw, dtype=torch.float32, device=device)
        n_extra_draws += int((torch.as_tensor(src_id) == 1).sum())
        n_total_draws += x.size(0)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred, flow, logvar = model(x, return_uncertainty=True)
            loss, m = criterion(pred, y, flow, pred_logvar=logvar, sample_weight=sw_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        optimizer.step()
        scheduler.step()
        for k, v in m.items():
            acc[k] = acc.get(k, 0.0) + float(v)

    avg = {k: round(v / max(steps, 1), 5) for k, v in acc.items()}
    avg["extra_draw_share"] = round(n_extra_draws / max(n_total_draws, 1), 4)
    return avg


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    in_ch = 5 if args.use_radical_mask else 4
    model = GlobalGlyphTransformerV46(
        in_channels=in_ch,
        use_radical_mask=args.use_radical_mask,
        canvas_size=args.canvas_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_transformer_layers=args.n_transformer_layers,
        dropout=args.dropout,
        mc_dropout=args.mc_dropout,
        max_shift_px=args.max_shift_px,
        use_local_attn_32=not args.no_local_attn_32,
    ).to(device)

    resumed_from: Optional[str] = None
    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state)
        resumed_from = str(args.resume)
        print(f"[v46-Train] Resumed v4.6 checkpoint: {args.resume}", flush=True)
    elif args.resume_v45 is not None:
        if not args.resume_v45.exists():
            raise FileNotFoundError(f"v4.5 checkpoint not found: {args.resume_v45}")
        info = load_v45_weights_into_v46(model, args.resume_v45, map_location=device)
        resumed_from = str(args.resume_v45)
        print(f"[v46-Train] Initialized from v4.5 checkpoint {args.resume_v45}: {info}", flush=True)

    n_params = model.count_parameters()
    print(
        f"[v46-Train] GlobalGlyphTransformerV46: {n_params:,} params ({n_params/1e6:.2f}M) | "
        f"in_ch={in_ch} local_attn_32={not args.no_local_attn_32} mc_drop={args.mc_dropout}",
        flush=True,
    )

    extra_radicals: Optional[Set[str]] = None
    if args.extra_radicals:
        extra_radicals = {r.strip() for r in args.extra_radicals.split(",") if r.strip()}

    train_ds, train_loader, val_ds, val_loader, val_extra_loader, ids_map = create_v46_dataloaders(
        skip_otf=args.skip_otf,
        noto_sans=args.noto_sans,
        cache_dir=args.cache_dir,
        noto_weight=args.noto_weight,
        ids_path=args.ids_path,
        canvas_size=args.canvas_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_target_size=args.val_target_size,
        use_radical_mask=args.use_radical_mask,
        radical_boost=args.radical_boost,
        extra_target_otf=args.extra_target_otf,
        extra_npz=args.extra_npz,
        extra_noto_sans=args.extra_noto_sans,
        extra_noto_lang=args.extra_noto_lang,
        extra_noto_weight=args.extra_noto_weight,
        extra_radicals=extra_radicals,
        extra_loss_weight=args.extra_loss_weight,
        extra_sample_boost=args.extra_sample_boost,
        extra_mix_ratio=args.extra_mix_ratio,
        extra_val_ratio=args.extra_val_ratio,
    )
    steps_per_epoch = len(train_loader)
    print(
        f"[v46-Train] Train={len(train_ds)} (primary={train_ds.n_primary}, extra={train_ds.n_extra}, "
        f"{steps_per_epoch} steps/ep, batch={args.batch_size}) | Val={len(val_ds)}"
        + (f" | ValExtra={len(val_extra_loader.dataset)}" if val_extra_loader is not None else ""),
        flush=True,
    )

    criterion = GlobalGlyphLossV46(
        fog_weight=args.fog_weight,
        uncert_weight=args.uncert_weight,
    ).to(device)

    epoch_logs: List[dict] = []
    t_start = time.time()
    stage1_epochs = 0 if (args.resume is not None or args.resume_v45 is not None) else args.stage1_epochs

    # ── Stage 1: Freeze CNN Encoder ONLY (fresh training) ──
    if stage1_epochs > 0:
        print(f"\n[v46-Train] === Stage 1: Freeze CNN Encoder ({stage1_epochs} ep, lr={args.stage1_lr}) ===", flush=True)
        trainable_s1: List[torch.nn.Parameter] = []
        for name, param in model.named_parameters():
            is_enc = name.startswith(("stem.", "down1.", "down2.", "down3."))
            param.requires_grad = not is_enc
            if param.requires_grad:
                trainable_s1.append(param)

        opt1 = torch.optim.AdamW(trainable_s1, lr=args.stage1_lr, weight_decay=args.weight_decay)
        warmup_s1 = max(1, min(args.warmup_epochs, stage1_epochs)) * steps_per_epoch
        total_s1 = stage1_epochs * steps_per_epoch

        def lr_s1(step: int) -> float:
            if step < warmup_s1:
                return (step + 1) / warmup_s1
            prog = (step - warmup_s1) / max(total_s1 - warmup_s1, 1)
            return 0.5 * (1.0 + np.cos(np.pi * prog))

        sched1 = torch.optim.lr_scheduler.LambdaLR(opt1, lr_s1)

        for ep in range(1, stage1_epochs + 1):
            t0 = time.time()
            avg_m = _run_epoch(
                model, train_loader, criterion, opt1, sched1, trainable_s1,
                device, use_amp, amp_dtype, args.grad_clip,
            )
            val_m = evaluate_holdout_v46(model, val_loader, ids_map, device, use_amp, amp_dtype)
            avg_m.update(val_m)
            if val_extra_loader is not None:
                vex = evaluate_holdout_v46(model, val_extra_loader, ids_map, device, use_amp, amp_dtype)
                avg_m["val_extra_l1"] = vex["val_sdf_l1"]
                avg_m["val_extra_iou"] = vex["val_iou"]
                avg_m["val_extra_fog"] = vex["val_fog"]
            avg_m.update({"epoch": ep, "stage": 1, "lr": opt1.param_groups[0]["lr"], "sec": round(time.time() - t0, 1)})
            epoch_logs.append(avg_m)
            extra_str = (
                f" | ex_l1={avg_m['val_extra_l1']:.4f} ex_iou={avg_m['val_extra_iou']:.4f}"
                if "val_extra_l1" in avg_m else ""
            )
            print(
                f"  [S1 Ep {ep:02d}/{stage1_epochs:02d}] ({avg_m['sec']:.1f}s) lr={avg_m['lr']:.2e} | "
                f"sdf={avg_m['sdf_l1']:.4f} fog={avg_m['fog_loss']:.4f} unc={avg_m['uncert_nll']:.4f} | "
                f"val_l1={avg_m['val_sdf_l1']:.4f} val_iou={avg_m['val_iou']:.4f} val_fog={avg_m['val_fog']:.3f}{extra_str}",
                flush=True,
            )

    # ── Stage 2: Unfreeze All (or Fine-Tune from v4.5) ──
    print(f"\n[v46-Train] === Stage 2: Full End-to-End ({args.stage2_epochs} ep, lr={args.stage2_lr}) ===", flush=True)
    for param in model.parameters():
        param.requires_grad = True
    all_params = list(model.parameters())

    opt2 = torch.optim.AdamW(all_params, lr=args.stage2_lr, weight_decay=args.weight_decay)
    warmup_s2 = max(1, args.stage2_epochs // 7) * steps_per_epoch
    total_s2 = max(1, args.stage2_epochs * steps_per_epoch)

    def lr_s2(step: int) -> float:
        if step < warmup_s2:
            return (step + 1) / warmup_s2
        prog = (step - warmup_s2) / max(total_s2 - warmup_s2, 1)
        return 0.5 * (1.0 + np.cos(np.pi * prog))

    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lr_s2)

    for ep in range(1, args.stage2_epochs + 1):
        t0 = time.time()
        avg_m = _run_epoch(
            model, train_loader, criterion, opt2, sched2, all_params,
            device, use_amp, amp_dtype, args.grad_clip,
        )
        val_m = evaluate_holdout_v46(model, val_loader, ids_map, device, use_amp, amp_dtype)
        avg_m.update(val_m)
        if val_extra_loader is not None:
            vex = evaluate_holdout_v46(model, val_extra_loader, ids_map, device, use_amp, amp_dtype)
            avg_m["val_extra_l1"] = vex["val_sdf_l1"]
            avg_m["val_extra_iou"] = vex["val_iou"]
            avg_m["val_extra_fog"] = vex["val_fog"]
        avg_m.update({"epoch": ep, "stage": 2, "lr": opt2.param_groups[0]["lr"], "sec": round(time.time() - t0, 1)})
        epoch_logs.append(avg_m)
        extra_str = (
            f" | ex_l1={avg_m['val_extra_l1']:.4f} ex_iou={avg_m['val_extra_iou']:.4f}"
            if "val_extra_l1" in avg_m else ""
        )
        print(
            f"  [S2 Ep {ep:02d}/{args.stage2_epochs:02d}] ({avg_m['sec']:.1f}s) lr={avg_m['lr']:.2e} | "
            f"sdf={avg_m['sdf_l1']:.4f} fog={avg_m['fog_loss']:.4f} unc={avg_m['uncert_nll']:.4f} | "
            f"val_l1={avg_m['val_sdf_l1']:.4f} val_iou={avg_m['val_iou']:.4f} val_fog={avg_m['val_fog']:.3f}{extra_str}",
            flush=True,
        )

        if ep % args.save_every == 0:
            tmp_ckpt = args.ckpt_out.with_suffix(f".s2_ep{ep}.pt")
            tmp_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), tmp_ckpt)

    args.ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.ckpt_out)
    print(f"\n[v46-Train] Final checkpoint -> {args.ckpt_out}", flush=True)

    report = {
        "model": "GlobalGlyphTransformerV46",
        "params": n_params,
        "in_channels": in_ch,
        "use_radical_mask": args.use_radical_mask,
        "use_local_attn_32": not args.no_local_attn_32,
        "mc_dropout": args.mc_dropout,
        "noto_sans": args.noto_sans.name,
        "noto_weight": args.noto_weight,
        "extra_target_otf": str(args.extra_target_otf) if args.extra_target_otf else None,
        "extra_npz": str(args.extra_npz) if args.extra_npz else None,
        "extra_loss_weight": args.extra_loss_weight,
        "extra_sample_boost": args.extra_sample_boost,
        "extra_mix_ratio": args.extra_mix_ratio,
        "n_primary_train": train_ds.n_primary,
        "n_extra_train": train_ds.n_extra,
        "n_val": len(val_ds),
        "resumed_from": resumed_from,
        "total_time_sec": round(time.time() - t_start, 1),
        "epoch_logs": epoch_logs,
        "checkpoint": str(args.ckpt_out),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v46-Train] Log -> {args.out_json}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="v4.6 Dual-Dataset Direct-SDF + Uncertainty Training")
    # Primary dataset
    parser.add_argument("--skip_otf", type=Path, default=ROOT / "FOT-SkipPro-B.OTF")
    parser.add_argument("--noto_sans", type=Path, default=Path(os.environ.get("CNS_NOTO_SANS", "")))
    parser.add_argument("--noto_weight", type=int, default=None, help="Primary VF wght axis (None or 300/400)")
    parser.add_argument("--ids_path", type=Path, default=ROOT / "data_v2" / "ids.txt")
    parser.add_argument("--cache_dir", type=Path, default=ROOT / "datasets")

    # Optional extended simplified-radical dataset
    parser.add_argument("--extra_target_otf", type=Path, default=None, help="Optional simplified-radical target OTF")
    parser.add_argument("--extra_npz", type=Path, default=None, help="Optional simplified-radical precomputed NPZ")
    parser.add_argument("--extra_noto_sans", type=Path, default=None, help="Optional Noto font for extra OTF (defaults to --noto_sans)")
    parser.add_argument("--extra_noto_lang", type=str, default="zh-Hans", help="Language tag for extra OTF ref (default: zh-Hans)")
    parser.add_argument("--extra_noto_weight", type=int, default=None, help="Optional VF weight for extra Noto ref")
    parser.add_argument("--extra_radicals", type=str, default="", help="Comma-separated radical filter for extra dataset (e.g. '纟,讠,钅,饣')")
    parser.add_argument("--extra_loss_weight", type=float, default=0.85, help="Base loss multiplier for extra pseudo-labeled samples")
    parser.add_argument("--extra_sample_boost", type=float, default=3.5, help="Sampling boost for extra dataset when extra_mix_ratio is unset")
    parser.add_argument("--extra_mix_ratio", type=float, default=None, help="Optional target fraction of extra samples per epoch (e.g. 0.20)")
    parser.add_argument("--extra_val_ratio", type=float, default=0.10, help="Holdout fraction carved from extra dataset for val_extra logging")

    # Model & loss
    parser.add_argument("--canvas_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_transformer_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mc_dropout", type=float, default=0.08)
    parser.add_argument("--max_shift_px", type=float, default=14.0)
    parser.add_argument("--use_radical_mask", action="store_true", help="Enable 5th spatial radical-mask input channel")
    parser.add_argument("--no_local_attn_32", action="store_true", help="Disable 32x32 local window attention block")
    parser.add_argument("--fog_weight", type=float, default=0.45)
    parser.add_argument("--uncert_weight", type=float, default=0.08)

    # Training schedule & outputs
    parser.add_argument("--val_target_size", type=int, default=200)
    parser.add_argument("--radical_boost", type=float, default=3.0)
    parser.add_argument("--stage1_epochs", type=int, default=5)
    parser.add_argument("--stage2_epochs", type=int, default=15)
    parser.add_argument("--stage1_lr", type=float, default=1.5e-4)
    parser.add_argument("--stage2_lr", type=float, default=5e-5)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--resume", type=Path, default=None, help="Resume v4.6 checkpoint")
    parser.add_argument("--resume_v45", type=Path, default=None, help="Fine-tune from v4.5 checkpoint")
    parser.add_argument("--ckpt_out", type=Path, default=ROOT / "checkpoints" / "v46_zh_sdf_global.pt")
    parser.add_argument("--out_json", type=Path, default=ROOT / "outputs" / "v46_train_log.json")
    parser.add_argument("--test_vram", action="store_true")

    args = parser.parse_args()
    if args.test_vram:
        test_vram(
            canvas_size=args.canvas_size,
            batch_size=args.batch_size,
            use_radical_mask=args.use_radical_mask,
            use_local_attn_32=not args.no_local_attn_32,
        )
        return
    if not str(args.noto_sans) or not args.noto_sans.exists():
        raise SystemExit(f"Provide valid --noto_sans path (or set CNS_NOTO_SANS). Got: {args.noto_sans}")
    train(args)


if __name__ == "__main__":
    main()
