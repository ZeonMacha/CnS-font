#!/usr/bin/env python3
"""
v4.5 Global-Context Direct-SDF Training (Clean Standalone).

Fixes over v4_train.py (from 审计v4.md):
  1. Stage 1 freezes ONLY the CNN encoder (stem, down1..down3), keeping Transformer +
     patch/unpatch + CNN decoder + flow_head + sdf_head ALL trainable (eliminates Stage 1 zero-gradient deadlock!).
  2. Per-epoch holdout validation logging (`val_sdf_l1`, `val_iou`) on `UNIFIED_HOLDOUT_28`.
  3. 3x WeightedRandomSampler oversampling on rare bowed `子` side-radicals & sparse dot/sweep glyphs.
  4. Portable CLI (`--noto_sans` / `CNS_NOTO_SANS`, `--noto_weight`, `--resume`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v45 import UNIFIED_HOLDOUT_28
from src.v45.dataset_v45 import create_v45_dataloaders
from src.v45.losses_v45 import GlobalGlyphLossV45
from src.v45.model_v45 import GlobalGlyphTransformerV45


@torch.no_grad()
def evaluate_holdout(
    model: GlobalGlyphTransformerV45,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, float]:
    """Compute exact holdout SDF L1 & binary IoU on UNIFIED_HOLDOUT_28 (chars present in SkipPro)."""
    model.eval()
    l1_sum, iou_sum, n_tot = 0.0, 0.0, 0
    for x, y, _ in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred, _ = model(x)
        p = pred.float()
        t = y.float()
        b = p.size(0)
        l1_sum += float((p - t).abs().mean(dim=(1, 2, 3)).sum())
        pm = p > 0.0
        tm = t > 0.0
        inter = (pm & tm).float().sum(dim=(1, 2, 3))
        union = (pm | tm).float().sum(dim=(1, 2, 3)).clamp_min(1.0)
        iou_sum += float((inter / union).sum())
        n_tot += b
    model.train()
    if n_tot == 0:
        return {"val_sdf_l1": 0.0, "val_iou": 0.0, "val_n": 0}
    return {
        "val_sdf_l1": round(l1_sum / n_tot, 5),
        "val_iou": round(iou_sum / n_tot, 5),
        "val_n": n_tot,
    }


def test_vram(canvas_size: int = 256, batch_size: int = 16) -> None:
    if not torch.cuda.is_available():
        print("[v45-VRAM] CUDA not available.")
        return
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = GlobalGlyphTransformerV45(canvas_size=canvas_size).to(device)
    criterion = GlobalGlyphLossV45().to(device)
    print(f"[v45-VRAM] Model: {model.count_parameters():,} params ({model.count_parameters()/1e6:.2f}M) | GPU: {torch.cuda.get_device_name(0)}")

    x = torch.randn(batch_size, 4, canvas_size, canvas_size, device=device)
    y = torch.randn(batch_size, 1, canvas_size, canvas_size, device=device)
    # Verify Stage 1 gradient flow (encoder frozen, transformer+decoder+heads trainable)
    for name, param in model.named_parameters():
        enc = name.startswith(("stem.", "down1.", "down2.", "down3."))
        param.requires_grad = not enc
    model.train()
    pred, flow = model(x)
    loss, m = criterion(pred, y, flow)
    loss.backward()
    t_grad_norm = sum(
        float(p.grad.detach().norm()) for n, p in model.named_parameters()
        if "transformer" in n and p.grad is not None
    )
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    print(f"[v45-VRAM] batch={batch_size} peak={peak_mb:.0f} MB | Stage1 transformer grad_norm={t_grad_norm:.4f} (must be > 0!)")
    assert t_grad_norm > 0.0, "Stage 1 zero-gradient deadlock detected!"


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    model = GlobalGlyphTransformerV45(
        in_channels=4,
        canvas_size=args.canvas_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_transformer_layers=args.n_transformer_layers,
        max_shift_px=args.max_shift_px,
    ).to(device)

    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[v45-Train] Resumed from {args.resume}", flush=True)

    n_params = model.count_parameters()
    print(f"[v45-Train] GlobalGlyphTransformerV45: {n_params:,} params ({n_params/1e6:.2f}M) | max_shift={args.max_shift_px}px", flush=True)

    train_ds, train_loader, val_ds, val_loader = create_v45_dataloaders(
        skip_otf=args.skip_otf,
        noto_sans=args.noto_sans,
        cache_dir=args.cache_dir,
        noto_weight=args.noto_weight,
        ids_path=args.ids_path,
        canvas_size=args.canvas_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        holdout_chars=UNIFIED_HOLDOUT_28,
    )
    steps_per_epoch = len(train_loader)
    print(f"[v45-Train] Train={len(train_ds)} ({steps_per_epoch} steps/ep, batch={args.batch_size}) | Val={len(val_ds)}", flush=True)

    criterion = GlobalGlyphLossV45().to(device)
    epoch_logs: List[dict] = []
    t_start = time.time()

    stage1_epochs = 0 if args.resume is not None else args.stage1_epochs

    # ── Stage 1: Freeze CNN Encoder ONLY (Transformer + Decoder + Heads ALL train!) ──
    if stage1_epochs > 0:
        print(f"\n[v45-Train] === Stage 1: Freeze CNN Encoder ({stage1_epochs} ep, lr={args.stage1_lr}) ===", flush=True)
        trainable_s1 = []
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
            model.train()
            t0 = time.time()
            acc: Dict[str, float] = {}
            for x, y, _ in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt1.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    pred, flow = model(x)
                    loss, m = criterion(pred, y, flow)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_s1, args.grad_clip)
                opt1.step()
                sched1.step()
                for k, v in m.items():
                    acc[k] = acc.get(k, 0.0) + float(v)

            avg_m = {k: round(v / steps_per_epoch, 5) for k, v in acc.items()}
            val_m = evaluate_holdout(model, val_loader, device, use_amp, amp_dtype)
            avg_m.update(val_m)
            avg_m.update({"epoch": ep, "stage": 1, "lr": opt1.param_groups[0]["lr"], "sec": round(time.time() - t0, 1)})
            epoch_logs.append(avg_m)
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if use_amp else 0
            print(
                f"  [S1 Ep {ep:02d}/{stage1_epochs:02d}] ({avg_m['sec']:.1f}s) lr={avg_m['lr']:.2e} | "
                f"sdf={avg_m['sdf_l1']:.4f} cldice={avg_m['cldice']:.4f} iou={avg_m['soft_iou']:.4f} "
                f"eik={avg_m['eikonal']:.4f} curv={avg_m['curv_l1']:.4f} ftv={avg_m['flow_tv']:.4f} | "
                f"val_l1={avg_m['val_sdf_l1']:.4f} val_iou={avg_m['val_iou']:.4f} | VRAM={peak_mb:.0f}MB",
                flush=True,
            )

    # ── Stage 2: Unfreeze All, Full End-to-End Training ──
    print(f"\n[v45-Train] === Stage 2: Unfreeze All ({args.stage2_epochs} ep, lr={args.stage2_lr}) ===", flush=True)
    for param in model.parameters():
        param.requires_grad = True

    opt2 = torch.optim.AdamW(model.parameters(), lr=args.stage2_lr, weight_decay=args.weight_decay)
    warmup_s2 = max(1, args.stage2_epochs // 7) * steps_per_epoch
    total_s2 = args.stage2_epochs * steps_per_epoch

    def lr_s2(step: int) -> float:
        if step < warmup_s2:
            return (step + 1) / warmup_s2
        prog = (step - warmup_s2) / max(total_s2 - warmup_s2, 1)
        return 0.5 * (1.0 + np.cos(np.pi * prog))

    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lr_s2)

    for ep in range(1, args.stage2_epochs + 1):
        model.train()
        t0 = time.time()
        acc: Dict[str, float] = {}
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt2.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred, flow = model(x)
                loss, m = criterion(pred, y, flow)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt2.step()
            sched2.step()
            for k, v in m.items():
                acc[k] = acc.get(k, 0.0) + float(v)

        avg_m = {k: round(v / steps_per_epoch, 5) for k, v in acc.items()}
        val_m = evaluate_holdout(model, val_loader, device, use_amp, amp_dtype)
        avg_m.update(val_m)
        avg_m.update({"epoch": ep, "stage": 2, "lr": opt2.param_groups[0]["lr"], "sec": round(time.time() - t0, 1)})
        epoch_logs.append(avg_m)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if use_amp else 0
        print(
            f"  [S2 Ep {ep:02d}/{args.stage2_epochs:02d}] ({avg_m['sec']:.1f}s) lr={avg_m['lr']:.2e} | "
            f"sdf={avg_m['sdf_l1']:.4f} cldice={avg_m['cldice']:.4f} iou={avg_m['soft_iou']:.4f} "
            f"eik={avg_m['eikonal']:.4f} curv={avg_m['curv_l1']:.4f} ftv={avg_m['flow_tv']:.4f} | "
            f"val_l1={avg_m['val_sdf_l1']:.4f} val_iou={avg_m['val_iou']:.4f} | VRAM={peak_mb:.0f}MB",
            flush=True,
        )

        if ep % args.save_every == 0:
            tmp_ckpt = args.ckpt_out.with_suffix(f".s2_ep{ep}.pt")
            tmp_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), tmp_ckpt)

    args.ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.ckpt_out)
    print(f"\n[v45-Train] Final checkpoint -> {args.ckpt_out}", flush=True)

    report = {
        "model": "GlobalGlyphTransformerV45",
        "params": n_params,
        "noto_sans": args.noto_sans.name,
        "noto_weight": args.noto_weight,
        "max_shift_px": args.max_shift_px,
        "resumed_from": str(args.resume) if args.resume else None,
        "total_time_sec": round(time.time() - t_start, 1),
        "epoch_logs": epoch_logs,
        "checkpoint": str(args.ckpt_out),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v45-Train] Log -> {args.out_json}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="v4.5 Global-Context Direct-SDF Training")
    parser.add_argument("--skip_otf", type=Path, default=ROOT / "FOT-SkipPro-B.OTF")
    parser.add_argument("--noto_sans", type=Path, default=Path(os.environ.get("CNS_NOTO_SANS", "")))
    parser.add_argument("--noto_weight", type=int, default=None, help="VF wght axis (None for static TTC, 300/400 for VF)")
    parser.add_argument("--ids_path", type=Path, default=ROOT / "data_v2" / "ids.txt")
    parser.add_argument("--cache_dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--ckpt_out", type=Path, default=ROOT / "checkpoints" / "v45_zh_sdf_global.pt")
    parser.add_argument("--out_json", type=Path, default=ROOT / "outputs" / "v45_train_log.json")

    parser.add_argument("--canvas_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_transformer_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--max_shift_px", type=float, default=14.0)

    parser.add_argument("--stage1_epochs", type=int, default=5)
    parser.add_argument("--stage2_epochs", type=int, default=35)
    parser.add_argument("--stage1_lr", type=float, default=1.5e-4)
    parser.add_argument("--stage2_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--test_vram", action="store_true")

    args = parser.parse_args()
    if args.test_vram:
        test_vram(canvas_size=args.canvas_size, batch_size=args.batch_size)
        return
    if not str(args.noto_sans) or not args.noto_sans.exists():
        raise SystemExit(f"Provide valid --noto_sans path (or set CNS_NOTO_SANS). Got: {args.noto_sans}")
    train(args)


if __name__ == "__main__":
    main()
