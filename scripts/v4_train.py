#!/usr/bin/env python3
"""
v4.0 Global-Context CNN-Transformer SDF Training.

Two-stage training:
  Stage 1: Freeze CNN, train Transformer (30 epochs, lr=5e-5)
  Stage 2: Unfreeze all, fine-tune end-to-end (15 epochs, lr=2e-5)

Also includes a --test_vram mode that measures peak VRAM usage
with a single forward+backward pass at the target batch size.

Usage:
  # VRAM test (no training):
  python scripts/v4_train.py --noto_sans /path/to/NotoSansCJK-VF.ttc --test_vram

  # Full training:
  python scripts/v4_train.py --noto_sans /path/to/NotoSansCJK-VF.ttc \
      --skip_otf FOT-SkipPro-B.OTF --epochs 45
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v4.dataset_v4 import FullGlyphDatasetV4, create_v4_dataloaders
from src.v4.losses_v4 import GlobalGlyphLossV4
from src.v4.model_v4 import GlobalGlyphTransformer


def test_vram(
    canvas_size: int = 256,
    batch_size: int = 8,
    d_model: int = 768,
    n_transformer_layers: int = 4,
    n_heads: int = 12,
) -> None:
    """Run a single forward+backward pass and report peak VRAM usage."""
    if not torch.cuda.is_available():
        print("[VRAM-Test] CUDA not available, skipping.")
        return

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    model = GlobalGlyphTransformer(
        in_channels=4,
        canvas_size=canvas_size,
        d_model=d_model,
        n_heads=n_heads,
        n_transformer_layers=n_transformer_layers,
    ).to(device)

    n_params = model.count_parameters()
    print(f"\n[VRAM-Test] Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")
    print(f"[VRAM-Test] Config: canvas={canvas_size}, batch={batch_size}, d_model={d_model}, "
          f"layers={n_transformer_layers}, heads={n_heads}")

    # Dummy input
    x = torch.randn(batch_size, 4, canvas_size, canvas_size, device=device)
    y = torch.randn(batch_size, 1, canvas_size, canvas_size, device=device)

    criterion = GlobalGlyphLossV4().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    model.train()
    for _ in range(2):
        pred, flow = model(x)
        loss, metrics = criterion(pred, y, flow)
        loss.backward()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    current_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)

    print(f"[VRAM-Test] Peak VRAM: {peak_mb:.0f} MB ({peak_mb / 1024:.1f} GB)")
    print(f"[VRAM-Test] Current VRAM: {current_mb:.0f} MB")
    print(f"[VRAM-Test] GPU: {torch.cuda.get_device_name(0)}")

    # Try larger batch sizes
    for bs in [12, 16, 24, 32]:
        try:
            torch.cuda.reset_peak_memory_stats(device)
            x_big = torch.randn(bs, 4, canvas_size, canvas_size, device=device)
            y_big = torch.randn(bs, 1, canvas_size, canvas_size, device=device)
            model.zero_grad(set_to_none=True)
            pred_big, flow_big = model(x_big)
            loss_big, _ = criterion(pred_big, y_big, flow_big)
            loss_big.backward()
            torch.cuda.synchronize()
            peak_big = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            print(f"[VRAM-Test] batch={bs:3d}: peak={peak_big:.0f} MB ({peak_big / 1024:.1f} GB)")
            del x_big, y_big, pred_big, flow_big, loss_big
        except RuntimeError as e:
            print(f"[VRAM-Test] batch={bs:3d}: OOM — {str(e)[:80]}")
            break

    print("\n[VRAM-Test] Done.", flush=True)


def train(
    skip_otf: Path,
    noto_sans: Path,
    cache_dir: Path,
    ckpt_out: Path,
    out_json: Path,
    canvas_size: int = 256,
    batch_size: int = 8,
    d_model: int = 768,
    n_transformer_layers: int = 4,
    n_heads: int = 12,
    stage1_epochs: int = 30,
    stage2_epochs: int = 15,
    stage1_lr: float = 5e-5,
    stage2_lr: float = 2e-5,
    warmup_epochs: int = 3,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    num_workers: int = 4,
    save_every: int = 5,
    resume_ckpt: Optional[Path] = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    # ── Model ────────────────────────────────────────────────────────
    decoder_version = "v44"
    resume_state = None
    if resume_ckpt is not None:
        if not resume_ckpt.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt}")
        resume_state = torch.load(resume_ckpt, map_location=device, weights_only=True)
        if "up0.0.weight" in resume_state:
            decoder_version = "v40"
        elif "sdf_head.1.weight" in resume_state:
            decoder_version = "v44"
        else:
            decoder_version = "v41"

    model = GlobalGlyphTransformer(
        in_channels=4,
        canvas_size=canvas_size,
        d_model=d_model,
        n_heads=n_heads,
        n_transformer_layers=n_transformer_layers,
        decoder_version=decoder_version,
    ).to(device)
    if resume_state is not None:
        model.load_state_dict(resume_state)
        print(f"[v4-Train] Resumed weights from {resume_ckpt} (decoder={decoder_version})", flush=True)
    n_params = model.count_parameters()
    print(f"[v4-Train] Model ({decoder_version}): {n_params:,} params ({n_params / 1e6:.1f}M)", flush=True)

    # ── Data ─────────────────────────────────────────────────────────
    holdout = set("志洳菜嘉改栅畴逡红语银饭伞丌习乡卫专书东言直令骨微道")
    dataset, dataloader = create_v4_dataloaders(
        skip_otf=skip_otf,
        noto_sans=noto_sans,
        cache_dir=cache_dir,
        canvas_size=canvas_size,
        batch_size=batch_size,
        num_workers=num_workers,
        holdout_chars=holdout,
    )
    n_samples = len(dataset)
    steps_per_epoch = len(dataloader)
    print(f"[v4-Train] Dataset: {n_samples} chars, {steps_per_epoch} steps/epoch, batch={batch_size}", flush=True)

    # ── Loss ─────────────────────────────────────────────────────────
    criterion = GlobalGlyphLossV4().to(device)

    # ── Stage 1: Freeze CNN, train Transformer ─────────────────────
    epoch_logs: List[dict] = []
    global_step = 0
    t_start = time.time()

    if resume_ckpt is not None:
        print("[v4-Train] Resume mode: skipping Stage 1, proceeding directly to Stage 2 fine-tuning.", flush=True)
        stage1_epochs = 0

    if stage1_epochs > 0:
        print(f"\n[v4-Train] === Stage 1: Freeze CNN, Train Transformer ({stage1_epochs} epochs, lr={stage1_lr}) ===", flush=True)

        # Freeze CNN encoder + decoder (keep Transformer + patch_conv/unpatch_conv trainable)
        frozen_params = []
        trainable_params = []
        for name, param in model.named_parameters():
            is_transformer = "transformer" in name or "patch_conv" in name or "unpatch_conv" in name
            if is_transformer:
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False
                frozen_params.append(param)

        opt1 = torch.optim.AdamW(trainable_params, lr=stage1_lr, weight_decay=weight_decay)

        # Linear warmup + cosine decay
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = stage1_epochs * steps_per_epoch

        def lr_lambda_s1(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))

        sched1 = torch.optim.lr_scheduler.LambdaLR(opt1, lr_lambda_s1)

        for ep in range(1, stage1_epochs + 1):
            model.train()
            t0 = time.time()
            acc: Dict[str, float] = {}

            for batch in dataloader:
                x, y, _ = batch
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                opt1.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    pred, flow = model(x)
                    loss, m = criterion(pred, y, flow)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                opt1.step()
                sched1.step()
                global_step += 1

                for k, v in m.items():
                    acc[k] = acc.get(k, 0.0) + float(v)

            avg_m = {k: round(v / steps_per_epoch, 5) for k, v in acc.items()}
            avg_m["epoch"] = ep
            avg_m["stage"] = 1
            avg_m["lr"] = opt1.param_groups[0]["lr"]
            avg_m["sec"] = round(time.time() - t0, 1)
            epoch_logs.append(avg_m)

            peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if use_amp else 0
            print(
                f"  [S1 Ep {ep:02d}/{stage1_epochs:02d}] ({avg_m['sec']:.1f}s) lr={avg_m['lr']:.2e} | "
                f"sdf={avg_m['sdf_l1']:.4f} grad={avg_m['grad_l1']:.4f} cldice={avg_m['cldice']:.4f} "
                f"iou={avg_m['soft_iou']:.4f} eik={avg_m['eikonal']:.4f} curv={avg_m['curv_l1']:.4f} | VRAM={peak_mb:.0f}MB",
                flush=True,
            )

            if ep % save_every == 0:
                tmp_ckpt = ckpt_out.with_suffix(f".s1_ep{ep}.pt")
                tmp_ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), tmp_ckpt)
                print(f"  [S1] Checkpoint → {tmp_ckpt}", flush=True)

    # ── Stage 2: Unfreeze all, fine-tune end-to-end ────────────────
    print(f"\n[v4-Train] === Stage 2: Unfreeze All, Fine-tune ({stage2_epochs} epochs, lr={stage2_lr}) ===", flush=True)

    for param in model.parameters():
        param.requires_grad = True

    opt2 = torch.optim.AdamW(model.parameters(), lr=stage2_lr, weight_decay=weight_decay)

    warmup_steps2 = max(1, stage2_epochs // 5) * steps_per_epoch
    total_steps2 = stage2_epochs * steps_per_epoch

    def lr_lambda_s2(step: int) -> float:
        if step < warmup_steps2:
            return (step + 1) / warmup_steps2
        progress = (step - warmup_steps2) / max(total_steps2 - warmup_steps2, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lr_lambda_s2)

    for ep in range(1, stage2_epochs + 1):
        model.train()
        t0 = time.time()
        acc: Dict[str, float] = {}

        for batch in dataloader:
            x, y, _ = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt2.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred, flow = model(x)
                loss, m = criterion(pred, y, flow)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt2.step()
            sched2.step()
            global_step += 1

            for k, v in m.items():
                acc[k] = acc.get(k, 0.0) + float(v)

        avg_m = {k: round(v / steps_per_epoch, 5) for k, v in acc.items()}
        avg_m["epoch"] = ep
        avg_m["stage"] = 2
        avg_m["lr"] = opt2.param_groups[0]["lr"]
        avg_m["sec"] = round(time.time() - t0, 1)
        epoch_logs.append(avg_m)

        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if use_amp else 0
        print(
            f"  [S2 Ep {ep:02d}/{stage2_epochs:02d}] ({avg_m['sec']:.1f}s) lr={avg_m['lr']:.2e} | "
            f"sdf={avg_m['sdf_l1']:.4f} grad={avg_m['grad_l1']:.4f} cldice={avg_m['cldice']:.4f} "
            f"iou={avg_m['soft_iou']:.4f} eik={avg_m['eikonal']:.4f} curv={avg_m['curv_l1']:.4f} | VRAM={peak_mb:.0f}MB",
            flush=True,
        )

        if ep % save_every == 0:
            tmp_ckpt = ckpt_out.with_suffix(f".s2_ep{ep}.pt")
            tmp_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), tmp_ckpt)
            print(f"  [S2] Checkpoint → {tmp_ckpt}", flush=True)

    # ── Save final checkpoint ────────────────────────────────────────
    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_out)
    print(f"\n[v4-Train] Final checkpoint → {ckpt_out}", flush=True)

    # ── Save training log ────────────────────────────────────────────
    total_time = time.time() - t_start
    report = {
        "model": "GlobalGlyphTransformer",
        "params": n_params,
        "config": {
            "canvas_size": canvas_size,
            "batch_size": batch_size,
            "d_model": d_model,
            "n_transformer_layers": n_transformer_layers,
            "n_heads": n_heads,
            "stage1_epochs": stage1_epochs,
            "stage2_epochs": stage2_epochs,
            "stage1_lr": stage1_lr,
            "stage2_lr": stage2_lr,
        },
        "total_time_sec": round(total_time, 1),
        "epoch_logs": epoch_logs,
        "checkpoint": str(ckpt_out),
        "resumed_from": str(resume_ckpt) if resume_ckpt is not None else None,
        "decoder_version": decoder_version,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v4-Train] Training log → {out_json}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="v4.0 Global-Context CNN-Transformer SDF Training")
    parser.add_argument("--skip_otf", type=Path, default=ROOT / "FOT-SkipPro-B.OTF")
    parser.add_argument("--noto_sans", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--ckpt_out", type=Path, default=ROOT / "checkpoints" / "v4_zh_sdf_global.pt")
    parser.add_argument("--out_json", type=Path, default=ROOT / "outputs" / "v4_train_log.json")

    # Model config
    parser.add_argument("--canvas_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_transformer_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=12)

    # Training config
    parser.add_argument("--stage1_epochs", type=int, default=30)
    parser.add_argument("--stage2_epochs", type=int, default=15)
    parser.add_argument("--stage1_lr", type=float, default=5e-5)
    parser.add_argument("--stage2_lr", type=float, default=2e-5)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--resume", type=Path, default=None, help="Resume fine-tuning from an existing checkpoint")

    # VRAM test mode
    parser.add_argument("--test_vram", action="store_true", help="Run VRAM test and exit")

    args = parser.parse_args()

    if args.test_vram:
        test_vram(
            canvas_size=args.canvas_size,
            batch_size=args.batch_size,
            d_model=args.d_model,
            n_transformer_layers=args.n_transformer_layers,
            n_heads=args.n_heads,
        )
        return

    train(
        skip_otf=args.skip_otf,
        noto_sans=args.noto_sans,
        cache_dir=args.cache_dir,
        ckpt_out=args.ckpt_out,
        out_json=args.out_json,
        canvas_size=args.canvas_size,
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_transformer_layers=args.n_transformer_layers,
        n_heads=args.n_heads,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage1_lr=args.stage1_lr,
        stage2_lr=args.stage2_lr,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        save_every=args.save_every,
        resume_ckpt=args.resume,
    )


if __name__ == "__main__":
    main()
