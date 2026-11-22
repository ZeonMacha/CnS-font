#!/usr/bin/env python3
"""
Deploy CnS v4.0 to the remote 4090 (llm24g) via SCP + SSH.

Usage:
  python scripts/v4_deploy.py              # Sync + VRAM test
  python scripts/v4_deploy.py --sync-only  # Sync files only
  python scripts/v4_deploy.py --dry-run    # Preview without executing
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HOST = "llm24g"
CONDA_ENV = "comfyui"
REMOTE_ROOT = "~/cns_font"

# All files needed for v4 training
FILES_TO_SYNC = [
    "src/v4/__init__.py",
    "src/v4/model_v4.py",
    "src/v4/dataset_v4.py",
    "src/v4/losses_v4.py",
    "scripts/v4_train.py",
    "scripts/v4_vram_test.py",
    "src/v3/__init__.py",
    "src/v3/hb_light.py",
    "src/v3/structural_gate.py",
    "src/v3/local_inpainter.py",
    "src/v3/curve_constrain.py",
    "src/v3/otf_packager.py",
    "src/v3/variant_classifier.py",
    "src/v2/__init__.py",
    "src/v2/hb_outlines.py",
    "src/v2/component_bank.py",
    "src/v2/curve_optimizer.py",
    "src/v2/stroke_assembler.py",
    "src/__init__.py",
    "src/model.py",
    "src/sdf_utils.py",
    "src/glyph_renderer.py",
    "src/vectorizer.py",
    "src/losses.py",
    "src/dataset.py",
    "data_v3/variant_policy.tsv",
    "FOT-SkipPro-B.OTF",
]


def ssh_run(command: str, conda_env: str = None, dry_run: bool = False, timeout: int = 120) -> str:
    """Execute a short command on remote host via SSH."""
    script_lines = []
    if conda_env:
        script_lines.append("source ~/miniconda3/etc/profile.d/conda.sh && conda activate " + conda_env)
    script_lines.append(command)
    remote_script = "\n".join(script_lines)

    if dry_run:
        print("  [DRY] " + command[:120])
        return ""

    # Use ssh with -c to pass command directly (avoids base64 length issues for short commands)
    ssh_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", HOST, remote_script]
    try:
        result = subprocess.run(
            ssh_cmd, check=True, text=True, encoding="utf-8", errors="replace",
            capture_output=True, timeout=timeout,
        )
        return result.stdout + (result.stderr if result.stderr else "")
    except subprocess.CalledProcessError as e:
        return (e.stdout or "") + (e.stderr or "")
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return "ERROR: " + str(e)[:100]


def scp_file(local_path: Path, remote_path: str, dry_run: bool = False) -> bool:
    """Copy a single file via SCP."""
    if dry_run:
        print("  [DRY SCP] " + str(local_path) + " -> " + remote_path)
        return True
    scp_cmd = ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", str(local_path), HOST + ":" + remote_path]
    try:
        result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0
    except Exception as e:
        print("    SCP error: " + str(e)[:80])
        return False


def main() -> None:
    dry = "--dry-run" in sys.argv
    sync_only = "--sync-only" in sys.argv

    # 1. Create remote directory structure
    print("[1/4] Creating remote workspace...", flush=True)
    dirs_cmd = "mkdir -p ~/cns_font/src/v4 ~/cns_font/src/v3 ~/cns_font/src/v2 ~/cns_font/scripts ~/cns_font/checkpoints ~/cns_font/outputs ~/cns_font/datasets ~/cns_font/data_v2/tg8105 ~/cns_font/data_v3"
    out = ssh_run(dirs_cmd, dry_run=dry)
    if out:
        print("  " + out.strip()[:200], flush=True)

    # 2. Sync files via SCP
    print("[2/4] Syncing source files via SCP...", flush=True)
    n_ok, n_fail = 0, 0
    for rel_path in FILES_TO_SYNC:
        local_path = ROOT / rel_path
        if not local_path.exists():
            print("  [SKIP] " + rel_path + " (not found)")
            n_fail += 1
            continue

        remote_path = REMOTE_ROOT + "/" + rel_path
        ok = scp_file(local_path, remote_path, dry_run=dry)
        size_kb = local_path.stat().st_size / 1024
        status = "OK" if ok else "FAIL"
        print("  [" + status + "] " + rel_path + " (" + str(int(size_kb)) + "KB)")
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print("  Synced: " + str(n_ok) + ", Failed: " + str(n_fail), flush=True)

    if dry:
        print("\n[Dry Run] Done.")
        return

    # 3. Verify
    print("[3/4] Verifying remote files...", flush=True)
    out = ssh_run("cd ~/cns_font && find . -name '*.py' | wc -l && du -sh .", timeout=30)
    print("  " + out.strip()[:200] if out else "  (no output)", flush=True)

    # 4. Run VRAM test
    if not sync_only:
        print("\n[4/4] Running VRAM test on RTX 4090...", flush=True)
        out = ssh_run("cd ~/cns_font && python scripts/v4_vram_test.py", conda_env=CONDA_ENV, timeout=300)
        print(out if out else "  (no output)", flush=True)

    print("\n[Done]", flush=True)


if __name__ == "__main__":
    main()
