#!/usr/bin/env python3
"""
Deploy v4.0 source files to the remote 4090 (llm24g) and run VRAM test.

Usage:
  python scripts/v4_deploy_test.py          # Sync files + run VRAM test
  python scripts/v4_deploy_test.py --dry    # Preview only
"""

from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HOST = "llm24g"
CONDA_ENV = "comfyui"
REMOTE_ROOT = "~/cns_font"

# Files to sync (relative to project root)
FILES_TO_SYNC = [
    "src/v4/__init__.py",
    "src/v4/model_v4.py",
    "src/v4/dataset_v4.py",
    "src/v4/losses_v4.py",
    "scripts/v4_train.py",
]


def b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def run_remote(command: str, conda_env: str = None, dry_run: bool = False) -> str:
    script_lines = []
    if conda_env:
        script_lines.append(f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate {conda_env}")
    script_lines.append(command)
    remote_script = "\n".join(script_lines)
    b64_payload = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
    ssh_cmd = ["ssh", "-o", "BatchMode=yes", HOST, f"echo {b64_payload} | base64 -d | bash"]

    if dry_run:
        print(f"[Dry Run] {HOST}:\n{remote_script}\n")
        return ""

    result = subprocess.run(ssh_cmd, check=True, text=True, encoding="utf-8", errors="replace", capture_output=True)
    output = result.stdout
    if result.stderr:
        output += result.stderr
    return output


def main() -> None:
    dry = "--dry" in sys.argv

    # 1. Create remote directory
    print("[1/3] Creating remote directory...", flush=True)
    out = run_remote(f"mkdir -p {REMOTE_ROOT}/src/v4 {REMOTE_ROOT}/scripts {REMOTE_ROOT}/checkpoints {REMOTE_ROOT}/outputs {REMOTE_ROOT}/datasets", dry_run=dry)
    print(out, end="")

    # 2. Sync files
    print("[2/3] Syncing v4 source files...", flush=True)
    for rel_path in FILES_TO_SYNC:
        local_path = ROOT / rel_path
        if not local_path.exists():
            print(f"  [SKIP] {rel_path} (not found locally)")
            continue
        remote_path = f"{REMOTE_ROOT}/{rel_path}"
        b64 = b64_file(local_path)
        # Write file via base64 decode
        cmd = f"echo {b64} | base64 -d > {remote_path}"
        out = run_remote(cmd, dry_run=dry)
        print(f"  [SYNC] {rel_path} → {remote_path}")

    if dry:
        print("\n[Dry Run] Done. No files transferred.")
        return

    # 3. Run VRAM test
    print("\n[3/3] Running VRAM test on 4090...", flush=True)
    test_cmd = f"cd {REMOTE_ROOT} && python scripts/v4_train.py --noto_sans /dev/null --test_vram"
    out = run_remote(test_cmd, conda_env=CONDA_ENV)
    print(out, end="")

    print("\n[Done]", flush=True)


if __name__ == "__main__":
    main()
