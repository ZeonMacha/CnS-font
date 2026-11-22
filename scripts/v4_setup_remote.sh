#!/usr/bin/env bash
# One-time setup: create remote CnS workspace and sync v4 source files.
# Run via: python safe_exec.py llm24g -e comfyui "bash /tmp/v4_setup.sh"

set -euo pipefail

REMOTE_ROOT=~/cns_font
mkdir -p "${REMOTE_ROOT}"/{src/v4,scripts,checkpoints,outputs,datasets,data_v2/tg8105,data_v3}

# We'll receive the source files via base64 from the local machine
# This script just creates the directory structure

echo "[setup] Remote workspace created at ${REMOTE_ROOT}"
ls -la "${REMOTE_ROOT}/"
