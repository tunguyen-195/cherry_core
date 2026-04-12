#!/bin/bash
# WSL Verify Script
# Maps to E:\research\Cherry2\cherry_core

WSL_PATH="/mnt/e/research/Cherry2/cherry_core"
VENV_PATH="$WSL_PATH/venv_wsl"

echo "👉 [WSL] Running Verification Test..."
source "$VENV_PATH/bin/activate"
export PYTHONUTF8=1
cd "$WSL_PATH"

python3 scripts/analyze_manual_transcript.py
