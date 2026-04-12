#!/bin/bash
# WSL Setup Script for Cherry Core V2
# Maps to E:\research\Cherry2\cherry_core

WSL_PATH="/mnt/e/research/Cherry2/cherry_core"
VENV_PATH="$WSL_PATH/venv_wsl"
PASSWORD="a"

echo "🚀 [WSL] Updating Ubuntu packages..."
echo "$PASSWORD" | sudo -S apt update
echo "$PASSWORD" | sudo -S apt install -y python3.11 python3.11-venv python3-pip

echo "🐍 [WSL] Creating Virtual Environment at $VENV_PATH..."
if [ ! -d "$VENV_PATH" ]; then
    python3.11 -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"

echo "📦 [WSL] Installing vLLM..."
pip install vllm
pip install -r "$WSL_PATH/requirements.txt"

echo "✅ [WSL] Setup Complete!"
