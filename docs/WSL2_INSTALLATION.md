# WSL2 SETUP GUIDE FOR vLLM (CHERRY CORE V2)

Since Native Windows support for vLLM is experimental, **WSL2 (Ubuntu 22.04)** is the recommended environment for High-Performance Serving.

## 1. Prerequisites
- **Windows 10/11** with WSL2 enabled.
- **NVIDIA Driver** (Windows side) installed. WSL2 shares this driver.
- **Docker Desktop** (Optional, if using Docker approach).

## 2. Setup Ubuntu Environment
Open your WSL2 terminal (Ubuntu) and run:

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install Python 3.11 and venv
sudo apt install python3.11 python3.11-venv python3-pip -y

# 3. Clone/Copy Project
# Access Windows files from /mnt/e/research/Cherry2...
mkdir ~/cherry_v2
cp -r /mnt/e/research/Cherry2/cherry_core ~/cherry_v2/
cd ~/cherry_v2/cherry_core

# 4. Create Virtual Env
python3.11 -m venv venv
source venv/bin/activate

# 5. Install Dependencies (Include vllm)
pip install vllm
pip install -r requirements.txt
```

## 3. Running Cherry Core in WSL2
To enable vLLM, ensure `src/config.py` has:
```python
USE_VLLM = True
```

Run the verification script:
```bash
export PYTHONUTF8=1
python scripts/analyze_manual_transcript.py
```

## 4. Troubleshooting
### "Could not load LLM Engine" (Model Format)
vLLM does **NOT** support `.gguf` files efficiently (it converts them or fails).
For best performance in Cycle 4, download the **AWQ** version of Vistral:
```bash
huggingface-cli download --resume-download "Viet-AI/vistral-7b-chat-awq" --local-dir models/vistral-awq
```
Then update `src/config.py`:
```python
LLM_MODEL_NAME = "../vistral-awq" # Path relative to models dir
```
