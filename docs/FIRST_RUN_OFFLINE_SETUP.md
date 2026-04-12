# First-Run Offline Setup

Date: 2026-04-12

This document defines the supported first-run path for a new machine.

## Scope

The supported default path is:

- local webapp
- PhoWhisper
- Whisper V2 on `faster-whisper`
- Silero VAD
- SpeechBrain diarization
- optional ProtonX correction
- optional Vistral local LLM through `llama.cpp`

This path intentionally excludes:

- `vllm`
- `pyannote` in the default web workflow
- `WhisperX` in the base install

Those components remain optional and should not block the first successful run.

## 1. Base environment

Requirements:

- Windows or Linux machine
- Python `3.11+`
- `ffmpeg` available in `PATH`
- internet access only for the initial dependency and model download phase

Install the supported runtime dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

If you want optional `WhisperX` or experimental extras:

```powershell
pip install -r requirements-optional.txt
```

If you want to run tests on the machine:

```powershell
pip install -r requirements-dev.txt
```

## 2. Download and store models for offline use

Core-only model pack:

```powershell
python scripts/setup_models.py --profile core
```

Full webapp pack including ProtonX and local LLM:

```powershell
python scripts/setup_models.py --profile full
```

Optional `WhisperX` cache:

```powershell
python scripts/setup_models.py --profile full --include-whisperx
```

By default the models are stored under:

- `models/phowhisper-safe`
- `models/faster-whisper-large-v2`
- `models/silero`
- `models/speechbrain`
- `models/protonx`
- `models/vistral/vistral-7b-chat-Q4_K_M.gguf`

After these files are in place, the app can run offline.

## 3. Verify the installation

Core profile:

```powershell
python scripts/check_installation.py --profile core
```

Full profile:

```powershell
python scripts/check_installation.py --profile full
```

The check validates:

- base Python imports
- `ffmpeg` availability
- webapp import
- offline model inventory

## 4. First run

Start the webapp:

```powershell
python webapp.py
```

Open:

- `http://127.0.0.1:8000`

Recommended first smoke run:

1. upload a short audio clip
2. use `PhoWhisper`
3. keep `Lọc khoảng lặng` enabled
4. keep the rest off
5. run one base transcription
6. if needed, add one optional step at a time

## 5. GPU and CPU behavior

The project now treats GPU as the preferred device in the UI and API defaults.

Expected behavior:

- if CUDA is available, GPU is offered as the default path
- if CUDA is not available, the UI falls back to CPU
- for protected environments or busy workstations, the operator can still force CPU manually

## 6. Offline transfer to another machine

If the target machine should run with no internet at all:

1. clone or copy the repository
2. create the Python environment and install dependencies
3. run `scripts/setup_models.py` on a connected machine, or copy the populated `models/` tree from an existing machine
4. copy the `models/` directory to the target machine
5. run `python scripts/check_installation.py --profile full`
6. start `python webapp.py`

The app itself does not require internet once dependencies and local model files are already in place.
