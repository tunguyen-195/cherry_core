from __future__ import annotations

import argparse
import io
import logging
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.config import MODELS_DIR, PROTONX_PATH, SILERO_PATH, SPEECHBRAIN_PATH  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("setup_models")

PHOWHISPER_REPO = "vinai/PhoWhisper-large"
WHISPER_V2_CT2_REPO = "Systran/faster-whisper-large-v2"
SPEECHBRAIN_REPO = "speechbrain/spkrec-ecapa-voxceleb"
PROTONX_REPO = "protonx-models/protonx-legal-tc"
VISTRAL_GGUF_REPO = "janhq/Vistral-7b-Chat-GGUF"
VISTRAL_TARGET = MODELS_DIR / "vistral" / "vistral-7b-chat-Q4_K_M.gguf"
SILERO_ZIP_URL = "https://github.com/snakers4/silero-vad/archive/refs/heads/master.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Cherry Core for the supported first-run offline path.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--profile",
        choices=["core", "full"],
        default="full",
        help="core downloads ASR/VAD/diarization essentials; full adds ProtonX and local LLM.",
    )
    parser.add_argument(
        "--include-whisperx",
        action="store_true",
        help="Also cache WhisperX-specific ASR/alignment assets via the dedicated setup script.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if target folders already contain files.",
    )
    return parser.parse_args()


def ensure_snapshot(repo_id: str, local_dir: Path, force: bool = False, allow_patterns: list[str] | None = None) -> None:
    if local_dir.exists() and any(local_dir.iterdir()) and not force:
        logger.info("✅ Already present: %s -> %s", repo_id, local_dir)
        return

    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info("⏳ Downloading %s -> %s", repo_id, local_dir)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
    )
    logger.info("✅ Saved: %s", local_dir)


def setup_phowhisper(force: bool = False) -> None:
    target = MODELS_DIR / "phowhisper-safe"
    ensure_snapshot(PHOWHISPER_REPO, target, force=force)


def setup_whisper_v2_ct2(force: bool = False) -> None:
    target = MODELS_DIR / "faster-whisper-large-v2"
    ensure_snapshot(WHISPER_V2_CT2_REPO, target, force=force)


def setup_speechbrain(force: bool = False) -> None:
    ensure_snapshot(SPEECHBRAIN_REPO, SPEECHBRAIN_PATH, force=force)


def setup_protonx(force: bool = False) -> None:
    ensure_snapshot(PROTONX_REPO, PROTONX_PATH, force=force)


def setup_vistral(force: bool = False) -> None:
    target_dir = VISTRAL_TARGET.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    if VISTRAL_TARGET.exists() and not force:
        logger.info("✅ Already present: %s", VISTRAL_TARGET)
        return

    temp_dir = target_dir / "_vistral_download"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("⏳ Downloading local LLM GGUF from %s", VISTRAL_GGUF_REPO)
        snapshot_download(
            repo_id=VISTRAL_GGUF_REPO,
            local_dir=str(temp_dir),
            allow_patterns=["*Q4_K_M.gguf", "*q4_k_m.gguf"],
        )

        candidates = sorted(temp_dir.rglob("*Q4_K_M.gguf")) + sorted(temp_dir.rglob("*q4_k_m.gguf"))
        if not candidates:
            raise FileNotFoundError("Could not find a Q4_K_M GGUF file in the downloaded snapshot.")

        shutil.copy2(candidates[0], VISTRAL_TARGET)
        logger.info("✅ Saved: %s", VISTRAL_TARGET)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def setup_silero(force: bool = False) -> None:
    SILERO_PATH.mkdir(parents=True, exist_ok=True)
    jit_target = SILERO_PATH / "silero_vad.jit"
    utils_target = SILERO_PATH / "utils_vad.py"

    if jit_target.exists() and utils_target.exists() and not force:
        logger.info("✅ Already present: %s", SILERO_PATH)
        return

    logger.info("⏳ Downloading Silero VAD assets -> %s", SILERO_PATH)
    with urllib.request.urlopen(SILERO_ZIP_URL) as response:
        zip_content = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_content)) as archive:
        jit_name = next((name for name in archive.namelist() if name.endswith("silero_vad.jit")), None)
        utils_name = next((name for name in archive.namelist() if name.endswith("utils_vad.py")), None)

        if not jit_name or not utils_name:
            raise FileNotFoundError("Silero VAD package layout changed; required files were not found in the archive.")

        with archive.open(jit_name) as source, open(jit_target, "wb") as target:
            shutil.copyfileobj(source, target)
        with archive.open(utils_name) as source, open(utils_target, "wb") as target:
            shutil.copyfileobj(source, target)

    (SILERO_PATH / "__init__.py").touch()
    logger.info("✅ Saved: %s", SILERO_PATH)


def setup_whisperx_assets(force: bool = False) -> None:
    import setup_whisperx_offline as whisperx_setup

    if force:
        logger.warning("WhisperX setup is delegated to scripts/setup_whisperx_offline.py and may reuse caches.")
    whisperx_setup.setup_directories()
    whisperx_setup.download_asr_model("large-v2")
    whisperx_setup.download_alignment_model()
    whisperx_setup.download_diarization_model()


def main() -> int:
    args = parse_args()

    logger.info("=" * 72)
    logger.info("Cherry Core first-run offline model setup")
    logger.info("Profile: %s", args.profile)
    logger.info("Models directory: %s", MODELS_DIR)
    logger.info("=" * 72)

    tasks = [
        ("PhoWhisper", lambda: setup_phowhisper(force=args.force)),
        ("Whisper V2 CTranslate2", lambda: setup_whisper_v2_ct2(force=args.force)),
        ("Silero VAD", lambda: setup_silero(force=args.force)),
        ("SpeechBrain diarization", lambda: setup_speechbrain(force=args.force)),
    ]
    if args.profile == "full":
        tasks.extend(
            [
                ("ProtonX correction", lambda: setup_protonx(force=args.force)),
                ("Vistral GGUF", lambda: setup_vistral(force=args.force)),
            ]
        )

    failures: list[str] = []
    for label, task in tasks:
        try:
            task()
        except Exception as exc:
            logger.error("❌ %s failed: %s", label, exc)
            failures.append(label)

    if args.include_whisperx:
        try:
            setup_whisperx_assets(force=args.force)
        except Exception as exc:
            logger.error("❌ WhisperX optional setup failed: %s", exc)
            failures.append("WhisperX optional setup")

    if failures:
        logger.error("Setup finished with failures: %s", ", ".join(failures))
        return 1

    logger.info("🎉 Offline model setup complete.")
    logger.info("Next step: python scripts/check_installation.py --profile %s", args.profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
