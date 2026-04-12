from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from application.services.model_inventory_service import ModelInventoryService  # noqa: E402
from presentation.web.app import create_app  # noqa: E402


BASE_IMPORTS = [
    "numpy",
    "scipy",
    "soundfile",
    "yaml",
    "jinja2",
    "fastapi",
    "uvicorn",
    "multipart",
    "torch",
    "transformers",
    "torchaudio",
    "sentencepiece",
    "huggingface_hub",
    "faster_whisper",
    "llama_cpp",
    "speechbrain",
    "intervaltree",
]

FULL_FEATURE_IDS = {
    "phowhisper",
    "whisper-v2",
    "silero-vad",
    "speechbrain",
    "protonx",
    "vistral",
}

CORE_FEATURE_IDS = {
    "phowhisper",
    "whisper-v2",
    "silero-vad",
    "speechbrain",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the supported first-run Cherry Core installation.")
    parser.add_argument("--profile", choices=["core", "full"], default="full")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text.")
    return parser.parse_args()


def collect_status(profile: str) -> dict[str, Any]:
    imports: dict[str, bool] = {}
    import_errors: dict[str, str] = {}
    for module_name in BASE_IMPORTS:
        try:
            importlib.import_module(module_name)
            imports[module_name] = True
        except Exception as exc:
            imports[module_name] = False
            import_errors[module_name] = str(exc)

    ffmpeg_path = shutil.which("ffmpeg")

    app = create_app()
    inventory = ModelInventoryService().get_inventory()
    items = {item["model_id"]: item for item in inventory["items"]}
    required_ids = FULL_FEATURE_IDS if profile == "full" else CORE_FEATURE_IDS
    missing_models = sorted(model_id for model_id in required_ids if not items.get(model_id, {}).get("offline_ready", False))

    status = {
        "python": sys.version.split()[0],
        "ffmpeg_in_path": bool(ffmpeg_path),
        "ffmpeg_path": ffmpeg_path,
        "imports": imports,
        "import_errors": import_errors,
        "app_import_ok": bool(app),
        "inventory": inventory,
        "missing_required_models": missing_models,
    }
    status["ready"] = status["ffmpeg_in_path"] and all(imports.values()) and not missing_models
    return status


def render_text(status: dict[str, Any], profile: str) -> str:
    lines = [
        "Cherry Core installation check",
        f"Profile: {profile}",
        f"Python: {status['python']}",
        f"ffmpeg: {'OK' if status['ffmpeg_in_path'] else 'MISSING'}",
        f"App import: {'OK' if status['app_import_ok'] else 'FAILED'}",
        "",
        "Base imports:",
    ]
    for module_name, ok in status["imports"].items():
        marker = "OK" if ok else "MISSING"
        suffix = ""
        if not ok:
            suffix = f" -> {status['import_errors'].get(module_name, 'unknown error')}"
        lines.append(f"- {module_name}: {marker}{suffix}")

    lines.append("")
    lines.append("Required offline models:")
    for model_id in (FULL_FEATURE_IDS if profile == "full" else CORE_FEATURE_IDS):
        item = next((item for item in status["inventory"]["items"] if item["model_id"] == model_id), None)
        ready = bool(item and item.get("offline_ready"))
        path = item.get("path") if item else "unknown"
        lines.append(f"- {model_id}: {'READY' if ready else 'MISSING'} ({path})")

    lines.append("")
    if status["ready"]:
        lines.append("Overall status: READY")
    else:
        lines.append("Overall status: NOT READY")
        if not status["ffmpeg_in_path"]:
            lines.append("- Add ffmpeg to PATH.")
        if status["missing_required_models"]:
            lines.append("- Download missing models with: python scripts/setup_models.py --profile %s" % profile)
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    status = collect_status(args.profile)
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        print(render_text(status, args.profile))
    return 0 if status["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
