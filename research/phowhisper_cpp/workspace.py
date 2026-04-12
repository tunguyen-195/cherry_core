from __future__ import annotations

import json
import hashlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PROJECT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "project_name": "phowhisper-cpp-research",
    "paths": {
        "archive": "archive",
        "artifacts": "artifacts",
        "configs": "configs",
        "data": "data",
        "docs": "docs",
        "external": "external",
        "reports": "reports",
        "benchmark_dataset_default": "data/datasets/benchmark_vi_longform_v1",
        "benchmark_dataset_archive_default": "data/archives/benchmark_vi_longform_v1.zip",
        "whisper_cpp_vendor": "external/vendor/whisper.cpp",
        "openai_whisper_vendor": "external/vendor/openai-whisper",
        "hf_phowhisper_large": "external/models/hf/phowhisper-large",
        "ggml_models": "external/models/ggml",
        "whispercpp_ad_hoc": "artifacts/benchmarks/whispercpp/ad_hoc",
        "whispercpp_q5_40_long": "artifacts/benchmarks/whispercpp/q5_40_long",
    },
    "datasets": {
        "benchmark_vi_longform_v1": {
            "path": "data/datasets/benchmark_vi_longform_v1",
            "archive": "data/archives/benchmark_vi_longform_v1.zip",
            "language": "vi",
            "index_range": [10, 49],
            "description": "Vietnamese long-form benchmark set for PhoWhisper whisper.cpp evaluation",
        }
    },
    "experiments": {
        "q5_40_long": {
            "artifact_root": "artifacts/benchmarks/whispercpp/q5_40_long",
            "default_run_id": "run_local",
            "description": "40-file q5 benchmark comparing PhoWhisper and Whisper large baselines",
        },
        "full_precision_sanity": {
            "artifact_root": "artifacts/benchmarks/whispercpp/full_precision_sanity",
            "description": "Short non-quantized benchmark sanity run",
        },
    },
}


WORKSPACE_DIRS = (
    "archive",
    "artifacts/benchmarks/whispercpp",
    "configs",
    "data/archives",
    "data/datasets",
    "docs",
    "external/models/ggml",
    "external/models/hf",
    "external/vendor",
    "reports/model-selection",
)


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path
    config: dict[str, Any]

    def resolve(self, key: str, default: str | Path | None = None) -> Path:
        raw_value = self.config.get("paths", {}).get(key)
        if raw_value is None:
            if default is None:
                raise KeyError(f"Unknown path key: {key}")
            raw_value = default
        return (self.root / Path(raw_value)).resolve()

    def dataset(self, dataset_id: str, default: str | Path | None = None) -> Path:
        dataset_entry = self.config.get("datasets", {}).get(dataset_id, {})
        raw_value = dataset_entry.get("path", default)
        if raw_value is None:
            raise KeyError(f"Unknown dataset id: {dataset_id}")
        return (self.root / Path(raw_value)).resolve()

    def dataset_archive(self, dataset_id: str) -> Path | None:
        dataset_entry = self.config.get("datasets", {}).get(dataset_id, {})
        raw_value = dataset_entry.get("archive")
        if not raw_value:
            return None
        return (self.root / Path(raw_value)).resolve()

    def experiment_root(self, experiment_id: str, default: str | Path | None = None) -> Path:
        experiment_entry = self.config.get("experiments", {}).get(experiment_id, {})
        raw_value = experiment_entry.get("artifact_root", default)
        if raw_value is None:
            raise KeyError(f"Unknown experiment id: {experiment_id}")
        return (self.root / Path(raw_value)).resolve()


def ensure_workspace_layout(root: Path) -> None:
    for relative_path in WORKSPACE_DIRS:
        (root / relative_path).mkdir(parents=True, exist_ok=True)


def load_project_paths_config(root: Path) -> dict[str, Any]:
    config_path = root / "configs" / "project_paths.json"
    if not config_path.exists():
        return deepcopy(DEFAULT_PROJECT_CONFIG)
    return json.loads(config_path.read_text(encoding="utf-8"))


def write_default_project_paths(root: Path) -> Path:
    ensure_workspace_layout(root)
    config_path = root / "configs" / "project_paths.json"
    config_path.write_text(
        json.dumps(deepcopy(DEFAULT_PROJECT_CONFIG), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return config_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def relative_to_root(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)
