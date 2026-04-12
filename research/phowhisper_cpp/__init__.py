"""Helpers for the PhoWhisper.cpp conversion and benchmark lane."""

from .benchmarking import (
    BenchmarkConfig,
    build_dataset_report,
    compute_max_ngram_repeat,
    collect_dataset_entries,
    compute_cer,
    compute_repeat_ngram_ratio,
    compute_wer,
    normalize_text,
    summarize_model_rows,
    write_outputs,
)
from .evidence import render_paper_evidence
from .transfer import materialize_dataset, package_artifacts_bundle
from .workspace import (
    DEFAULT_PROJECT_CONFIG,
    ExperimentPaths,
    ensure_workspace_layout,
    load_project_paths_config,
    relative_to_root,
    sha256_file,
    write_default_project_paths,
    write_json,
)

__all__ = [
    "BenchmarkConfig",
    "DEFAULT_PROJECT_CONFIG",
    "ExperimentPaths",
    "build_dataset_report",
    "collect_dataset_entries",
    "compute_cer",
    "compute_max_ngram_repeat",
    "compute_repeat_ngram_ratio",
    "compute_wer",
    "ensure_workspace_layout",
    "load_project_paths_config",
    "materialize_dataset",
    "normalize_text",
    "package_artifacts_bundle",
    "render_paper_evidence",
    "relative_to_root",
    "sha256_file",
    "summarize_model_rows",
    "write_default_project_paths",
    "write_json",
    "write_outputs",
]
