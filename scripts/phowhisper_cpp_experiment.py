from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from research.phowhisper_cpp.benchmarking import (
    BenchmarkConfig,
    benchmark_models,
    build_dataset_report,
    collect_dataset_entries,
    summarize_model_rows,
    write_outputs,
)
from research.phowhisper_cpp.evidence import render_paper_evidence
from research.phowhisper_cpp.transfer import materialize_dataset, package_artifacts_bundle
from research.phowhisper_cpp.workspace import (
    ensure_workspace_layout,
    load_project_paths_config,
    sha256_file,
    write_default_project_paths,
    write_json,
)

DEFAULT_WORKSPACE_ROOT = ROOT_DIR / "output" / "phowhisper_cpp_workspace"

WHISPER_CPP_CONVERTER_CANDIDATES = (
    "models/convert-h5-to-ggml.py",
    "models/convert-pt-to-ggml.py",
    "models/convert-h5-to-gguf.py",
    "models/convert-pt-to-gguf.py",
)
WHISPER_QUANTIZE_CANDIDATES = (
    "build/bin/whisper-quantize.exe",
    "build/bin/whisper-quantize",
    "build/bin/Release/whisper-quantize.exe",
    "build/bin/Release/whisper-quantize",
)
BUNDLE_FILES = (
    "scripts/phowhisper_cpp_experiment.py",
    "research/__init__.py",
    "research/phowhisper_cpp/__init__.py",
    "research/phowhisper_cpp/workspace.py",
    "research/phowhisper_cpp/benchmarking.py",
    "research/phowhisper_cpp/evidence.py",
    "research/phowhisper_cpp/transfer.py",
    "docs/PHOWHISPER_CPP_RESEARCH_2026-04-11.md",
    "docs/PHOWHISPER_CPP_EXPERIMENT_PROTOCOL.md",
    "docs/PHOWHISPER_CPP_COLAB_PRO_WORKFLOW.md",
    "notebooks/colab/PhoWhisper_CPP_Colab_Pro.ipynb",
)


def parse_model_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected model format label=path")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("Model label and path cannot be empty")
    return label, Path(raw_path).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PhoWhisper.cpp experiment toolkit for conversion, quantization, and benchmark packaging.",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-workspace", help="Create a clean PhoWhisper.cpp experiment workspace.")
    init_parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)

    convert_parser = subparsers.add_parser("convert", help="Convert PhoWhisper Hugging Face weights for whisper.cpp.")
    convert_parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    convert_parser.add_argument("--whisper-cpp-dir", type=Path, required=True)
    convert_parser.add_argument("--openai-whisper-dir", type=Path, required=True)
    convert_parser.add_argument("--hf-model-dir", type=Path, required=True)
    convert_parser.add_argument("--artifact-tag", default=None, help="Optional tag appended to the conversion output directory.")
    convert_parser.add_argument("--q4-mode", default="q4_0")
    convert_parser.add_argument("--q5-mode", default="q5_0")
    convert_parser.add_argument("--skip-q4", action="store_true")
    convert_parser.add_argument("--skip-q5", action="store_true")
    convert_parser.add_argument("--force", action="store_true", help="Overwrite existing converted outputs.")

    fetch_parser = subparsers.add_parser(
        "fetch-dataset",
        help="Populate the benchmark dataset inside the workspace from a local archive or direct cloud URL.",
    )
    fetch_parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    fetch_parser.add_argument("--dataset-id", default="benchmark_vi_longform_v1")
    fetch_source = fetch_parser.add_mutually_exclusive_group(required=True)
    fetch_source.add_argument("--archive-path", type=Path)
    fetch_source.add_argument("--source-url")
    fetch_parser.add_argument("--force", action="store_true")
    fetch_parser.add_argument("--drop-archive", action="store_true", help="Delete the downloaded archive after extraction.")

    benchmark_parser = subparsers.add_parser("benchmark", help="Run or score a whisper.cpp benchmark.")
    benchmark_parser.add_argument("--dataset-dir", type=Path, required=True)
    benchmark_parser.add_argument("--output-dir", type=Path, required=True)
    benchmark_parser.add_argument("--whisper-cli", type=Path, required=True)
    benchmark_parser.add_argument("--language", default="vi")
    benchmark_parser.add_argument("--threads", type=int, default=4)
    benchmark_parser.add_argument("--min-index", type=int, default=10)
    benchmark_parser.add_argument("--max-index", type=int, default=49)
    benchmark_parser.add_argument("--max-files", type=int, default=None)
    benchmark_parser.add_argument("--model", dest="models", action="append", type=parse_model_arg, required=True)
    benchmark_parser.add_argument("--extra-arg", dest="extra_args", action="append", default=[])
    benchmark_parser.add_argument("--no-reuse-existing-output", action="store_true")
    benchmark_parser.add_argument("--report-only", action="store_true")

    evidence_parser = subparsers.add_parser(
        "render-evidence",
        help="Generate paper-ready figures and qualitative evidence from a benchmark run.",
    )
    evidence_parser.add_argument("--benchmark-dir", type=Path, required=True)
    evidence_parser.add_argument("--conversion-dir", type=Path, default=None)
    evidence_parser.add_argument("--qualitative-top-n", type=int, default=3)

    package_results_parser = subparsers.add_parser(
        "package-results",
        help="Create a single ZIP bundle containing benchmark artifacts, evidence, and optionally converted models.",
    )
    package_results_parser.add_argument("--benchmark-dir", type=Path, required=True)
    package_results_parser.add_argument("--output-zip", type=Path, required=True)
    package_results_parser.add_argument("--conversion-dir", type=Path, default=None)

    package_parser = subparsers.add_parser("package-colab", help="Bundle the experiment toolkit for Colab.")
    package_parser.add_argument("--bundle-path", type=Path, default=ROOT_DIR / "output" / "phowhisper_cpp_colab_bundle.zip")
    return parser


def run_command(command: list[str], cwd: Path | None = None) -> None:
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=str(cwd) if cwd else None)


def locate_existing(base_dir: Path, candidates: tuple[str, ...], label: str) -> Path:
    for relative_path in candidates:
        candidate = base_dir / relative_path
        if candidate.exists():
            return candidate
    checked = [str(base_dir / item) for item in candidates]
    raise FileNotFoundError(f"Could not locate {label}. Checked: {checked}")


def ensure_safetensors(hf_model_dir: Path) -> Path | None:
    safe_path = hf_model_dir / "model.safetensors"
    if safe_path.exists():
        return safe_path

    bin_path = hf_model_dir / "pytorch_model.bin"
    if not bin_path.exists():
        return None

    try:
        import torch
        from safetensors.torch import save_file
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Need `torch` and `safetensors` to create a fallback model.safetensors from pytorch_model.bin."
        ) from exc

    state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
    if "proj_out.weight" in state_dict:
        del state_dict["proj_out.weight"]
    contiguous_state_dict = {
        key: value.contiguous()
        for key, value in state_dict.items()
        if hasattr(value, "contiguous")
    }
    save_file(contiguous_state_dict, str(safe_path))
    return safe_path


def create_conversion_manifest(
    workspace_root: Path,
    artifact_root: Path,
    hf_model_dir: Path,
    whisper_cpp_dir: Path,
    openai_whisper_dir: Path,
    q4_mode: str,
    q5_mode: str,
    converted_files: list[Path],
) -> Path:
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "workspace_root": str(workspace_root.resolve()),
        "artifact_root": str(artifact_root.resolve()),
        "hf_model_dir": str(hf_model_dir.resolve()),
        "whisper_cpp_dir": str(whisper_cpp_dir.resolve()),
        "openai_whisper_dir": str(openai_whisper_dir.resolve()),
        "quantization": {
            "q4_mode": q4_mode,
            "q5_mode": q5_mode,
        },
        "files": [
            {
                "name": path.name,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(converted_files)
            if path.exists()
        ],
    }
    return write_json(artifact_root / "conversion_manifest.json", manifest)


def init_workspace(workspace_root: Path) -> None:
    config_path = write_default_project_paths(workspace_root)
    readme_path = workspace_root / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# PhoWhisper.cpp Experiment Workspace",
                "",
                "This workspace mirrors the old research layout but is generated from Cherry Core.",
                "",
                "Key paths:",
                "",
                "- `configs/project_paths.json`",
                "- `external/vendor/whisper.cpp`",
                "- `external/vendor/openai-whisper`",
                "- `external/models/hf/phowhisper-large`",
                "- `external/models/ggml`",
                "- `artifacts/benchmarks/whispercpp`",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote workspace config: {config_path}")
    print(f"Wrote workspace README: {readme_path}")


def run_conversion(args: argparse.Namespace) -> None:
    workspace_root = args.workspace_root.resolve()
    ensure_workspace_layout(workspace_root)
    load_project_paths_config(workspace_root)

    whisper_cpp_dir = args.whisper_cpp_dir.expanduser().resolve()
    openai_whisper_dir = args.openai_whisper_dir.expanduser().resolve()
    hf_model_dir = args.hf_model_dir.expanduser().resolve()
    artifact_tag = args.artifact_tag or datetime.now(timezone.utc).strftime("conversion_%Y-%m-%d_%H%M%SZ")
    artifact_root = workspace_root / "artifacts" / "conversion" / artifact_tag
    raw_convert_dir = artifact_root / "convert_raw"
    model_out_dir = artifact_root / "models"
    raw_convert_dir.mkdir(parents=True, exist_ok=True)
    model_out_dir.mkdir(parents=True, exist_ok=True)

    converter_script = locate_existing(whisper_cpp_dir, WHISPER_CPP_CONVERTER_CANDIDATES, "whisper.cpp converter script")
    quantize_bin = locate_existing(whisper_cpp_dir, WHISPER_QUANTIZE_CANDIDATES, "whisper-quantize binary")

    safe_path = ensure_safetensors(hf_model_dir)
    if safe_path:
        print(f"Safetensors ready: {safe_path}")

    fp16_output = model_out_dir / "ggml-phowhisper-large.bin"
    if fp16_output.exists() and not args.force:
        print(f"Reusing existing converted model: {fp16_output}")
    else:
        run_command(
            [
                sys.executable,
                str(converter_script),
                str(hf_model_dir),
                str(openai_whisper_dir),
                str(raw_convert_dir),
            ]
        )
        candidate_paths = (
            raw_convert_dir / "ggml-model.bin",
            raw_convert_dir / "ggml-phowhisper-large.bin",
            raw_convert_dir / "gguf-model.bin",
            raw_convert_dir / "gguf-phowhisper-large.bin",
        )
        source_path = next((path for path in candidate_paths if path.exists()), None)
        if source_path is None:
            raise FileNotFoundError(f"Conversion finished but no runtime model file was found under {raw_convert_dir}")
        if fp16_output.exists():
            fp16_output.unlink()
        shutil.copy2(source_path, fp16_output)

    converted_files = [fp16_output]
    if not args.skip_q4:
        q4_output = model_out_dir / f"ggml-phowhisper-large-{args.q4_mode}.bin"
        if not q4_output.exists() or args.force:
            run_command([str(quantize_bin), str(fp16_output), str(q4_output), args.q4_mode])
        converted_files.append(q4_output)

    if not args.skip_q5:
        q5_output = model_out_dir / f"ggml-phowhisper-large-{args.q5_mode}.bin"
        if not q5_output.exists() or args.force:
            run_command([str(quantize_bin), str(fp16_output), str(q5_output), args.q5_mode])
        converted_files.append(q5_output)

    manifest_path = create_conversion_manifest(
        workspace_root=workspace_root,
        artifact_root=artifact_root,
        hf_model_dir=hf_model_dir,
        whisper_cpp_dir=whisper_cpp_dir,
        openai_whisper_dir=openai_whisper_dir,
        q4_mode=args.q4_mode,
        q5_mode=args.q5_mode,
        converted_files=converted_files,
    )
    print(f"Wrote conversion manifest: {manifest_path}")


def fetch_dataset(args: argparse.Namespace) -> None:
    workspace_root = args.workspace_root.expanduser().resolve()
    ensure_workspace_layout(workspace_root)
    dataset_dir, manifest_path = materialize_dataset(
        workspace_root=workspace_root,
        dataset_id=args.dataset_id,
        archive_path=args.archive_path,
        source_url=args.source_url,
        force=args.force,
        keep_archive=not args.drop_archive,
    )
    print(f"Dataset directory: {dataset_dir}")
    print(f"Dataset manifest: {manifest_path}")


def run_benchmark(args: argparse.Namespace) -> None:
    config = BenchmarkConfig(
        dataset_dir=args.dataset_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        whisper_cli=args.whisper_cli.expanduser().resolve(),
        language=args.language,
        threads=args.threads,
        min_index=args.min_index,
        max_index=args.max_index,
        max_files=args.max_files,
        models=args.models,
        extra_args=args.extra_args,
        reuse_existing_output=not args.no_reuse_existing_output,
        report_only=args.report_only,
    )

    if not config.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {config.dataset_dir}")
    if not config.report_only and not config.whisper_cli.exists():
        raise FileNotFoundError(f"whisper-cli not found: {config.whisper_cli}")

    entries = collect_dataset_entries(config.dataset_dir, config.min_index, config.max_index, config.max_files)
    if not entries:
        raise RuntimeError("No paired audio/transcript files were found in the requested benchmark slice.")

    dataset_report, dataset_rows = build_dataset_report(entries, config.dataset_dir, config.min_index, config.max_index)
    benchmark_rows = benchmark_models(entries, config)
    model_summaries, hardest_files = summarize_model_rows(benchmark_rows)
    written_paths = write_outputs(
        output_dir=config.output_dir,
        config=config,
        dataset_report=dataset_report,
        dataset_rows=dataset_rows,
        benchmark_rows=benchmark_rows,
        model_summaries=model_summaries,
        hardest_files=hardest_files,
    )
    for label, path in written_paths.items():
        print(f"{label}: {path}")


def render_evidence(args: argparse.Namespace) -> None:
    written_paths = render_paper_evidence(
        benchmark_dir=args.benchmark_dir,
        conversion_dir=args.conversion_dir,
        qualitative_top_n=args.qualitative_top_n,
    )
    for label, path in written_paths.items():
        print(f"{label}: {path}")


def package_results(args: argparse.Namespace) -> None:
    bundle_path = package_artifacts_bundle(
        output_zip=args.output_zip,
        benchmark_dir=args.benchmark_dir,
        conversion_dir=args.conversion_dir,
    )
    print(f"Packaged results bundle: {bundle_path}")


def package_colab_bundle(bundle_path: Path) -> None:
    bundle_path = bundle_path.expanduser().resolve()
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative_file in BUNDLE_FILES:
            source_path = ROOT_DIR / relative_file
            if source_path.exists():
                archive.write(source_path, arcname=relative_file.replace("\\", "/"))
    print(f"Created Colab bundle: {bundle_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "init-workspace":
        init_workspace(args.workspace_root.expanduser().resolve())
        return
    if args.command == "convert":
        run_conversion(args)
        return
    if args.command == "fetch-dataset":
        fetch_dataset(args)
        return
    if args.command == "benchmark":
        run_benchmark(args)
        return
    if args.command == "render-evidence":
        render_evidence(args)
        return
    if args.command == "package-results":
        package_results(args)
        return
    if args.command == "package-colab":
        package_colab_bundle(args.bundle_path)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
