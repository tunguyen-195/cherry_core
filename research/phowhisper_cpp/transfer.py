from __future__ import annotations

import shutil
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .workspace import sha256_file, write_json


def extract_zip_portably(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            normalized_name = member.filename.replace("\\", "/").lstrip("/")
            if not normalized_name:
                continue
            target_path = destination / normalized_name
            if normalized_name.endswith("/"):
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source_handle, target_path.open("wb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle)


def _download_http(source_url: str, destination: Path) -> None:
    request = urllib.request.Request(
        source_url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; CherryCorePhoWhisperCpp/1.0)",
        },
    )
    with urllib.request.urlopen(request) as response, destination.open("wb") as output_handle:
        shutil.copyfileobj(response, output_handle, length=1024 * 1024)


def _download_google_drive(source_url: str, destination: Path) -> None:
    file_id = source_url.removeprefix("gdrive://").strip().strip("/")
    if not file_id:
        raise ValueError("Expected Google Drive source in the form gdrive://<file_id>")
    try:
        import gdown
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Downloading from Google Drive requires `gdown` in the active environment.") from exc

    downloaded = gdown.download(id=file_id, output=str(destination), quiet=False, fuzzy=False)
    if not downloaded:
        raise RuntimeError(f"Failed to download Google Drive file: {source_url}")


def download_archive(source_url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_url.startswith("gdrive://"):
        _download_google_drive(source_url, destination)
        return

    parsed = urllib.parse.urlparse(source_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(
            "Only direct http(s) URLs and gdrive://<file_id> are supported for automatic dataset download."
        )
    _download_http(source_url, destination)


def _find_dataset_candidate(root: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for candidate in [root, *[path for path in root.rglob("*") if path.is_dir()]]:
        wav_count = len(list(candidate.glob("audio_*.wav")))
        txt_count = len(list(candidate.glob("audio_*.txt")))
        if wav_count and txt_count:
            candidates.append((min(wav_count, txt_count), candidate))
    if not candidates:
        raise RuntimeError(f"No extracted dataset directory containing paired audio_*.wav/txt files found under {root}")
    candidates.sort(key=lambda item: (item[0], len(str(item[1]))), reverse=True)
    return candidates[0][1]


def materialize_dataset(
    workspace_root: Path,
    dataset_id: str,
    *,
    archive_path: Path | None = None,
    source_url: str | None = None,
    force: bool = False,
    keep_archive: bool = True,
) -> tuple[Path, Path]:
    if bool(archive_path) == bool(source_url):
        raise ValueError("Provide exactly one of `archive_path` or `source_url`.")

    workspace_root = workspace_root.resolve()
    archive_dir = workspace_root / "data" / "archives"
    datasets_root = workspace_root / "data" / "datasets"
    archive_dir.mkdir(parents=True, exist_ok=True)
    datasets_root.mkdir(parents=True, exist_ok=True)

    destination_archive = archive_dir / f"{dataset_id}.zip"
    dataset_dir = datasets_root / dataset_id
    staging_dir = datasets_root / f"{dataset_id}__extracting"

    if dataset_dir.exists():
        if not force:
            raise FileExistsError(f"Dataset directory already exists: {dataset_dir}")
        shutil.rmtree(dataset_dir)
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    if archive_path is not None:
        archive_source = archive_path.expanduser().resolve()
        if not archive_source.exists():
            raise FileNotFoundError(f"Dataset archive not found: {archive_source}")
        shutil.copy2(archive_source, destination_archive)
    else:
        download_archive(source_url=source_url or "", destination=destination_archive)

    staging_dir.mkdir(parents=True, exist_ok=True)
    extract_zip_portably(destination_archive, staging_dir)
    detected_root = _find_dataset_candidate(staging_dir)

    if detected_root == staging_dir:
        shutil.move(str(staging_dir), str(dataset_dir))
    else:
        shutil.copytree(detected_root, dataset_dir)
        shutil.rmtree(staging_dir)

    if not keep_archive and destination_archive.exists():
        destination_archive.unlink()

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "workspace_root": str(workspace_root),
        "dataset_id": dataset_id,
        "dataset_dir": str(dataset_dir),
        "archive_path": str(destination_archive),
        "archive_sha256": sha256_file(destination_archive) if destination_archive.exists() else None,
        "archive_size_bytes": destination_archive.stat().st_size if destination_archive.exists() else None,
        "source": {
            "type": "local_archive" if archive_path is not None else "remote_url",
            "value": str(archive_path) if archive_path is not None else source_url,
        },
        "file_counts": {
            "wav": len(list(dataset_dir.glob("audio_*.wav"))),
            "txt": len(list(dataset_dir.glob("audio_*.txt"))),
            "meta": len(list(dataset_dir.glob("audio_*.meta.txt"))),
        },
    }
    manifest_path = write_json(workspace_root / "artifacts" / "dataset" / f"{dataset_id}_manifest.json", manifest)
    return dataset_dir, manifest_path


def package_artifacts_bundle(
    output_zip: Path,
    *,
    benchmark_dir: Path,
    conversion_dir: Path | None = None,
    extra_paths: list[Path] | None = None,
) -> Path:
    output_zip = output_zip.expanduser().resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    included_items: list[dict[str, Any]] = []
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        def include_path(path: Path, prefix: str) -> None:
            resolved = path.expanduser().resolve()
            if resolved.is_file():
                archive_name = f"{prefix}/{resolved.name}"
                archive.write(resolved, arcname=archive_name)
                included_items.append(
                    {
                        "source_path": str(resolved),
                        "archive_name": archive_name,
                        "size_bytes": resolved.stat().st_size,
                    }
                )
                return
            for child in sorted(resolved.rglob("*")):
                if not child.is_file():
                    continue
                archive_name = f"{prefix}/{child.relative_to(resolved).as_posix()}"
                archive.write(child, arcname=archive_name)
                included_items.append(
                    {
                        "source_path": str(child),
                        "archive_name": archive_name,
                        "size_bytes": child.stat().st_size,
                    }
                )

        include_path(benchmark_dir, "benchmark")
        if conversion_dir is not None:
            include_path(conversion_dir, "conversion")
        for extra_path in extra_paths or []:
            include_path(extra_path, f"extra/{extra_path.name}")

        manifest_payload = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "benchmark_dir": str(benchmark_dir.resolve()),
            "conversion_dir": str(conversion_dir.resolve()) if conversion_dir is not None else None,
            "included_files": included_items,
        }
        archive.writestr("bundle_manifest.json", write_json_string(manifest_payload))
    return output_zip


def write_json_string(payload: dict[str, Any] | list[Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
