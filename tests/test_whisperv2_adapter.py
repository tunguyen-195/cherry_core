from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from application.services.model_inventory_service import ModelInventoryService
from infrastructure.adapters.asr import whisperv2_adapter as whisperv2_module


def _create_ct2_cache(root: Path) -> Path:
    cache_root = root / "whisperx" / "asr" / "models--Systran--faster-whisper-large-v2"
    snapshot_id = "snapshot-1"
    snapshot_dir = cache_root / "snapshots" / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (cache_root / "refs").mkdir(parents=True, exist_ok=True)
    (cache_root / "refs" / "main").write_text(snapshot_id, encoding="utf-8")

    for filename in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"):
        (snapshot_dir / filename).write_text("x", encoding="utf-8")

    return snapshot_dir


def test_whisper_v2_resolves_faster_whisper_snapshot_cache(tmp_path: Path, monkeypatch):
    models_dir = tmp_path / "models"
    snapshot_dir = _create_ct2_cache(models_dir)

    monkeypatch.setattr(whisperv2_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(whisperv2_module, "WHISPER_V2_PATH", models_dir / "whisper-large-v2")

    assert whisperv2_module.WhisperV2Adapter.get_local_model_path() == snapshot_dir


def test_whisper_v2_transcribe_uses_faster_whisper_backend(tmp_path: Path, monkeypatch):
    models_dir = tmp_path / "models"
    snapshot_dir = _create_ct2_cache(models_dir)

    monkeypatch.setattr(whisperv2_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(whisperv2_module, "WHISPER_V2_PATH", models_dir / "whisper-large-v2")

    captured: dict[str, object] = {}

    class FakeWord:
        def __init__(self, word: str, start: float, end: float, probability: float):
            self.word = word
            self.start = start
            self.end = end
            self.probability = probability

    class FakeSegment:
        def __init__(self, start: float, end: float, text: str, words: list[FakeWord]):
            self.start = start
            self.end = end
            self.text = text
            self.words = words

    class FakeModel:
        def __init__(self, model_path: str, **kwargs):
            captured["model_path"] = model_path
            captured["load_kwargs"] = kwargs

        def transcribe(self, audio_path: str, **kwargs):
            captured["audio_path"] = audio_path
            captured["kwargs"] = kwargs
            segments = [
                FakeSegment(
                    0.0,
                    1.2,
                    "Quyên. Quyên. Quyên.",
                    [FakeWord(" Quyên", 0.0, 0.4, 0.98)],
                ),
                FakeSegment(
                    1.2,
                    2.1,
                    "đang nghe máy",
                    [FakeWord(" đang", 1.2, 1.5, 0.92), FakeWord(" nghe", 1.5, 1.8, 0.93)],
                ),
            ]
            info = SimpleNamespace(language="vi", language_probability=0.99, duration=2.1)
            return iter(segments), info

    monkeypatch.setattr(
        whisperv2_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(WhisperModel=FakeModel) if name == "faster_whisper" else __import__(name),
    )

    adapter = whisperv2_module.WhisperV2Adapter(use_vad=True, device="cpu", cpu_threads=1)
    transcript = adapter.transcribe("sample.wav")

    assert captured["model_path"] == str(snapshot_dir)
    assert captured["load_kwargs"]["device"] == "cpu"
    assert captured["load_kwargs"]["compute_type"] == "int8"
    assert captured["load_kwargs"]["cpu_threads"] == 1
    assert captured["load_kwargs"]["num_workers"] == 1
    assert captured["load_kwargs"]["local_files_only"] is True
    assert captured["audio_path"] == "sample.wav"
    assert captured["kwargs"]["condition_on_previous_text"] is False
    assert captured["kwargs"]["vad_filter"] is True
    assert captured["kwargs"]["word_timestamps"] is True
    assert captured["kwargs"]["hallucination_silence_threshold"] == 1.5
    assert transcript.text == "Quyên. đang nghe máy"
    assert transcript.segments[0]["text"] == "Quyên."
    assert transcript.segments[1]["words"][0]["word"] == "đang"
    assert transcript.metadata["backend"] == "faster-whisper"
    assert transcript.metadata["model_path"] == str(snapshot_dir)
    assert transcript.metadata["cpu_threads"] == 1


def test_model_inventory_reports_whisper_v2_ready_from_faster_whisper_cache(tmp_path: Path, monkeypatch):
    fake_model_dir = tmp_path / "fw-large-v2"
    fake_model_dir.mkdir()

    monkeypatch.setattr(
        whisperv2_module.WhisperV2Adapter,
        "get_local_model_path",
        lambda: fake_model_dir,
    )
    monkeypatch.setattr(
        ModelInventoryService,
        "_whisper_v2_runtime_ready",
        staticmethod(lambda: True),
    )

    inventory = ModelInventoryService().get_inventory()

    assert inventory["capabilities"]["asr_engines"]["whisper-v2"] is True
    whisper_v2_item = next(item for item in inventory["items"] if item["model_id"] == "whisper-v2")
    assert whisper_v2_item["offline_ready"] is True
    assert whisper_v2_item["path"] == str(fake_model_dir)
