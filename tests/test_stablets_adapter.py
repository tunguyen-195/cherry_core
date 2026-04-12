from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from infrastructure.adapters.asr import stablets_adapter as stablets_module


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


def test_stable_ts_adapter_uses_offline_faster_whisper_model(tmp_path: Path, monkeypatch):
    models_dir = tmp_path / "models"
    snapshot_dir = _create_ct2_cache(models_dir)

    monkeypatch.setattr(stablets_module, "MODELS_DIR", models_dir, raising=False)
    monkeypatch.setattr(stablets_module, "WHISPER_V2_PATH", models_dir / "whisper-large-v2", raising=False)
    monkeypatch.setattr(stablets_module.WhisperV2Adapter, "get_local_model_path", classmethod(lambda cls: snapshot_dir))

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

    class FakeStableModel:
        def transcribe(self, audio_path: str, **kwargs):
            captured["audio_path"] = audio_path
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                text=" Xin chào ổn định",
                segments=[
                    FakeSegment(
                        0.1,
                        0.9,
                        "Xin chào ổn định",
                        [FakeWord(" Xin", 0.1, 0.3, 0.91), FakeWord(" chào", 0.3, 0.6, 0.92)],
                    )
                ],
            )

    def fake_import_vendor_module(name: str):
        assert name == "stable_whisper"

        def fake_load_faster_whisper(model_size_or_path: str, **kwargs):
            captured["model_path"] = model_size_or_path
            captured["load_kwargs"] = kwargs
            return FakeStableModel()

        return SimpleNamespace(load_faster_whisper=fake_load_faster_whisper)

    monkeypatch.setattr(stablets_module, "import_vendor_module", fake_import_vendor_module)

    adapter = stablets_module.StableTsAdapter(device="cpu", cpu_threads=1)
    transcript = adapter.transcribe("sample.wav")

    assert captured["model_path"] == str(snapshot_dir)
    assert captured["load_kwargs"]["compute_type"] == "int8"
    assert captured["load_kwargs"]["local_files_only"] is True
    assert captured["load_kwargs"]["cpu_threads"] == 1
    assert captured["load_kwargs"]["num_workers"] == 1
    assert captured["audio_path"] == "sample.wav"
    assert captured["kwargs"]["suppress_silence"] is True
    assert transcript.text == "Xin chào ổn định"
    assert transcript.segments[0]["words"][0]["word"] == "Xin"
    assert transcript.metadata["backend"] == "stable-ts+faster-whisper"
    assert transcript.metadata["cpu_threads"] == 1
