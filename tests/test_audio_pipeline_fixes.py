from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch

from application.services.model_inventory_service import ModelInventoryService
from infrastructure.adapters.asr import phowhisper_adapter as phowhisper_module
from infrastructure.adapters.asr.phowhisper_adapter import PhoWhisperAdapter
from infrastructure.adapters.diarization.speechbrain_adapter import SpeechBrainAdapter
from infrastructure.adapters.llm.llamacpp_adapter import LlamaCppAdapter
from infrastructure.adapters.vad.silero_adapter import SileroVADAdapter


def test_silero_load_audio_without_librosa(tmp_path: Path):
    sample_rate = 8000
    duration_sec = 0.25
    time_axis = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    mono_wave = 0.2 * np.sin(2 * np.pi * 220 * time_axis).astype(np.float32)
    stereo_wave = np.stack([mono_wave, mono_wave], axis=1)
    audio_path = tmp_path / "stereo_8k.wav"
    sf.write(audio_path, stereo_wave, sample_rate)

    waveform, loaded_sample_rate = SileroVADAdapter()._load_audio(str(audio_path))

    assert loaded_sample_rate == 16000
    assert waveform.ndim == 1
    assert waveform.dtype == np.float32
    assert len(waveform) > len(mono_wave)


def test_phowhisper_fuzzy_overlap_merge_removes_chunk_duplicates():
    adapter = PhoWhisperAdapter(device="cpu")
    chunk_segments = [
        {
            "start": 0.0,
            "end": 30.0,
            "text": "và đó là lý do tôi có mặt trong ngày hôm nay như tình trạng lệch trục cửi xe trâu khám phá bát tràng mọi người nghía thử.",
            "words": [],
        },
        {
            "start": 25.0,
            "end": 55.0,
            "text": "cưỡi xe trâu khám phá bát tràng mọi người nghía thử những người này đều khai biết lâm hùn với bảo làm chủ vốn.",
            "words": [],
        },
    ]

    merged_segments = adapter._merge_overlapping_segments(chunk_segments)

    assert len(merged_segments) == 2
    assert "cưỡi xe trâu khám phá bát tràng mọi người nghía thử" not in merged_segments[1]["text"].lower()
    assert merged_segments[1]["text"].lower().startswith("những người này đều khai biết")


def test_phowhisper_transcribe_chunk_passes_decoder_prompt_via_generation_config(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "phowhisper-safe"
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "preprocessor_config.json", "tokenizer.json", "model.safetensors"):
        (model_dir / filename).write_text("x", encoding="utf-8")

    captured: dict[str, object] = {}

    class FakeTokenizer:
        def get_decoder_prompt_ids(self, language: str, task: str):
            captured["decoder_request"] = (language, task)
            return [(1, 2), (2, 3)]

    class FakeBatch:
        def __init__(self):
            self.input_features = torch.zeros((1, 80, 8), dtype=torch.float32)
            self.attention_mask = torch.ones((1, 8), dtype=torch.long)

    class FakeProcessor:
        def __init__(self):
            self.tokenizer = FakeTokenizer()

        def __call__(self, *_args, **_kwargs):
            return FakeBatch()

        def batch_decode(self, _ids, skip_special_tokens=True):
            assert skip_special_tokens is True
            return ["xin chào"]

    class FakeModel:
        def __init__(self):
            self.generation_config = SimpleNamespace(forced_decoder_ids=None)

        def to(self, _device):
            return self

        def eval(self):
            return self

        def generate(self, _input_features, **kwargs):
            captured["generate_kwargs"] = kwargs
            return torch.tensor([[1, 2, 3]])

    monkeypatch.setattr(PhoWhisperAdapter, "MODEL_PATHS", [model_dir])
    monkeypatch.setattr(phowhisper_module, "WhisperProcessor", SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: FakeProcessor()))
    monkeypatch.setattr(phowhisper_module, "WhisperForConditionalGeneration", SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: FakeModel()))

    adapter = PhoWhisperAdapter(device="cpu")
    adapter._load_model()
    segments = adapter._transcribe_chunk(np.zeros(16000, dtype=np.float32), 0.0)

    assert captured["decoder_request"] == ("vi", "transcribe")
    generation_config = captured["generate_kwargs"]["generation_config"]
    assert generation_config is not adapter._model.generation_config
    assert generation_config.forced_decoder_ids == [(1, 2), (2, 3)]
    assert "forced_decoder_ids" not in captured["generate_kwargs"]
    assert segments[0]["text"] == "xin chào"


def test_phowhisper_transcribe_loads_audio_without_torchaudio_backend(monkeypatch, tmp_path: Path):
    sample_rate = 8000
    duration_sec = 0.25
    time_axis = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    mono_wave = 0.2 * np.sin(2 * np.pi * 220 * time_axis).astype(np.float32)
    stereo_wave = np.stack([mono_wave, mono_wave], axis=1)
    audio_path = tmp_path / "stereo_8k.wav"
    sf.write(audio_path, stereo_wave, sample_rate)

    model_dir = tmp_path / "phowhisper-safe"
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "preprocessor_config.json", "tokenizer.json", "model.safetensors"):
        (model_dir / filename).write_text("x", encoding="utf-8")

    class FakeTokenizer:
        def get_decoder_prompt_ids(self, language: str, task: str):
            assert (language, task) == ("vi", "transcribe")
            return [(1, 2)]

    class FakeBatch:
        def __init__(self):
            self.input_features = torch.zeros((1, 80, 8), dtype=torch.float32)
            self.attention_mask = torch.ones((1, 8), dtype=torch.long)

    class FakeProcessor:
        def __init__(self):
            self.tokenizer = FakeTokenizer()

        def __call__(self, audio_array, **_kwargs):
            assert len(audio_array) > len(mono_wave)
            return FakeBatch()

        def batch_decode(self, _ids, skip_special_tokens=True):
            assert skip_special_tokens is True
            return ["xin chào ngoại tuyến"]

    class FakeModel:
        def __init__(self):
            self.generation_config = SimpleNamespace(forced_decoder_ids=None)

        def to(self, _device):
            return self

        def eval(self):
            return self

        def generate(self, _input_features, **_kwargs):
            return torch.tensor([[1, 2, 3]])

    def fail_torchaudio_load(*_args, **_kwargs):
        raise AssertionError("PhoWhisperAdapter should not call torchaudio.load")

    monkeypatch.setattr(PhoWhisperAdapter, "MODEL_PATHS", [model_dir])
    monkeypatch.setattr(phowhisper_module, "WhisperProcessor", SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: FakeProcessor()))
    monkeypatch.setattr(phowhisper_module, "WhisperForConditionalGeneration", SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: FakeModel()))
    monkeypatch.setattr(phowhisper_module.torchaudio, "load", fail_torchaudio_load)

    transcript = PhoWhisperAdapter(device="cpu").transcribe(str(audio_path))

    assert transcript.text == "xin chào ngoại tuyến"
    assert transcript.segments[0]["text"] == "xin chào ngoại tuyến"


def test_speechbrain_load_audio_without_torchaudio_backend(tmp_path: Path):
    sample_rate = 8000
    duration_sec = 0.25
    time_axis = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    mono_wave = 0.2 * np.sin(2 * np.pi * 220 * time_axis).astype(np.float32)
    stereo_wave = np.stack([mono_wave, mono_wave], axis=1)
    audio_path = tmp_path / "speechbrain_stereo_8k.wav"
    sf.write(audio_path, stereo_wave, sample_rate)

    signal, loaded_sample_rate = SpeechBrainAdapter(use_vad=False)._load_audio(str(audio_path))

    assert loaded_sample_rate == 16000
    assert signal.ndim == 2
    assert signal.shape[0] == 1
    assert signal.dtype == torch.float32
    assert signal.shape[1] > len(mono_wave)


def test_model_inventory_uses_runtime_ready_signals(monkeypatch, tmp_path: Path):
    fake_whisper_v2 = tmp_path / "fw-large-v2"
    fake_whisper_v2.mkdir()
    fake_phowhisper = tmp_path / "phowhisper-safe"
    fake_phowhisper.mkdir()

    monkeypatch.setattr(ModelInventoryService, "_phowhisper_path", staticmethod(lambda: fake_phowhisper))
    monkeypatch.setattr(PhoWhisperAdapter, "runtime_ready", staticmethod(lambda: False))
    monkeypatch.setattr(phowhisper_module, "MODELS_DIR", tmp_path)
    monkeypatch.setattr("application.services.model_inventory_service.WhisperV2Adapter.get_local_model_path", lambda: fake_whisper_v2)
    monkeypatch.setattr(ModelInventoryService, "_whisper_v2_runtime_ready", staticmethod(lambda: True))
    monkeypatch.setattr(LlamaCppAdapter, "runtime_ready", staticmethod(lambda model_type="vistral": False))

    inventory = ModelInventoryService().get_inventory()

    assert inventory["capabilities"]["asr_engines"]["phowhisper"] is False
    assert inventory["capabilities"]["asr_engines"]["whisper-v2"] is True
    assert inventory["capabilities"]["features"]["apply_intel_summary"] is False
    phowhisper_item = next(item for item in inventory["items"] if item["model_id"] == "phowhisper")
    assert phowhisper_item["offline_ready"] is False
