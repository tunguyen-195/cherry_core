from __future__ import annotations

from presentation.web import app as web_app_module


def test_disable_windows_quick_edit_mode_returns_false_off_windows(monkeypatch):
    monkeypatch.setattr(web_app_module.os, "name", "posix")

    assert web_app_module._disable_windows_quick_edit_mode() is False


def test_configure_runtime_environment_sets_non_interactive_defaults(monkeypatch):
    monkeypatch.setattr(web_app_module.os, "name", "posix")
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.delenv("TRANSFORMERS_NO_ADVISORY_WARNINGS", raising=False)

    web_app_module._configure_runtime_environment()

    assert web_app_module.os.environ["TOKENIZERS_PARALLELISM"] == "false"
    assert web_app_module.os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert web_app_module.os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] == "1"
