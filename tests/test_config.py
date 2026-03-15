"""Unit tests for config.py — environment validation and defaults."""

import os
import pytest


class TestValidateRequiredKeys:
    def test_passes_with_key_set(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
        # Re-import to pick up the patched env
        import importlib
        import config
        importlib.reload(config)
        # Should not raise
        config.validate_required_keys()

    def test_raises_when_key_missing(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        import importlib
        import config
        importlib.reload(config)
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            config.validate_required_keys()


class TestConfigDefaults:
    def test_default_llm_model(self, monkeypatch):
        monkeypatch.delenv("CALDRON_LLM_MODEL", raising=False)
        import importlib
        import config
        importlib.reload(config)
        assert config.LLM_MODEL == "gpt-3.5-turbo"

    def test_default_db_path(self, monkeypatch):
        monkeypatch.delenv("CALDRON_DB_PATH", raising=False)
        import importlib
        import config
        importlib.reload(config)
        assert "recipes" in config.DB_PATH

    def test_default_state_dir(self, monkeypatch):
        monkeypatch.delenv("CALDRON_STATE_DIR", raising=False)
        import importlib
        import config
        importlib.reload(config)
        assert config.STATE_DIR == "."

    def test_tracing_default_false(self, monkeypatch):
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
        import importlib
        import config
        importlib.reload(config)
        assert config.LANGCHAIN_TRACING_V2 is False
