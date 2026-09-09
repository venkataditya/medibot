from pathlib import Path

from medibot.config import Settings


def test_defaults_point_at_local_qdrant_and_bundled_data(monkeypatch) -> None:
    monkeypatch.delenv("QDRANT_URL", raising=False)
    settings = Settings(_env_file=None, groq_api_key="x")
    assert settings.qdrant_url is None
    assert settings.qdrant_path == Path(".qdrant")
    assert settings.data_dir == Path("data")
    assert settings.db_path == Path("data/db/mediassist.db")
    assert settings.candidate_k > settings.top_k


def test_env_overrides_are_read(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("GROQ_MODEL", "some-model")
    settings = Settings(_env_file=None, groq_api_key="x")
    assert settings.qdrant_url == "http://qdrant:6333"
    assert settings.groq_model == "some-model"
