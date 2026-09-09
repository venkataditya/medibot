import sys
from types import SimpleNamespace

from medibot import ingest
from medibot.ingestion import ChunkRecord


def record(collection: str, chunk_type: str, doc: str) -> ChunkRecord:
    return ChunkRecord(text="t", body="b", source_document=doc, collection=collection, access_roles=["admin"],
                       section_title="s", chunk_type=chunk_type)


def test_summary_counts_by_collection_type_and_document() -> None:
    text = ingest.summarise([record("clinical", "table", "a.pdf"), record("clinical", "text", "a.pdf"), record("general", "text", "b.md")])
    assert "3 chunks" in text
    assert "clinical=2" in text and "general=1" in text
    assert "table=1" in text and "text=2" in text
    assert "a.pdf" in text and "b.md" in text


def test_teardown_noise_is_silenced_only_at_interpreter_exit(monkeypatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(sys, "unraisablehook", lambda unraisable: calls.append(unraisable))
    ingest.silence_teardown_noise()
    sys.unraisablehook(SimpleNamespace(exc_value=ImportError("sys.meta_path is None")))
    assert calls == []  # swallowed, nothing printed
