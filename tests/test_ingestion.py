from pathlib import Path
from types import SimpleNamespace

import pytest

from medibot import ingestion


def fake_chunk(text: str, headings: list[str] | None, labels: list[str]) -> SimpleNamespace:
    items = [SimpleNamespace(label=label) for label in labels]
    return SimpleNamespace(text=text, meta=SimpleNamespace(headings=headings, doc_items=items))


class FakeChunker:
    def contextualize(self, chunk: SimpleNamespace) -> str:
        return "\n".join((chunk.meta.headings or []) + [chunk.text])


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (["text"], "text"),
        (["paragraph", "list_item"], "text"),
        (["table"], "table"),
        (["text", "table"], "table"),
        (["code"], "code"),
        (["code", "code"], "code"),
        (["text", "code"], "text"),
        (["section_header"], "heading"),
        (["title"], "heading"),
        (["section_header", "text"], "text"),
        ([], "text"),
    ],
)
def test_chunk_type_from_doc_item_labels(labels: list[str], expected: str) -> None:
    assert ingestion.chunk_type_for(labels) == expected


def test_section_title_is_the_deepest_heading() -> None:
    assert ingestion.section_title_for(["1. Antimicrobials", "Dosing"], "drug_formulary") == "Dosing"


def test_section_title_falls_back_to_document_name_when_no_headings() -> None:
    assert ingestion.section_title_for(None, "drug_formulary") == "drug_formulary"
    assert ingestion.section_title_for([], "drug_formulary") == "drug_formulary"


def test_chunk_record_carries_full_metadata_schema_and_heading_context() -> None:
    chunk = fake_chunk("Meropenem 1 g Q8H", ["1. Antimicrobials"], ["table"])
    record = ingestion.chunk_record(
        chunk, FakeChunker(), source_document="drug_formulary.pdf", collection="clinical", index=7
    )
    assert record.source_document == "drug_formulary.pdf"
    assert record.collection == "clinical"
    assert record.access_roles == ["doctor", "admin"]
    assert record.section_title == "1. Antimicrobials"
    assert record.chunk_type == "table"
    assert record.heading_path == ["1. Antimicrobials"]
    assert record.chunk_index == 7
    assert record.body == "Meropenem 1 g Q8H"
    # what gets embedded is heading + body, not the bare body
    assert record.text.startswith("1. Antimicrobials")
    assert record.text.endswith("Meropenem 1 g Q8H")


def test_chunk_record_rejects_unknown_collection() -> None:
    chunk = fake_chunk("x", None, ["text"])
    with pytest.raises(ingestion.UnknownCollectionError):
        ingestion.chunk_record(chunk, FakeChunker(), "x.pdf", "finance", 0)


def test_iter_source_files_yields_pdf_and_md_per_collection_folder(tmp_path: Path) -> None:
    (tmp_path / "clinical").mkdir()
    (tmp_path / "clinical" / "drug_formulary.pdf").write_bytes(b"%PDF")
    (tmp_path / "billing").mkdir()
    (tmp_path / "billing" / "guide.md").write_text("# hi")
    (tmp_path / "billing" / "notes.txt").write_text("skip me")
    (tmp_path / "db").mkdir()
    (tmp_path / "db" / "mediassist.db").write_bytes(b"")

    found = sorted(ingestion.iter_source_files(tmp_path))
    assert found == [
        ("billing", tmp_path / "billing" / "guide.md"),
        ("clinical", tmp_path / "clinical" / "drug_formulary.pdf"),
    ]


def test_iter_source_files_fails_loudly_on_missing_data_dir(tmp_path: Path) -> None:
    with pytest.raises(ingestion.DataDirectoryError):
        list(ingestion.iter_source_files(tmp_path / "nope"))


def test_records_from_document_numbers_chunks_and_skips_empty_ones() -> None:
    chunks = [
        fake_chunk("Intro", ["Title"], ["text"]),
        fake_chunk("   ", ["Title"], ["text"]),
        fake_chunk("Dose table", ["Title", "Dosing"], ["table"]),
    ]
    records = ingestion.records_from_chunks(chunks, FakeChunker(), "a.pdf", "general")
    assert [r.chunk_index for r in records] == [0, 1]
    assert [r.section_title for r in records] == ["Title", "Dosing"]
    assert all(r.access_roles == list(ingestion.ROLES) for r in records)


class FakeConverter:
    """Pretends to be Docling's DocumentConverter: one chunk per file, named after it."""

    def __init__(self, chunks_by_name: dict[str, list[SimpleNamespace]]) -> None:
        self.chunks_by_name = chunks_by_name

    def convert(self, source: str) -> SimpleNamespace:
        return SimpleNamespace(document=Path(source).name)


class FakeStructuredChunker(FakeChunker):
    def __init__(self, chunks_by_name: dict[str, list[SimpleNamespace]]) -> None:
        self.chunks_by_name = chunks_by_name

    def chunk(self, dl_doc: str) -> list[SimpleNamespace]:
        return self.chunks_by_name[dl_doc]


def test_chunk_corpus_walks_every_collection_and_tags_records(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ingestion, "assign_heading_levels", lambda document: None)
    (tmp_path / "general").mkdir()
    (tmp_path / "general" / "faq.pdf").write_bytes(b"%PDF")
    (tmp_path / "billing").mkdir()
    (tmp_path / "billing" / "guide.md").write_text("# Guide")
    chunks = {
        "faq.pdf": [fake_chunk("Salary is paid monthly", ["Payroll"], ["text"])],
        "guide.md": [fake_chunk("Within 6 hours", ["1. Cashless", "1.1 Timeline"], ["table"])],
    }
    chunker = FakeStructuredChunker(chunks)

    records = ingestion.chunk_corpus(tmp_path, chunker, FakeConverter(chunks))

    assert [(r.collection, r.source_document, r.section_title) for r in records] == [
        ("billing", "guide.md", "1.1 Timeline"),
        ("general", "faq.pdf", "Payroll"),
    ]
    assert records[0].access_roles == ["billing_executive", "admin"]
    assert records[0].chunk_type == "table"


def test_converter_skips_ocr_but_keeps_table_structure() -> None:
    from docling.datamodel.base_models import InputFormat

    converter = ingestion.build_converter()
    pdf_options = converter.format_to_options[InputFormat.PDF].pipeline_options
    assert pdf_options.do_ocr is False
    assert pdf_options.do_table_structure is True
    assert InputFormat.MD in converter.allowed_formats


def test_clean_heading_path_drops_list_steps_promoted_to_headings() -> None:
    # the PDF hierarchy inference sometimes treats a bold numbered step as a heading;
    # real headings in this corpus never end with a full stop
    path = [
        "Equipment Manual",
        "A. Patient Monitoring System",
        "4 Confirm a valid trace on each parameter before leaving the bedside.",
        "Fault codes",
    ]
    assert ingestion.clean_heading_path(path) == [
        "Equipment Manual",
        "A. Patient Monitoring System",
        "Fault codes",
    ]


def test_clean_heading_path_keeps_ordinary_headings_and_strips_whitespace() -> None:
    assert ingestion.clean_heading_path([" 1. Antimicrobials ", "Q1. When is salary credited?"]) == [
        "1. Antimicrobials",
        "Q1. When is salary credited?",
    ]


def test_chunk_record_uses_cleaned_heading_path() -> None:
    chunk = fake_chunk("E-12 internal sensor failure", ["Manual", "3 Connect leads first.", "Fault codes"], ["table"])
    record = ingestion.chunk_record(chunk, FakeChunker(), "equipment_manual.pdf", "equipment", 0)
    assert record.heading_path == ["Manual", "Fault codes"]
    assert record.section_title == "Fault codes"


def test_pdf_conversion_assigns_heading_levels_but_markdown_does_not(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    class Result:
        document = "doc"

    class Converter:
        def convert(self, source: str) -> Result:
            return Result()

    chunker = FakeStructuredChunker({"doc": [fake_chunk("body", ["H"], ["text"])]})
    monkeypatch.setattr(ingestion, "assign_heading_levels", lambda document: calls.append("pdf"))

    ingestion.chunk_document(tmp_path / "a.pdf", "general", chunker, Converter())
    ingestion.chunk_document(tmp_path / "b.md", "general", chunker, Converter())
    assert calls == ["pdf"]


@pytest.mark.parametrize(
    ("heading", "top_level"),
    [
        ("1. Antimicrobials", True),
        ("11. IV Fluids Quick Reference", True),
        ("A. Patient Monitoring System - MediAssist BM-500", True),
        ("SOP 4 - IV Cannula Insertion", True),
        ("Appendix: Key Drug Interactions & Cautions", True),
        ("Q17. Who approves leave for doctors in the ICU department?", True),
        ("Fault codes", False),
        ("Frequency", False),
        ("Pharmacological management", False),
        ("1.2 Documents required", False),
    ],
)
def test_top_level_headings_are_the_numbered_ones(heading: str, top_level: bool) -> None:
    assert ingestion.is_top_level_heading(heading) is top_level


def test_assign_heading_levels_nests_unnumbered_headings_under_numbered_ones() -> None:
    from docling_core.types.doc import DoclingDocument, DocItemLabel

    doc = DoclingDocument(name="equipment_manual")
    doc.add_title(text="Equipment Operation & Maintenance Manual")
    doc.add_heading(text="Conventions", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="Asset tags follow EQ-CAMPUS-XXXX")
    doc.add_heading(text="A. Patient Monitoring System", level=1)
    doc.add_heading(text="Fault codes", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="E-12 internal sensor failure")
    doc.add_heading(text="B. Infusion Pump", level=1)
    doc.add_heading(text="Fault codes", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="F-12 drug library update")

    ingestion.assign_heading_levels(doc)

    levels = [(item.text, item.level) for item, _ in doc.iterate_items() if hasattr(item, "level")]
    assert levels == [
        ("Conventions", 2),
        ("A. Patient Monitoring System", 1),
        ("Fault codes", 2),
        ("B. Infusion Pump", 1),
        ("Fault codes", 2),
    ]
    # the text of every item survives untouched: levels change, content never does
    texts = [item.text for item, _ in doc.iterate_items() if item.label == DocItemLabel.TEXT]
    assert texts == ["Asset tags follow EQ-CAMPUS-XXXX", "E-12 internal sensor failure", "F-12 drug library update"]
