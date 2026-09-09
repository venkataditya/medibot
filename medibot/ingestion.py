"""Parse PDFs and Markdown with Docling and cut them into heading-aware chunks."""

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medibot.rbac import COLLECTIONS, ROLES, UnknownCollectionError, roles_for

SOURCE_SUFFIXES = (".pdf", ".md")

# Every document in the corpus numbers its top-level sections: "1. Antimicrobials",
# "A. Patient Monitoring System", "SOP 4 - IV Cannula Insertion", "Q17. Who approves...".
# Sub-headings ("Fault codes", "Dosing") carry no number. "1.2 Documents required" is
# a Markdown sub-heading and is deliberately excluded (dotted, no trailing period).
TOP_LEVEL_HEADING = re.compile(r"^(\d+\.\s|[A-Z]\.\s|SOP \d+|Q\d+\.\s|Appendix\b)")

# docling item labels -> the four chunk types the spec asks for
HEADING_LABELS = {"section_header", "title"}
TABLE_LABEL = "table"
CODE_LABEL = "code"


class DataDirectoryError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class ChunkRecord:
    """One chunk as it will be stored in Qdrant.

    `text` is what gets embedded: the heading path followed by the body, so a row
    like "25 mg BD" still carries "Drug Formulary > Antimicrobials" with it.
    """

    text: str
    body: str
    source_document: str
    collection: str
    access_roles: list[str]
    section_title: str
    chunk_type: str
    heading_path: list[str] = field(default_factory=list)
    chunk_index: int = 0


def iter_source_files(data_dir: Path) -> Iterator[tuple[str, Path]]:
    """Yield (collection, path) for every PDF/Markdown file, folder name = collection."""
    if not data_dir.is_dir():
        raise DataDirectoryError(f"Data directory not found: {data_dir}")
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if folder.name not in COLLECTIONS:
            continue  # e.g. db/, which holds the SQLite file, not documents
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in SOURCE_SUFFIXES:
                yield folder.name, path


def chunk_type_for(labels: Iterable[str]) -> str:
    labels = [str(label) for label in labels]
    if TABLE_LABEL in labels:
        return "table"
    if labels and all(label == CODE_LABEL for label in labels):
        return "code"  # a whole code block; inline `code` spans inside prose stay "text"
    if labels and all(label in HEADING_LABELS for label in labels):
        return "heading"
    return "text"


def section_title_for(headings: list[str] | None, fallback: str) -> str:
    return headings[-1] if headings else fallback


def clean_heading_path(headings: list[str]) -> list[str]:
    """Drop entries that are sentences, not headings.

    The PDF hierarchy inference works from font style and occasionally promotes a
    bold numbered step ("4 Confirm a valid trace ... bedside.") to a heading. No real
    heading in the corpus ends with a full stop, so that is the tell.
    """
    return [h.strip() for h in headings if h.strip() and not h.strip().endswith(".")]


def chunk_record(
    chunk: Any, chunker: Any, source_document: str, collection: str, index: int
) -> ChunkRecord:
    if collection not in COLLECTIONS:
        raise UnknownCollectionError(f"'{source_document}' sits in unknown collection '{collection}'")
    headings = clean_heading_path(list(chunk.meta.headings or []))
    labels = [getattr(item, "label", "text") for item in (chunk.meta.doc_items or [])]
    return ChunkRecord(
        text=chunker.contextualize(chunk),
        body=chunk.text.strip(),
        source_document=source_document,
        collection=collection,
        access_roles=roles_for(collection),
        section_title=section_title_for(headings, Path(source_document).stem),
        chunk_type=chunk_type_for(labels),
        heading_path=headings,
        chunk_index=index,
    )


def records_from_chunks(
    chunks: Iterable[Any], chunker: Any, source_document: str, collection: str
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for chunk in chunks:
        if not chunk.text.strip():
            continue
        records.append(chunk_record(chunk, chunker, source_document, collection, len(records)))
    return records


def build_chunker(tokenizer_model: str, max_tokens: int) -> Any:
    """HybridChunker: split on document structure first, then cap by tokens.

    The tokenizer is the dense embedding model's own, so the cap is measured in the
    same tokens the embedder will see.
    """
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer

    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(tokenizer_model), max_tokens=max_tokens
    )
    return HybridChunker(tokenizer=tokenizer, merge_peers=True)


def build_converter() -> Any:
    """Docling converter for PDF + Markdown.

    OCR is off: every PDF in the corpus is born-digital with a real text layer, and
    OCR roughly triples parse time for no gain. Table structure stays on so dosage
    and billing tables come out as tables, not flattened text.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pdf_options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.MD],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)},
    )


def is_top_level_heading(text: str) -> bool:
    return bool(TOP_LEVEL_HEADING.match(text.strip()))


def assign_heading_levels(document: Any) -> None:
    """Give PDF headings a two-level hierarchy, in place.

    Docling detects PDF section headers but gives them all level 1, so the chunker
    would see "Fault codes" and forget which device it belongs to. Numbered headings
    become level 1 and the rest level 2, which the chunker turns into paths like
    "D. Portable X-Ray Unit - RadiPro MX-150 > Fault codes". Only the level changes;
    no item is added, removed or relabelled, so no text can go missing.
    Markdown already carries its levels from the # markers and is left alone.
    """
    for item, _ in document.iterate_items():
        if hasattr(item, "level"):
            item.level = 1 if is_top_level_heading(item.text) else 2


def chunk_document(path: Path, collection: str, chunker: Any, converter: Any) -> list[ChunkRecord]:
    document = converter.convert(str(path)).document
    if path.suffix.lower() == ".pdf":
        assign_heading_levels(document)
    return records_from_chunks(chunker.chunk(dl_doc=document), chunker, path.name, collection)


def chunk_corpus(data_dir: Path, chunker: Any, converter: Any) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for collection, path in iter_source_files(data_dir):
        records.extend(chunk_document(path, collection, chunker, converter))
    return records


__all__ = ["ChunkRecord", "ROLES", "chunk_corpus", "iter_source_files"]
