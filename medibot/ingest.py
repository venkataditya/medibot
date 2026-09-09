"""Command line: parse the corpus, embed it and load it into Qdrant.

    uv run python -m medibot.ingest            # full run
    uv run python -m medibot.ingest --dry-run  # parse and chunk only, print a summary
"""

import argparse
import logging
import time
from collections import Counter

from medibot import index, ingestion
from medibot.config import get_settings


def summarise(records: list[ingestion.ChunkRecord]) -> str:
    by_collection = Counter(r.collection for r in records)
    by_type = Counter(r.chunk_type for r in records)
    by_doc = Counter(r.source_document for r in records)
    lines = [f"{len(records)} chunks"]
    lines.append("  by collection: " + ", ".join(f"{k}={v}" for k, v in sorted(by_collection.items())))
    lines.append("  by chunk_type: " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
    lines.append("  by document:")
    lines.extend(f"    {doc:32} {n:3}" for doc, n in sorted(by_doc.items()))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="parse and chunk, but do not touch Qdrant")
    parser.add_argument("--show", type=int, default=2, help="print this many sample chunks")
    args = parser.parse_args(argv)
    logging.disable(logging.WARNING)  # docling and onnx are chatty; errors still surface as exceptions

    settings = get_settings()
    started = time.time()
    print(f"Parsing {settings.data_dir}/ with Docling ...")
    chunker = ingestion.build_chunker(settings.dense_model, settings.chunk_max_tokens)
    records = ingestion.chunk_corpus(settings.data_dir, chunker, ingestion.build_converter())
    print(summarise(records))
    for record in records[: args.show]:
        print(f"\n--- sample [{record.collection}/{record.source_document} #{record.chunk_index}, {record.chunk_type}] ---")
        print(record.text[:400])
    print(f"\nparsed in {time.time() - started:.0f}s")
    if args.dry_run:
        return 0

    print(f"\nEmbedding with {settings.dense_model} + {settings.sparse_model} ...")
    embedder = index.FastEmbedder(settings.dense_model, settings.sparse_model)
    client = index.make_client(settings.qdrant_url, settings.qdrant_path)
    index.ensure_collection(client, settings.collection_name, embedder.dense_dim)
    total = index.index_records(client, settings.collection_name, records, embedder)
    where = settings.qdrant_url or settings.qdrant_path
    print(f"Indexed {total} chunks into '{settings.collection_name}' at {where} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
