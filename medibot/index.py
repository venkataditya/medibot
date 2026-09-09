"""Qdrant collection layout and the dense + BM25 indexing of chunk records."""

import uuid
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from medibot.ingestion import ChunkRecord

DENSE = "dense"
SPARSE = "sparse"
ID_NAMESPACE = uuid.UUID("6d1c3f2e-7b0a-4d5e-9c8b-2a1f0e9d8c7b")


@dataclass(frozen=True)
class Vectors:
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]

    def as_point_vector(self) -> dict[str, Any]:
        return {
            DENSE: self.dense,
            SPARSE: models.SparseVector(indices=self.sparse_indices, values=self.sparse_values),
        }


class FastEmbedder:
    """Dense (bge-small) and sparse (BM25) embeddings through fastembed's ONNX runtime."""

    def __init__(self, dense_model: str, sparse_model: str) -> None:
        from fastembed import SparseTextEmbedding, TextEmbedding

        self._dense = TextEmbedding(dense_model)
        self._sparse = SparseTextEmbedding(sparse_model)
        self.dense_dim = len(next(self._dense.embed(["probe"])))

    def embed_documents(self, texts: Sequence[str]) -> list[Vectors]:
        dense = self._dense.embed(list(texts))
        sparse = self._sparse.embed(list(texts))
        return [_vectors(d, s) for d, s in zip(dense, sparse, strict=True)]

    def embed_query(self, text: str) -> Vectors:
        dense = next(self._dense.query_embed(text))
        sparse = next(self._sparse.query_embed(text))
        return _vectors(dense, sparse)


def _vectors(dense: Any, sparse: Any) -> Vectors:
    return Vectors(
        dense=dense.tolist(),
        sparse_indices=sparse.indices.tolist(),
        sparse_values=sparse.values.tolist(),
    )


def client_kwargs(url: str | None, path: Path) -> dict[str, str]:
    return {"url": url} if url else {"path": str(path)}


class StoreLockedError(RuntimeError):
    pass


def make_client(url: str | None, path: Path) -> QdrantClient:
    try:
        return QdrantClient(**client_kwargs(url, path))
    except RuntimeError as e:
        if "already accessed" not in str(e):
            raise
        # the embedded store is single-process: the API server and the scripts cannot share it
        raise StoreLockedError(
            f"The Qdrant store at {path} is open in another process: stop the backend "
            "(or any running script) first, or set QDRANT_URL to use a Qdrant server."
        ) from e


def ensure_collection(client: QdrantClient, name: str, dense_dim: int) -> None:
    """(Re)create the collection: named dense + sparse vectors, so one query can
    search both. The IDF modifier is what makes the sparse side real BM25 rather
    than raw term counts."""
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config={DENSE: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    # Keyword indexes speed up the access_roles filter on a Qdrant server. The
    # embedded local mode has no indexes and warns about it; that is harmless.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for field in ("access_roles", "collection", "chunk_type"):
            client.create_payload_index(name, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD)


def payload_for(record: ChunkRecord) -> dict[str, Any]:
    return {
        "text": record.text,
        "body": record.body,
        "source_document": record.source_document,
        "collection": record.collection,
        "access_roles": list(record.access_roles),
        "section_title": record.section_title,
        "chunk_type": record.chunk_type,
        "heading_path": list(record.heading_path),
        "chunk_index": record.chunk_index,
    }


def point_id(record: ChunkRecord) -> str:
    # deterministic, so re-running ingest overwrites rather than duplicates
    return str(uuid.uuid5(ID_NAMESPACE, f"{record.source_document}#{record.chunk_index}"))


def _batches(items: Sequence[ChunkRecord], size: int) -> Iterator[Sequence[ChunkRecord]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def index_records(
    client: QdrantClient, name: str, records: Sequence[ChunkRecord], embedder: Any, batch_size: int = 32
) -> int:
    total = 0
    for batch in _batches(records, batch_size):
        vectors = embedder.embed_documents([r.text for r in batch])
        points = [
            models.PointStruct(id=point_id(r), vector=v.as_point_vector(), payload=payload_for(r))
            for r, v in zip(batch, vectors, strict=True)
        ]
        client.upsert(collection_name=name, points=points, wait=True)
        total += len(points)
    return total
