"""Hybrid (dense + BM25) retrieval from Qdrant with the role filter applied in the query."""

from dataclasses import dataclass
from typing import Any, Literal

from qdrant_client import models

from medibot.index import DENSE, SPARSE, Vectors
from medibot.rbac import UnknownRoleError, access_filter

Mode = Literal["hybrid", "dense"]


@dataclass(frozen=True)
class Candidate:
    id: str
    score: float
    text: str
    body: str
    source_document: str
    collection: str
    section_title: str
    chunk_type: str
    heading_path: list[str]
    rerank_score: float | None = None


def _candidate(point: Any) -> Candidate:
    p = point.payload
    return Candidate(
        id=str(point.id),
        score=float(point.score),
        text=p["text"],
        body=p["body"],
        source_document=p["source_document"],
        collection=p["collection"],
        section_title=p["section_title"],
        chunk_type=p["chunk_type"],
        heading_path=list(p.get("heading_path", [])),
    )


def hybrid_search(client: Any, collection_name: str, query: Vectors, role: str, limit: int) -> list[Candidate]:
    """One Qdrant query: a dense prefetch and a BM25 prefetch, fused with reciprocal
    rank fusion inside Qdrant. The role filter sits on both prefetches and on the
    fused query, so a chunk outside the role's collections is never a candidate."""
    role_filter = access_filter(role)
    sparse = models.SparseVector(indices=query.sparse_indices, values=query.sparse_values)
    response = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(query=query.dense, using=DENSE, filter=role_filter, limit=limit * 2),
            models.Prefetch(query=sparse, using=SPARSE, filter=role_filter, limit=limit * 2),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=role_filter,
        limit=limit,
        with_payload=True,
    )
    return [_candidate(p) for p in response.points]


def dense_search(client: Any, collection_name: str, query: Vectors, role: str, limit: int) -> list[Candidate]:
    """Semantic-only search. Exists so the retrieval comparison can show what BM25 adds."""
    response = client.query_points(
        collection_name=collection_name,
        query=query.dense,
        using=DENSE,
        query_filter=access_filter(role),
        limit=limit,
        with_payload=True,
    )
    return [_candidate(p) for p in response.points]


def retrieve(
    client: Any, collection_name: str, embedder: Any, question: str, role: str, limit: int, mode: Mode = "hybrid"
) -> list[Candidate]:
    if mode not in ("hybrid", "dense"):
        raise ValueError(f"mode must be 'hybrid' or 'dense', got {mode!r}")
    access_filter(role)  # fail fast on an unknown role, before embedding anything
    vectors = embedder.embed_query(question)
    search = hybrid_search if mode == "hybrid" else dense_search
    return search(client, collection_name, vectors, role, limit)


__all__ = ["Candidate", "Mode", "UnknownRoleError", "dense_search", "hybrid_search", "retrieve"]
