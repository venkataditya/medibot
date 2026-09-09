from types import SimpleNamespace

import pytest
from qdrant_client import models

from medibot import reranking, retrieval
from medibot.index import DENSE, SPARSE, Vectors
from medibot.rbac import access_filter

QUERY_VECTORS = Vectors(dense=[0.1, 0.2], sparse_indices=[3, 9], sparse_values=[1.0, 0.5])


def scored_point(pid: str, score: float, collection: str = "clinical") -> SimpleNamespace:
    return SimpleNamespace(
        id=pid,
        score=score,
        payload={
            "text": f"Heading\nbody of {pid}",
            "body": f"body of {pid}",
            "source_document": "drug_formulary.pdf",
            "collection": collection,
            "section_title": "Heading",
            "chunk_type": "text",
            "heading_path": ["Heading"],
        },
    )


class RecordingClient:
    def __init__(self, points: list[SimpleNamespace] | None = None) -> None:
        self.calls: list[dict] = []
        self.points = points or [scored_point("a", 0.9), scored_point("b", 0.8)]

    def query_points(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(points=self.points)


class FakeEmbedder:
    def embed_query(self, text: str) -> Vectors:
        return QUERY_VECTORS


def test_hybrid_search_is_one_query_with_dense_and_sparse_prefetch_fused_by_rrf() -> None:
    client = RecordingClient()
    retrieval.hybrid_search(client, "docs", QUERY_VECTORS, role="doctor", limit=10)

    assert len(client.calls) == 1, "dense and sparse must be fused inside Qdrant, not merged in Python"
    call = client.calls[0]
    assert call["collection_name"] == "docs"
    assert call["limit"] == 10
    assert call["query"] == models.FusionQuery(fusion=models.Fusion.RRF)
    using = {p.using for p in call["prefetch"]}
    assert using == {DENSE, SPARSE}


def test_hybrid_search_applies_the_role_filter_inside_every_prefetch_and_the_fused_query() -> None:
    client = RecordingClient()
    retrieval.hybrid_search(client, "docs", QUERY_VECTORS, role="nurse", limit=10)
    call = client.calls[0]
    expected = access_filter("nurse")
    assert call["query_filter"] == expected
    assert all(p.filter == expected for p in call["prefetch"])
    assert all(p.limit >= 10 for p in call["prefetch"])


def test_hybrid_search_sends_the_sparse_query_as_a_sparse_vector() -> None:
    client = RecordingClient()
    retrieval.hybrid_search(client, "docs", QUERY_VECTORS, role="admin", limit=5)
    sparse = next(p for p in client.calls[0]["prefetch"] if p.using == SPARSE)
    assert sparse.query == models.SparseVector(indices=[3, 9], values=[1.0, 0.5])
    dense = next(p for p in client.calls[0]["prefetch"] if p.using == DENSE)
    assert dense.query == [0.1, 0.2]


def test_dense_search_has_no_prefetch_but_still_carries_the_role_filter() -> None:
    client = RecordingClient()
    retrieval.dense_search(client, "docs", QUERY_VECTORS, role="technician", limit=4)
    call = client.calls[0]
    assert "prefetch" not in call
    assert call["using"] == DENSE
    assert call["query_filter"] == access_filter("technician")
    assert call["limit"] == 4


def test_search_rejects_unknown_role_before_touching_qdrant() -> None:
    client = RecordingClient()
    with pytest.raises(retrieval.UnknownRoleError):
        retrieval.hybrid_search(client, "docs", QUERY_VECTORS, role="ceo", limit=3)
    assert client.calls == []


def test_candidates_are_built_from_payload_with_scores() -> None:
    client = RecordingClient()
    found = retrieval.hybrid_search(client, "docs", QUERY_VECTORS, role="doctor", limit=2)
    assert [c.id for c in found] == ["a", "b"]
    assert found[0].score == 0.9
    assert found[0].source_document == "drug_formulary.pdf"
    assert found[0].section_title == "Heading"
    assert found[0].text == "Heading\nbody of a"
    assert found[0].rerank_score is None


def test_retrieve_embeds_the_question_and_dispatches_on_mode() -> None:
    client = RecordingClient()
    retrieval.retrieve(client, "docs", FakeEmbedder(), "meropenem dose", role="doctor", limit=7, mode="dense")
    retrieval.retrieve(client, "docs", FakeEmbedder(), "meropenem dose", role="doctor", limit=7, mode="hybrid")
    assert "prefetch" not in client.calls[0] and "prefetch" in client.calls[1]
    with pytest.raises(ValueError):
        retrieval.retrieve(client, "docs", FakeEmbedder(), "q", role="doctor", limit=7, mode="magic")


class FakeCrossEncoder:
    """Scores by how many query words appear in the document."""

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        words = set(query.lower().split())
        return [sum(w in doc.lower() for w in words) / len(words) for doc in documents]


def candidate(pid: str, text: str, score: float) -> retrieval.Candidate:
    return retrieval.Candidate(
        id=pid, score=score, text=text, body=text, source_document="d.pdf", collection="general",
        section_title="s", chunk_type="text", heading_path=["s"],
    )


def test_rerank_reorders_by_joint_query_document_score_and_keeps_top_n() -> None:
    candidates = [
        candidate("first", "unrelated leave policy text", 0.9),
        candidate("second", "vancomycin trough monitoring", 0.8),
        candidate("third", "vancomycin 15-20 mg/kg Q12H trough 15-20 mg/L TDM", 0.7),
        candidate("fourth", "aspirin", 0.6),
    ]
    top = reranking.rerank(FakeCrossEncoder(), "vancomycin trough", candidates, top_n=2)
    assert [c.id for c in top] == ["second", "third"] or [c.id for c in top] == ["third", "second"]
    assert all(c.rerank_score is not None for c in top)
    assert top[0].rerank_score >= top[1].rerank_score
    # the originals are untouched
    assert all(c.rerank_score is None for c in candidates)


def test_rerank_of_nothing_is_nothing() -> None:
    assert reranking.rerank(FakeCrossEncoder(), "q", [], top_n=3) == []
