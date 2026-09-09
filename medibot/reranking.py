"""Cross-encoder reranking: score (question, chunk) pairs jointly and keep the best few."""

from dataclasses import replace
from typing import Any

from medibot.retrieval import Candidate


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name)

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, documents)]


def rerank(reranker: Any, question: str, candidates: list[Candidate], top_n: int) -> list[Candidate]:
    """Unlike the bi-encoder used for retrieval, the cross-encoder reads the question
    and each chunk together, so it can tell "vancomycin trough" from a chunk that
    merely mentions vancomycin. Only the top_n survivors reach the LLM."""
    if not candidates:
        return []
    scores = reranker.rerank(question, [c.text for c in candidates])
    scored = [replace(c, rerank_score=s) for c, s in zip(candidates, scores, strict=True)]
    scored.sort(key=lambda c: c.rerank_score, reverse=True)
    return scored[:top_n]
