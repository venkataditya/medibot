"""The /chat flow: route, enforce role access, then hybrid RAG or SQL RAG."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from medibot import answer, reranking, retrieval, router
from medibot.rbac import (
    UnknownRoleError,
    can_use_sql,
    collections_for,
    refusal_message,
    sql_refusal_message,
)

RetrievalType = Literal["hybrid_rag", "sql_rag"]


@dataclass(frozen=True)
class ChatResult:
    answer: str
    retrieval_type: RetrievalType
    role: str
    sources: list[answer.Source] = field(default_factory=list)
    access_denied: bool = False


class MediBot:
    """Holds the loaded models and store; `chat` is the whole request flow.

    The `*_fn` hooks exist so the flow can be tested without models or an index.
    """

    def __init__(
        self,
        client: Any,
        collection_name: str,
        embedder: Any,
        reranker: Any,
        llm: Any,
        sql_chain: Callable[[str], str],
        candidate_k: int,
        top_k: int,
        route_fn: Callable[..., router.Route] = router.route_question,
        retrieve_fn: Callable[..., list[retrieval.Candidate]] = retrieval.retrieve,
        rerank_fn: Callable[..., list[retrieval.Candidate]] = reranking.rerank,
        answer_fn: Callable[..., str] = answer.answer_from_chunks,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.embedder = embedder
        self.reranker = reranker
        self.llm = llm
        self.sql_chain = sql_chain
        self.candidate_k = candidate_k
        self.top_k = top_k
        self._route = route_fn
        self._retrieve = retrieve_fn
        self._rerank = rerank_fn
        self._answer = answer_fn

    def chat(self, question: str, role: str) -> ChatResult:
        allowed = collections_for(role)  # raises UnknownRoleError
        if not question or not question.strip():
            raise ValueError("Question is empty.")
        question = question.strip()

        route = self._route(self.llm, question)
        if route.kind == "sql":
            if not can_use_sql(role):
                return ChatResult(sql_refusal_message(role), "sql_rag", role, access_denied=True)
            return ChatResult(self.sql_chain(question), "sql_rag", role)

        # The router only shapes the *message*; the Qdrant filter is what actually
        # keeps restricted chunks out, whatever the router says.
        if route.collection != "unknown" and route.collection not in allowed:
            return ChatResult(refusal_message(role, route.collection), "hybrid_rag", role, access_denied=True)

        candidates = self._retrieve(
            self.client, self.collection_name, self.embedder, question, role=role, limit=self.candidate_k, mode="hybrid"
        )
        top = self._rerank(self.reranker, question, candidates, top_n=self.top_k)
        return ChatResult(self._answer(self.llm, question, top), "hybrid_rag", role, sources=answer.sources_from(top))


__all__ = ["ChatResult", "MediBot", "RetrievalType", "UnknownRoleError"]
