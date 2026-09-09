"""Dense-only vs hybrid vs hybrid + rerank on keyword-heavy medical queries.

Each query names the chunk that should come back (document + a phrase in its body).
The table shows at which rank that chunk appears under each strategy.

    uv run python scripts/compare_retrieval.py
"""

import logging
import sys

from medibot import index, reranking, retrieval
from medibot.config import get_settings

# (question, expected source_document, phrase that must be in the expected chunk body)
CASES = [
    # bare identifiers, the way staff actually type them
    ("F-09", "equipment_manual.pdf", "F-09, Meaning"),
    ("E-07 SterilPro", "equipment_manual.pdf", "E-07, Meaning = Temperature sensor fault"),
    ("N18.3", "billing_codes.pdf", "N18.3"),
    ("EXCL-08", "billing_codes.pdf", "EXCL-08"),
    ("PROC-OPTH-01", "billing_codes.pdf", "PROC-OPTH-01"),
    ("MX150 serial number format", "equipment_manual.pdf", "MX150-YYYY-NNNNN"),
    ("1800-425-2255", "billing_codes.pdf", "1800-425-2255"),
    ("VIP score", "icu_nursing_procedures.pdf", "VIP"),
    # natural questions with an exact term inside
    ("What does fault code F-09 mean and what should I do?", "equipment_manual.pdf", "F-09, Meaning"),
    ("E-12 fault on the BM-500 monitor", "equipment_manual.pdf", "E-12, Meaning"),
    ("DriveFlow IP-200 occlusion pressure alarm for venous lines", "equipment_manual.pdf", "200 mmHg"),
    ("SterilPro 3000 Bowie-Dick test frequency", "equipment_manual.pdf", "Bowie-Dick"),
    ("Package rate for ICD-10 I21.4", "billing_codes.pdf", "I21.4"),
    ("PROC-CARD-02 package rate", "billing_codes.pdf", "PROC-CARD-02"),
    ("What does rejection code EXCL-04 mean?", "billing_codes.pdf", "EXCL-04"),
    ("Star Health pre-auth SLA", "claim_submission_guide.md", "Star Health"),
    ("Meropenem standard dose and tier", "drug_formulary.pdf", "Meropenem, Class"),
    ("Vancomycin trough target", "drug_formulary.pdf", "trough 15"),
    ("Metformin dose with eGFR under 30", "drug_formulary.pdf", "Metformin, eGFR"),
    ("CURB-65 score of 3, where should the patient go?", "treatment_protocols.pdf", "Score"),
    ("HbA1c threshold for endocrinology referral", "treatment_protocols.pdf", "HbA1c > 10%"),
    ("IV cannula gauge for a paediatric patient under 5 kg", "icu_nursing_procedures.pdf", "24G"),
    ("Braden Scale cut-off for pressure injury risk", "icu_nursing_procedures.pdf", "Braden"),
    ("Which bin does a used IV set go in?", "infection_control.pdf", "Red, Waste Type"),
    ("What does Code Pink mean?", "staff_handbook.pdf", "Code Pink"),
    ("Is there a gratuity benefit?", "general_faqs.pdf", "Gratuity"),
    ("I stopped coming to work two weeks ago without telling anyone, what happens?", "leave_policy.pdf", "10 or more consecutive days"),
    ("Can I accept a present from a patient?", "code_of_conduct.pdf", "Gifts"),
]


def rank_of(candidates: list[retrieval.Candidate], document: str, phrase: str) -> int | None:
    for position, c in enumerate(candidates, start=1):
        if c.source_document == document and phrase in c.body:
            return position
    return None


def fmt(rank: int | None) -> str:
    return "miss" if rank is None else str(rank)


def main() -> int:
    logging.disable(logging.WARNING)
    s = get_settings()
    client = index.make_client(s.qdrant_url, s.qdrant_path)
    embedder = index.FastEmbedder(s.dense_model, s.sparse_model)
    reranker = reranking.CrossEncoderReranker(s.rerank_model)
    role = "admin"  # sees every collection, so the comparison is about retrieval, not access

    print(f"{'question':70} {'dense@3':>7} {'dense@10':>8} {'hybrid@10':>9} {'rerank@3':>8}")
    print("-" * 107)
    hits = {"dense3": [], "dense": [], "hybrid": [], "rerank": []}
    for question, document, phrase in CASES:
        vectors = embedder.embed_query(question)
        dense = retrieval.dense_search(client, s.collection_name, vectors, role, s.candidate_k)
        hybrid = retrieval.hybrid_search(client, s.collection_name, vectors, role, s.candidate_k)
        top = reranking.rerank(reranker, question, hybrid, s.top_k)
        ranks = (
            rank_of(dense[: s.top_k], document, phrase),
            rank_of(dense, document, phrase),
            rank_of(hybrid, document, phrase),
            rank_of(top, document, phrase),
        )
        for key, rank in zip(hits, ranks, strict=True):
            hits[key].append(rank)
        print(f"{question[:70]:70} {fmt(ranks[0]):>7} {fmt(ranks[1]):>8} {fmt(ranks[2]):>9} {fmt(ranks[3]):>8}")

    n = len(CASES)
    print("-" * 107)
    for key, label in (
        ("dense3", "dense-only @3"),
        ("dense", "dense-only @10"),
        ("hybrid", "hybrid @10"),
        ("rerank", "hybrid + rerank @3"),
    ):
        ranks = hits[key]
        found = sum(r is not None for r in ranks)
        at1 = sum(r == 1 for r in ranks)
        mrr = sum(1 / r for r in ranks if r) / n
        print(f"{label:20} found {found:2}/{n}   rank-1 {at1:2}/{n}   MRR {mrr:.2f}")

    print("\nReranker scores for one query, to show why the reorder matters:")
    question, document, phrase = next(c for c in CASES if c[0].startswith("Which bin"))
    vectors = embedder.embed_query(question)
    hybrid = retrieval.hybrid_search(client, s.collection_name, vectors, role, s.candidate_k)
    for c in reranking.rerank(reranker, question, hybrid, len(hybrid)):
        was = hybrid.index(next(h for h in hybrid if h.id == c.id)) + 1
        print(f"  hybrid rank {was:2} -> score {c.rerank_score:6.2f}  {c.source_document} / {c.section_title}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
