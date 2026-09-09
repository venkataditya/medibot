import pytest

from medibot import answer
from medibot.retrieval import Candidate


def chunk(pid: str, doc: str, section: str, collection: str = "clinical", text: str = "body") -> Candidate:
    return Candidate(
        id=pid, score=1.0, text=text, body=text, source_document=doc, collection=collection,
        section_title=section, chunk_type="text", heading_path=["Top", section], rerank_score=0.5,
    )


class FakeLLM:
    def __init__(self, reply: str = "Meropenem is 1 g Q8H [1].") -> None:
        self.reply = reply
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, **kwargs) -> str:
        self.prompts.append((system, user))
        return self.reply


def test_context_numbers_each_chunk_and_names_document_and_heading_path() -> None:
    ctx = answer.build_context([chunk("a", "drug_formulary.pdf", "1. Antimicrobials", text="Meropenem 1 g Q8H")])
    assert "[1]" in ctx
    assert "drug_formulary.pdf" in ctx
    assert "Top > 1. Antimicrobials" in ctx
    assert "Meropenem 1 g Q8H" in ctx


def test_sources_are_deduplicated_and_keep_rank_order() -> None:
    chunks = [
        chunk("a", "drug_formulary.pdf", "1. Antimicrobials"),
        chunk("b", "treatment_protocols.pdf", "Antimicrobial therapy"),
        chunk("c", "drug_formulary.pdf", "1. Antimicrobials"),
    ]
    assert answer.sources_from(chunks) == [
        answer.Source("drug_formulary.pdf", "1. Antimicrobials", "clinical"),
        answer.Source("treatment_protocols.pdf", "Antimicrobial therapy", "clinical"),
    ]


def test_answer_from_chunks_grounds_the_prompt_in_the_context_only() -> None:
    llm = FakeLLM()
    reply = answer.answer_from_chunks(llm, "Meropenem dose?", [chunk("a", "drug_formulary.pdf", "1. Antimicrobials", text="Meropenem 1 g Q8H")])
    assert reply == "Meropenem is 1 g Q8H [1]."
    system, user = llm.prompts[0]
    assert "Meropenem 1 g Q8H" in user and "Meropenem dose?" in user
    assert "only" in system.lower()


def test_no_chunks_means_a_fixed_answer_and_no_llm_call() -> None:
    llm = FakeLLM()
    reply = answer.answer_from_chunks(llm, "anything", [])
    assert reply == answer.NO_CONTEXT_ANSWER
    assert llm.prompts == []


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("Dose is 1 g Q8H【1】.", "Dose is 1 g Q8H [1]."),
        ("Within 6 hours【1†L4-L5】【3†L3-L4】", "Within 6 hours [1] [3]"),
        ("Already fine [2].", "Already fine [2]."),
        ("Two in a row【1】【2】", "Two in a row [1] [2]"),
    ],
)
def test_citation_glyphs_are_normalised_to_plain_brackets(raw: str, clean: str) -> None:
    assert answer.tidy_citations(raw) == clean


def test_answer_from_chunks_tidies_the_model_output() -> None:
    llm = FakeLLM("1 g Q8H【1†L2-L3】")
    assert answer.answer_from_chunks(llm, "q", [chunk("a", "d.pdf", "s")]) == "1 g Q8H [1]"
