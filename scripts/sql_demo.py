"""Run sql_rag_chain on analytical questions and check each answer against hand-written SQL.

    uv run python scripts/sql_demo.py
"""

import sys

from medibot import sql_rag
from medibot.config import get_settings
from medibot.llm import GroqLLM

# (question, ground-truth SQL, values that must appear in the answer)
CASES = [
    (
        "How many claims were rejected?",
        "SELECT count(*) FROM claims WHERE status = 'rejected'",
    ),
    (
        "Which equipment category has the most open maintenance tickets?",
        "SELECT category, count(*) FROM maintenance_tickets WHERE status = 'open' GROUP BY category ORDER BY 2 DESC LIMIT 1",
    ),
    (
        "What is the total approved amount for cardiology claims?",
        "SELECT sum(approved_amount) FROM claims WHERE department = 'cardiology'",
    ),
    (
        "How many claims were escalated in December 2024?",
        "SELECT count(*) FROM claims WHERE status = 'escalated' AND submitted_date LIKE '2024-12%'",
    ),
    (
        "Which insurer has the highest number of pending claims?",
        "SELECT insurer, count(*) FROM claims WHERE status = 'pending' GROUP BY insurer ORDER BY 2 DESC LIMIT 1",
    ),
    (
        "How many maintenance tickets are still unresolved, and how many of those are escalated?",
        "SELECT count(*), sum(status = 'escalated') FROM maintenance_tickets WHERE status != 'resolved'",
    ),
]


class Tracing:
    """Wraps the LLM so the generated SQL is visible in the output."""

    def __init__(self, inner: GroqLLM) -> None:
        self.inner = inner
        self.last_sql = ""

    def complete(self, system: str, user: str, **kwargs) -> str:
        reply = self.inner.complete(system, user, **kwargs)
        if user.rstrip().endswith("SQL:"):
            self.last_sql = sql_rag.clean_sql(reply)
        return reply


def main() -> int:
    s = get_settings()
    llm = Tracing(GroqLLM(model=s.groq_model, api_key=s.groq_api_key))
    for question, truth_sql in CASES:
        _, truth_rows = sql_rag.run_sql(s.db_path, truth_sql)
        expected = [str(v) for v in truth_rows[0]]
        reply = sql_rag.sql_rag_chain(question, llm=llm, db_path=s.db_path)
        print(f"Q: {question}")
        print(f"   generated SQL: {llm.last_sql}")
        print(f"   answer:        {reply}")
        print(f"   ground truth:  {truth_rows[0]}   ({truth_sql})")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
