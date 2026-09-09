"""SQL RAG over mediassist.db: question -> SQL -> rows -> natural-language answer."""

import re
import sqlite3
from pathlib import Path
from typing import Any

from medibot.config import get_settings


class SQLRAGError(RuntimeError):
    """Something went wrong running the generated SQL; message is user-safe."""


class UnsafeSQLError(SQLRAGError):
    """The model produced something other than a single SELECT."""


VALUE_HINTS = """Value formats:
- All dates are TEXT in ISO format 'YYYY-MM-DD' (e.g. '2024-03-15'); compare with strftime() or string prefixes.
- claims.status is one of: pending, approved, rejected, submitted, escalated.
- claims.claim_type is one of: cashless, reimbursement. Amounts are in rupees (REAL).
- claims.department is lowercase, e.g. cardiology, nephrology, orthopaedics, general_medicine.
- maintenance_tickets.status is one of: open, in_progress, resolved, escalated.
- maintenance_tickets.category is one of: sterilisation, infusion, radiology, monitoring, surgical, laboratory.
- maintenance_tickets.issue_type is one of: preventive_maintenance, sensor_failure, battery_replacement, fault_reported, calibration_due.
- Months: use strftime('%Y-%m', submitted_date). "Last month" relative to the data means the latest month present."""

SQL_SYSTEM_PROMPT = """You translate a staff question into ONE SQLite SELECT statement.

{schema}

{hints}

Rules: output only the SQL, no explanation, no markdown. Use only the tables and columns above.
Always alias aggregate columns with a readable name. Never modify data."""

ANSWER_SYSTEM_PROMPT = """You are MediBot's analytics assistant for MediAssist Health Network.
Given a staff question, the SQL that was run and its result rows, answer the question in one or
two plain sentences, quoting the numbers exactly. Format rupee amounts with commas. If the
result is empty, say that no matching records were found. Do not mention SQL."""

_FENCE = re.compile(r"```[a-zA-Z]*\n?|```")
_PREFIX = re.compile(r"^\s*(SQLQuery|SQL|Query)\s*:\s*", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|PRAGMA|ATTACH|DETACH|VACUUM)\b",
    re.IGNORECASE,
)


def schema_summary(db_path: Path) -> str:
    """The CREATE statements straight from the database, so the prompt never drifts from it."""
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' ORDER BY name").fetchall()
    return "\n\n".join(row[0] for row in rows)


def clean_sql(raw: str) -> str:
    """Models wrap SQL in fences, prefix it with 'SQLQuery:' or add prose. Keep the statement."""
    text = _FENCE.sub("", raw or "")
    text = _PREFIX.sub("", text)
    lines = text.strip().splitlines()
    # drop leading prose: start from the first line that begins like a statement
    for i, line in enumerate(lines):
        if re.match(r"^\s*(SELECT|WITH)\b", line, re.IGNORECASE):
            lines = lines[i:]
            break
    statement = " ".join(l.strip() for l in lines).strip()
    return statement.split(";")[0].strip()


def ensure_read_only(sql: str) -> None:
    if not re.match(r"^\s*(SELECT|WITH)\b", sql or "", re.IGNORECASE):
        raise UnsafeSQLError("The generated query is not a SELECT statement, so it was not run.")
    if ";" in sql:
        raise UnsafeSQLError("Only a single statement may be run.")
    if _FORBIDDEN.search(sql):
        raise UnsafeSQLError("The generated query tries to modify data, so it was not run.")


def _connect(db_path: Path) -> sqlite3.Connection:
    if not Path(db_path).is_file():
        raise SQLRAGError(f"Database not found at {db_path}.")
    # mode=ro: the OS-level guarantee behind ensure_read_only
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def run_sql(db_path: Path, sql: str, max_rows: int = 50) -> tuple[list[str], list[tuple]]:
    try:
        with _connect(db_path) as conn:
            cursor = conn.execute(sql)
            columns = [d[0] for d in cursor.description or []]
            rows = cursor.fetchmany(max_rows)
    except sqlite3.Error as e:
        raise SQLRAGError(f"The database could not run the generated query: {e}") from e
    return columns, rows


def format_result(columns: list[str], rows: list[tuple]) -> str:
    if not rows:
        return "(no rows returned)"
    header = " | ".join(columns)
    body = "\n".join(" | ".join("" if v is None else str(v) for v in row) for row in rows)
    return f"{header}\n{body}"


def sql_rag_chain(question: str, llm: Any | None = None, db_path: Path | None = None) -> str:
    """Plain function: natural-language question in, natural-language answer out."""
    settings = get_settings()
    db_path = db_path or settings.db_path
    if llm is None:
        from medibot.llm import GroqLLM

        llm = GroqLLM(model=settings.groq_model, api_key=settings.groq_api_key)

    # Step 1: question -> SQL
    system = SQL_SYSTEM_PROMPT.format(schema=schema_summary(db_path), hints=VALUE_HINTS)
    raw_sql = llm.complete(system=system, user=f"Question: {question}\nSQL:")

    # Step 2: keep only the statement, and only if it is a read
    sql = clean_sql(raw_sql)
    ensure_read_only(sql)

    # Step 3: run it, then let the model phrase the rows as an answer
    columns, rows = run_sql(db_path, sql)
    user = f"Question: {question}\nSQL: {sql}\nResult:\n{format_result(columns, rows)}\n\nAnswer:"
    return llm.complete(system=ANSWER_SYSTEM_PROMPT, user=user)
