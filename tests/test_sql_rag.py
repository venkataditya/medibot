import sqlite3
from pathlib import Path

import pytest

from medibot import sql_rag


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "t.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE claims (claim_id TEXT PRIMARY KEY, department TEXT, status TEXT, approved_amount REAL, submitted_date TEXT);
        INSERT INTO claims VALUES ('C1','cardiology','approved',100.0,'2024-03-02');
        INSERT INTO claims VALUES ('C2','cardiology','rejected',NULL,'2024-03-09');
        INSERT INTO claims VALUES ('C3','neurology','approved',50.0,'2024-04-01');
        CREATE TABLE maintenance_tickets (ticket_id TEXT PRIMARY KEY, category TEXT, status TEXT);
        INSERT INTO maintenance_tickets VALUES ('T1','radiology','open');
        """
    )
    conn.commit()
    conn.close()
    return path


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("SELECT 1", "SELECT 1"),
        ("```sql\nSELECT count(*) FROM claims;\n```", "SELECT count(*) FROM claims"),
        ("```\nSELECT 1\n```", "SELECT 1"),
        ("SQLQuery: SELECT status FROM claims", "SELECT status FROM claims"),
        ("Here is the query:\nSELECT 1;\nThis counts rows.", "SELECT 1"),
        ("SELECT 1; SELECT 2", "SELECT 1"),
        ("  select 1  ", "select 1"),
        ("WITH t AS (SELECT 1) SELECT * FROM t", "WITH t AS (SELECT 1) SELECT * FROM t"),
    ],
)
def test_clean_sql_extracts_just_the_first_statement(raw: str, expected: str) -> None:
    assert sql_rag.clean_sql(raw) == expected


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM claims",
        "DROP TABLE claims",
        "UPDATE claims SET status='x'",
        "INSERT INTO claims VALUES (1)",
        "PRAGMA table_info(claims)",
        "ATTACH DATABASE 'x' AS y",
        "SELECT 1; DROP TABLE claims",
        "WITH t AS (SELECT 1) UPDATE claims SET status = 'x'",
        "",
        "explain the schema",
    ],
)
def test_only_a_single_select_is_allowed(sql: str) -> None:
    with pytest.raises(sql_rag.UnsafeSQLError):
        sql_rag.ensure_read_only(sql)


def test_select_and_cte_pass_the_guard() -> None:
    sql_rag.ensure_read_only("SELECT count(*) FROM claims WHERE status = 'rejected'")
    sql_rag.ensure_read_only("WITH t AS (SELECT 1 AS n) SELECT n FROM t")


def test_run_sql_returns_columns_and_rows(db: Path) -> None:
    columns, rows = sql_rag.run_sql(db, "SELECT department, count(*) AS n FROM claims GROUP BY department ORDER BY department")
    assert columns == ["department", "n"]
    assert rows == [("cardiology", 2), ("neurology", 1)]


def test_run_sql_caps_rows(db: Path) -> None:
    _, rows = sql_rag.run_sql(db, "SELECT claim_id FROM claims", max_rows=2)
    assert len(rows) == 2


def test_run_sql_is_read_only_even_if_the_guard_were_bypassed(db: Path) -> None:
    with pytest.raises(sql_rag.SQLRAGError):
        sql_rag.run_sql(db, "DELETE FROM claims")
    assert sql_rag.run_sql(db, "SELECT count(*) FROM claims")[1] == [(3,)]


def test_bad_sql_becomes_a_friendly_error(db: Path) -> None:
    with pytest.raises(sql_rag.SQLRAGError, match="no such column"):
        sql_rag.run_sql(db, "SELECT nope FROM claims")


def test_format_result_is_a_small_table_or_a_no_rows_note() -> None:
    assert "no rows" in sql_rag.format_result(["a"], []).lower()
    text = sql_rag.format_result(["department", "n"], [("cardiology", 2)])
    assert "department" in text and "cardiology" in text and "2" in text


def test_schema_summary_is_read_from_the_database_itself(db: Path) -> None:
    summary = sql_rag.schema_summary(db)
    assert "CREATE TABLE claims" in summary
    assert "CREATE TABLE maintenance_tickets" in summary


def test_missing_database_is_a_friendly_error(tmp_path: Path) -> None:
    with pytest.raises(sql_rag.SQLRAGError, match="not found"):
        sql_rag.run_sql(tmp_path / "missing.db", "SELECT 1")


class ScriptedLLM:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, **kwargs) -> str:
        self.calls.append((system, user))
        return self.replies.pop(0)


def test_sql_rag_chain_runs_the_three_steps(db: Path) -> None:
    llm = ScriptedLLM(
        ["```sql\nSELECT count(*) AS rejected FROM claims WHERE status = 'rejected';\n```", "One claim was rejected."]
    )
    reply = sql_rag.sql_rag_chain("How many claims were rejected?", llm=llm, db_path=db)
    assert reply == "One claim was rejected."
    # step 1: the question and the schema went to the model
    assert "How many claims were rejected?" in llm.calls[0][1]
    assert "CREATE TABLE claims" in llm.calls[0][0]
    assert "YYYY-MM-DD" in llm.calls[0][0]  # the value-format hints ride along with the schema
    # step 3: the executed result, not the raw SQL text, went back to the model
    assert "rejected" in llm.calls[1][1] and "1" in llm.calls[1][1]
    assert "How many claims were rejected?" in llm.calls[1][1]


def test_sql_rag_chain_refuses_to_run_destructive_sql(db: Path) -> None:
    llm = ScriptedLLM(["DELETE FROM claims"])
    with pytest.raises(sql_rag.UnsafeSQLError):
        sql_rag.sql_rag_chain("wipe it", llm=llm, db_path=db)
    assert sql_rag.run_sql(db, "SELECT count(*) FROM claims")[1] == [(3,)]


def test_sql_rag_chain_reports_bad_sql_from_the_model(db: Path) -> None:
    llm = ScriptedLLM(["SELECT nope FROM claims"])
    with pytest.raises(sql_rag.SQLRAGError):
        sql_rag.sql_rag_chain("?", llm=llm, db_path=db)


def test_sql_rag_chain_builds_its_own_groq_client_when_none_is_given(monkeypatch, db: Path) -> None:
    import medibot.llm as llm_module

    replies = ScriptedLLM(["SELECT count(*) AS n FROM claims", "There are 3 claims."])
    monkeypatch.setattr(llm_module, "GroqLLM", lambda model, api_key: replies)
    assert sql_rag.sql_rag_chain("How many claims?", db_path=db) == "There are 3 claims."
