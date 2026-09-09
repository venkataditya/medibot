import pytest
from fastapi.testclient import TestClient

from medibot import api
from medibot.answer import Source
from medibot.config import Settings
from medibot.llm import LLMError
from medibot.pipeline import ChatResult
from medibot.sql_rag import SQLRAGError


class FakeBot:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def chat(self, question: str, role: str) -> ChatResult:
        self.calls.append((question, role))
        if question == "boom":
            raise LLMError("Groq rate limit reached. Wait a moment and try again.")
        if question == "badsql":
            raise SQLRAGError("The database could not run the generated query: no such column")
        if question.startswith("how many"):
            return ChatResult("12 claims.", "sql_rag", role)
        if "billing" in question and role == "nurse":
            return ChatResult("As a nurse, you don't have access to billing documents.", "hybrid_rag", role, access_denied=True)
        return ChatResult(
            "Meropenem is 1 g Q8H [1].", "hybrid_rag", role,
            sources=[Source("drug_formulary.pdf", "1. Antimicrobials", "clinical")],
        )


@pytest.fixture
def bot() -> FakeBot:
    return FakeBot()


@pytest.fixture
def client(bot: FakeBot) -> TestClient:
    settings = Settings(_env_file=None, groq_api_key="k", medibot_secret="api-test-secret-that-is-long-enough-for-hs256")
    app = api.create_app(settings=settings, bot_factory=lambda: bot)
    with TestClient(app) as c:
        yield c


def login(client: TestClient, username: str = "dr.mehta", password: str = "doctor123") -> str:
    r = client.post("/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["token"]


def test_health_reports_ok_and_the_model_in_use(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"]


def test_login_returns_a_role_tagged_token_and_the_accessible_collections(client: TestClient) -> None:
    r = client.post("/login", json={"username": "nurse.priya", "password": "nurse123"})
    assert r.status_code == 200
    body = r.json()
    assert body["role"] == "nurse"
    assert body["username"] == "nurse.priya"
    assert body["collections"] == ["general", "nursing"]
    assert body["sql_access"] is False
    assert body["token"].count(".") == 2


def test_login_with_bad_password_is_401_without_saying_which_part_was_wrong(client: TestClient) -> None:
    r = client.post("/login", json={"username": "nurse.priya", "password": "nope"})
    assert r.status_code == 401
    assert r.json()["detail"] == "Invalid username or password."


def test_collections_per_role_and_404_for_unknown(client: TestClient) -> None:
    r = client.get("/collections/technician")
    assert r.status_code == 200
    assert r.json() == {"role": "technician", "collections": ["general", "equipment"], "sql_access": False}
    assert client.get("/collections/admin").json()["sql_access"] is True
    assert client.get("/collections/ceo").status_code == 404


def test_chat_requires_a_token(client: TestClient) -> None:
    assert client.post("/chat", json={"question": "hi"}).status_code == 401
    r = client.post("/chat", json={"question": "hi"}, headers={"Authorization": "Bearer nonsense"})
    assert r.status_code == 401


def test_chat_takes_the_role_from_the_token_never_from_the_body(client: TestClient, bot: FakeBot) -> None:
    token = login(client, "nurse.priya", "nurse123")
    r = client.post(
        "/chat",
        json={"question": "show me all billing codes", "role": "admin"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    assert bot.calls == [("show me all billing codes", "nurse")]
    body = r.json()
    assert body["role"] == "nurse"
    assert body["access_denied"] is True
    assert body["retrieval_type"] == "hybrid_rag"
    assert body["sources"] == []


def test_chat_returns_answer_sources_retrieval_type_and_role(client: TestClient) -> None:
    token = login(client)
    r = client.post("/chat", json={"question": "Meropenem dose?"}, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json() == {
        "answer": "Meropenem is 1 g Q8H [1].",
        "sources": [{"source_document": "drug_formulary.pdf", "section_title": "1. Antimicrobials", "collection": "clinical"}],
        "retrieval_type": "hybrid_rag",
        "role": "doctor",
        "access_denied": False,
    }


def test_chat_sql_route_reports_sql_rag(client: TestClient) -> None:
    token = login(client, "admin.sys", "admin123")
    r = client.post("/chat", json={"question": "how many claims?"}, headers={"Authorization": f"Bearer {token}"})
    assert r.json()["retrieval_type"] == "sql_rag"
    assert r.json()["sources"] == []


def test_empty_question_is_a_422(client: TestClient) -> None:
    token = login(client)
    r = client.post("/chat", json={"question": "   "}, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 422


@pytest.mark.parametrize(("question", "hint"), [("boom", "rate limit"), ("badsql", "could not run")])
def test_upstream_failures_are_502_with_the_friendly_message(client: TestClient, question: str, hint: str) -> None:
    token = login(client)
    r = client.post("/chat", json={"question": question}, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 502
    assert hint in r.json()["detail"].lower()


def test_login_is_rate_limited_per_client(client: TestClient) -> None:
    for _ in range(api.LOGIN_LIMIT_PER_MINUTE):
        client.post("/login", json={"username": "x", "password": "y"})
    r = client.post("/login", json={"username": "x", "password": "y"})
    assert r.status_code == 429
