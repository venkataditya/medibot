from types import SimpleNamespace

import groq
import httpx
import pytest

from medibot.llm import GroqLLM, LLMError


def _response(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeCompletions:
    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return _response(self.outcome)


def fake_client(outcome) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(outcome)))


def _status_error(cls, status: int):
    request = httpx.Request("POST", "https://api.groq.com/v1/chat/completions")
    response = httpx.Response(status, request=request, json={"error": {"message": "nope"}})
    return cls("nope", response=response, body=None)


def test_complete_returns_stripped_content_and_passes_prompts() -> None:
    client = fake_client("  42  ")
    llm = GroqLLM(model="m", client=client)
    assert llm.complete(system="sys", user="usr") == "42"
    call = client.chat.completions.calls[0]
    assert call["model"] == "m"
    assert call["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]
    assert call["temperature"] == 0


def test_json_mode_asks_groq_for_a_json_object() -> None:
    client = fake_client('{"a": 1}')
    GroqLLM(model="m", client=client).complete(system="s", user="u", json_mode=True)
    assert client.chat.completions.calls[0]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    ("error", "hint"),
    [
        (_status_error(groq.AuthenticationError, 401), "API key"),
        (_status_error(groq.NotFoundError, 404), "model"),
        (_status_error(groq.RateLimitError, 429), "rate limit"),
        (_status_error(groq.InternalServerError, 500), "Groq"),
        (groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com")), "reach"),
    ],
)
def test_groq_failures_become_one_friendly_llm_error(error: Exception, hint: str) -> None:
    llm = GroqLLM(model="m", client=fake_client(error))
    with pytest.raises(LLMError) as exc:
        llm.complete(system="s", user="u")
    assert hint.lower() in str(exc.value).lower()


@pytest.mark.parametrize("content", [None, "", "   "])
def test_empty_completion_is_an_error_not_a_blank_answer(content) -> None:
    with pytest.raises(LLMError):
        GroqLLM(model="m", client=fake_client(content)).complete(system="s", user="u")


def test_missing_api_key_fails_at_construction() -> None:
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        GroqLLM(model="m", api_key="")
