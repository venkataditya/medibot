import pytest

from medibot import router


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, **kwargs) -> str:
        self.prompts.append((system, user))
        return self.reply


@pytest.mark.parametrize(
    ("raw", "kind", "collection"),
    [
        ('{"route": "sql", "collection": null}', "sql", "unknown"),
        ('{"route": "docs", "collection": "billing"}', "docs", "billing"),
        ('```json\n{"route": "docs", "collection": "clinical"}\n```', "docs", "clinical"),
        ('Sure! Here you go: {"route":"docs","collection":"general"} hope that helps', "docs", "general"),
        ('{"route": "DOCS", "collection": "Nursing"}', "docs", "nursing"),
        ('{"route": "docs", "collection": "finance"}', "docs", "unknown"),
        ('{"route": "teleport"}', "docs", "unknown"),
        ("not json at all", "docs", "unknown"),
        ("{route: docs, collection: billing}", "docs", "unknown"),
        ("", "docs", "unknown"),
    ],
)
def test_parse_route_is_lenient_and_defaults_to_docs_unknown(raw: str, kind: str, collection: str) -> None:
    assert router.parse_route(raw) == router.Route(kind=kind, collection=collection)


def test_route_question_sends_the_question_and_uses_json_mode() -> None:
    llm = FakeLLM('{"route": "sql", "collection": null}')
    assert router.route_question(llm, "How many claims were rejected?") == router.Route("sql", "unknown")
    system, user = llm.prompts[0]
    assert "How many claims were rejected?" in user
    assert "claims" in system and "maintenance_tickets" in system
    for name in ("general", "clinical", "nursing", "billing", "equipment"):
        assert name in system
