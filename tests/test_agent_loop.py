"""The live agent loop against a scripted client: proves what gets recorded
(calls, results, turns, tokens, stop reason) without calling the API."""
from __future__ import annotations

from types import SimpleNamespace

import anthropic
from anthropic.types import Message

from harness import config
from agents.support.agent import run_agent
from agents.support.cases import SupportCase as Case
from agents.support.graders import completed

ARGS = {"order_id": "ORD-1041", "account_id": "ACC-77"}
CASE = Case(id="c", message="m", lang="en", account_id="ACC-77")


def message(content: list[dict], stop_reason: str) -> Message:
    return Message.model_validate({
        "id": "msg", "type": "message", "role": "assistant", "model": "stub",
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    })


def lookup(id_: str = "t1") -> dict:
    return {"type": "tool_use", "id": id_, "name": "lookup_order", "input": ARGS}


def text(t: str) -> dict:
    return {"type": "text", "text": t}


class ScriptedClient:
    def __init__(self, replies: list[Message]):
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **request):
        # Snapshot the history as the real SDK does when it serializes the
        # request: the agent keeps appending to the same list afterwards.
        self.requests.append({**request, "messages": list(request["messages"])})
        return self.replies.pop(0)


def test_records_the_call_its_result_and_the_run():
    client = ScriptedClient([message([lookup()], "tool_use"), message([text("It shipped.")], "end_turn")])
    out = run_agent("where is ORD-1041?", "ACC-77", client=client)  # type: ignore[arg-type]

    assert out.reply == "It shipped." and out.error is None
    assert (out.turns, out.input_tokens, out.output_tokens, out.stop_reason) == (2, 20, 10, "end_turn")
    [call] = out.tool_calls
    assert call["args"] == ARGS and '"found": true' in call["result"]
    # The tool result went back to the model, and no sampling parameter was sent.
    assert client.requests[1]["messages"][-1]["content"][0]["tool_use_id"] == "t1"
    assert "temperature" not in client.requests[0]


def test_turn_limit_is_a_recorded_failure(monkeypatch):
    monkeypatch.setattr(config, "AGENT_MAX_TURNS", 2)
    client = ScriptedClient([message([lookup(f"t{i}")], "tool_use") for i in range(3)])
    out = run_agent("loop", "ACC-77", client=client)  # type: ignore[arg-type]
    assert out.turns == 2 and out.error and "turn limit" in out.error
    assert not completed(CASE, out).passed


def test_truncated_reply_fails_completed():
    client = ScriptedClient([message([text("Your order ORD-1041 has")], "max_tokens")])
    out = run_agent("where?", "ACC-77", client=client)  # type: ignore[arg-type]
    result = completed(CASE, out)
    assert not result.passed and "truncated" in result.detail


def test_refusal_is_recorded():
    out = run_agent("x", "ACC-77", client=ScriptedClient([message([], "refusal")]))  # type: ignore[arg-type]
    assert out.refused and out.error == "api_refusal"


def test_api_error_is_recorded_not_raised():
    def fail(**_):
        raise anthropic.APIConnectionError(request=SimpleNamespace(method="POST", url="x"))  # type: ignore[arg-type]

    out = run_agent("x", "ACC-77", client=SimpleNamespace(messages=SimpleNamespace(create=fail)))  # type: ignore[arg-type]
    assert out.error and out.error.startswith("APIConnectionError")
