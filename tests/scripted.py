"""A stand-in for the Anthropic client that replays scripted replies, so the
agent loops can be tested without calling the API."""
from __future__ import annotations

from types import SimpleNamespace

from anthropic.types import Message


def message(content: list[dict], stop_reason: str) -> Message:
    return Message.model_validate({
        "id": "msg", "type": "message", "role": "assistant", "model": "stub",
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    })


def text(t: str) -> dict:
    return {"type": "text", "text": t}


def tool_use(id_: str, name: str, args: dict) -> dict:
    return {"type": "tool_use", "id": id_, "name": name, "input": args}


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
