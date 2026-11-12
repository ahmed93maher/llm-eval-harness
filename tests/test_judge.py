"""The judge call itself, against a fake client: what it sends, and how it
handles a verdict that never came. Uses the shop agent, whose faithfulness is
about figures, so the judge must see the tool results that hold them."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agents.shop import SYSTEM, dataset_cases, tools
from harness import config, judge
from harness.schema import AgentOutput, JudgeVerdict

CASE = dataset_cases()["revenue-yesterday"]
RESULT, _ = tools.execute("get_sales", {"store_id": "all", "start_date": "2026-10-14", "end_date": "2026-10-14"})
OUT = AgentOutput(
    reply="Yesterday all stores took $2,195.00 from 46 orders.",
    tool_calls=[{"name": "get_sales", "args": {"store_id": "all"}, "result": json.dumps(RESULT), "is_error": False}],
)
VERDICT = JudgeVerdict(helpfulness=5, faithfulness=5, tone=4, reasoning="figures match the tool result")


class FakeJudgeClient:
    def __init__(self, stop_reason: str = "end_turn"):
        self.requests: list[dict] = []
        self.stop_reason = stop_reason
        self.messages = SimpleNamespace(parse=self.parse)

    def parse(self, **request):
        self.requests.append(request)
        return SimpleNamespace(stop_reason=self.stop_reason, parsed_output=VERDICT)


@pytest.fixture(autouse=True)
def live(monkeypatch):
    monkeypatch.setattr(config, "FAKE", False)


def test_the_judge_scores_against_the_agents_own_rubric():
    client = FakeJudgeClient()
    assert judge.judge(SYSTEM.rubric, CASE, OUT, client=client) == VERDICT  # type: ignore[arg-type]
    [req] = client.requests
    assert req["system"] == SYSTEM.rubric and "store manager" in req["system"]
    assert req["model"] == config.JUDGE_MODEL and req["output_format"] is JudgeVerdict


def test_the_judge_sees_the_figures_the_agent_was_given():
    client = FakeJudgeClient()
    judge.judge(SYSTEM.rubric, CASE, OUT, client=client)  # type: ignore[arg-type]
    prompt = client.requests[0]["messages"][0]["content"]
    assert "2195.0" in prompt and "$2,195.00" in prompt and CASE.message in prompt


def test_no_sampling_parameters_and_explicit_effort():
    client = FakeJudgeClient()
    judge.judge(SYSTEM.rubric, CASE, OUT, client=client)  # type: ignore[arg-type]
    req = client.requests[0]
    assert "temperature" not in req and req["output_config"] == {"effort": config.JUDGE_EFFORT}


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_no_verdict_is_reported_as_unjudged_never_as_a_score(stop_reason):
    assert judge.judge(SYSTEM.rubric, CASE, OUT, client=FakeJudgeClient(stop_reason)) is None  # type: ignore[arg-type]


def test_a_run_that_never_answered_is_not_judged():
    client = FakeJudgeClient()
    assert judge.judge(SYSTEM.rubric, CASE, AgentOutput(reply="", error="boom"), client=client) is None  # type: ignore[arg-type]
    assert client.requests == []


def test_offline_runs_never_call_the_judge(monkeypatch):
    monkeypatch.setattr(config, "FAKE", True)
    client = FakeJudgeClient()
    assert judge.judge(SYSTEM.rubric, CASE, OUT, client=client) is None  # type: ignore[arg-type]
    assert client.requests == []
