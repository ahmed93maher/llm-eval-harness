"""Offline replay must keep the variance between runs, and must never invent a pass.

Pins a real weakness: fixtures used to be one sample per case keyed by message,
so every offline repetition replayed the same output and a flaky case could
never show up offline.
"""
from __future__ import annotations

import pytest

from harness import config, dataset, fixtures, reporting
from harness.graders import deterministic
from harness.runner import grader_rates, run_case, summarise
from harness.schema import AgentOutput, Case, RunResult

CASE = Case(id="c", message="Where is ORD-1041?", lang="en", account_id="ACC-77", tags=["x"], must_contain=["shipped"])


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(fixtures, "FIXTURES", tmp_path)
    monkeypatch.setattr(config, "FAKE", True)
    return tmp_path


def test_each_repetition_replays_its_own_recording(store):
    fixtures.save("c", 0, AgentOutput(reply="It shipped.", stop_reason="end_turn"))
    fixtures.save("c", 1, AgentOutput(reply="It is on its way.", stop_reason="end_turn"))
    runs = run_case(CASE, repetitions=2, use_judge=False)
    assert [r.output.reply for r in runs] == ["It shipped.", "It is on its way."]
    # The second recording lacks "shipped": offline replay now shows the flake.
    assert summarise(CASE, runs).flaky


def test_a_missing_recording_fails_loudly(store):
    out = fixtures.load("c", 3)
    assert out.error == "no fixture recorded for c#3"
    assert not deterministic.completed(CASE, out).passed


def test_missing_recordings_are_listed_before_a_run(store):
    fixtures.save("c", 0, AgentOutput(reply="x"))
    assert fixtures.missing(["c"], 2) == ["c#1"]


def test_rerecording_drops_old_repetitions(store):
    for rep in range(3):
        fixtures.save("c", rep, AgentOutput(reply="old"))
    fixtures.clear("c")
    fixtures.save("c", 0, AgentOutput(reply="new"))
    assert fixtures.missing(["c"], 3) == ["c#1", "c#2"]


def test_every_shipped_case_has_a_first_recording():
    """The offline CI job runs --reps 1; a case without recording 0 would fail it."""
    assert fixtures.missing([c.id for c in dataset.load_cases()], 1) == []


# --- applicable ------------------------------------------------------------


def test_a_grader_with_nothing_to_check_is_not_applicable():
    case = Case(id="c", message="hi", lang="en", account_id="A", must_refuse=True)
    result = deterministic.asks_for_missing_info(case, AgentOutput(reply="Hello"))
    assert result.passed and not result.applicable


def test_grader_rates_leave_out_runs_a_grader_did_not_apply_to():
    asks = Case(id="a", message="hi", lang="en", account_id="A", must_ask_for=["order"])
    other = Case(id="b", message="hi", lang="en", account_id="A", must_refuse=True)
    runs = [
        RunResult(case_id=c.id, repetition=0, output=AgentOutput(reply="Hello"),
                  deterministic=deterministic.grade(c, AgentOutput(reply="Hello")))
        for c in (asks, other)
    ]
    [rate] = [g for g in grader_rates(runs) if g.name == "asks_for_missing_info"]
    # 0/1, not 1/2: the second case never tested this grader.
    assert (rate.passed, rate.applicable) == (0, 1)


def test_markdown_lists_graders_and_flags_hand_authored_fixtures():
    report = {
        "agent_model": "m", "judge_model": "", "repetitions": 1, "judged": False,
        "source": "hand-authored", "cases": [],
        "graders": [{"name": "completed", "passed": 2, "applicable": 2},
                    {"name": "asks_for_missing_info", "passed": 0, "applicable": 0}],
    }
    md = reporting.to_markdown(report, [])
    assert "| `completed` | 2/2 | 100% |" in md
    assert "| `asks_for_missing_info` | 0/0 | n/a |" in md
    assert "hand-authored" in md
