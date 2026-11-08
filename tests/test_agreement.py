"""Judge agreement must be measured on what the judge sees in a real run.

Pins a real bug: the agreement script used to re-judge each labelled reply with
no tool calls, so the judge scored faithfulness blind and the agreement number
described a judge that never runs in the suite.
"""
from __future__ import annotations

from harness import agreement
from harness.schema import AgentOutput, Case, JudgeVerdict

CASE = Case(id="order_status", message="Where is ORD-1041?", lang="en", account_id="ACC-77", expect_tool="lookup_order")

CALLS = [{
    "name": "lookup_order",
    "args": {"order_id": "ORD-1041", "account_id": "ACC-77"},
    "result": '{"found": true, "status": "shipped", "carrier": "Aramex"}',
}]

RUN = {"case_id": "order_status", "repetition": 0, "output": {"reply": "ORD-1041 shipped with Aramex.", "tool_calls": CALLS}}


def verdict(score: int) -> JudgeVerdict:
    return JudgeVerdict(helpfulness=score, faithfulness=score, tone=score, reasoning="r")


def test_label_rows_keep_the_tool_calls_and_existing_verdicts():
    [row] = agreement.label_rows([RUN], {("order_status", 0): "pass"}, limit=30)
    assert row["tool_calls"] == CALLS
    assert row["human_verdict"] == "pass"


def test_the_judge_sees_the_tool_calls_and_results():
    seen: list[AgentOutput] = []

    def judge(case: Case, out: AgentOutput) -> JudgeVerdict:
        seen.append(out)
        return verdict(5)

    rows = agreement.label_rows([RUN], {("order_status", 0): "pass"}, limit=30)
    agreement.measure(rows, {CASE.id: CASE}, judge, judge_min=4)
    assert seen[0].tool_calls == CALLS


def test_agreement_and_disagreements_are_counted():
    rows = [
        {**agreement.label_rows([RUN], {}, 30)[0], "human_verdict": "pass"},
        {**agreement.label_rows([RUN], {}, 30)[0], "repetition": 1, "human_verdict": "fail"},
    ]
    result = agreement.measure(rows, {CASE.id: CASE}, lambda c, o: verdict(5), judge_min=4)
    assert (result.agreed, result.total, result.rate) == (1, 2, 0.5)
    assert result.disagreements[0][:3] == ("order_status", "fail", "pass")


def test_rows_labelled_without_tool_calls_are_stale_not_judged_blind():
    old_row = {"case_id": "order_status", "repetition": 0, "reply": "x", "human_verdict": "pass"}
    called = []
    result = agreement.measure([old_row], {CASE.id: CASE}, lambda c, o: called.append(o) or verdict(5), judge_min=4)
    assert result.stale == ["order_status#0"] and result.total == 0 and not called


def test_unlabelled_rows_are_ignored():
    row = {**agreement.label_rows([RUN], {}, 30)[0], "human_verdict": ""}
    assert agreement.measure([row], {CASE.id: CASE}, lambda c, o: verdict(5), judge_min=4).total == 0
