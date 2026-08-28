"""The judge prompt is testable without calling the API.

What the judge is *shown* determines what it can possibly score correctly, so
the prompt gets pinned the same way any other contract does. These caught a real
bug: the judge was being asked to score faithfulness while seeing only which
tools were called, not what they returned.
"""
from __future__ import annotations

from harness.graders.judge import _format_context, _prompt
from harness.schema import AgentOutput, Case

CASE = Case(
    id="order_status",
    message="Where is ORD-1041?",
    lang="en",
    account_id="ACC-77",
    tags=["tool-use"],
    expect_tool="lookup_order",
    rubric="Should state the real status from the tool result.",
)

WITH_RESULT = AgentOutput(
    reply="ORD-1041 has shipped with Aramex.",
    tool_calls=[
        {
            "name": "lookup_order",
            "args": {"order_id": "ORD-1041", "account_id": "ACC-77"},
            "result": '{"found": true, "status": "shipped", "carrier": "Aramex"}',
        }
    ],
)


def test_context_includes_what_the_tool_returned():
    """Without this, faithfulness is unscoreable -- the bug this test pins."""
    ctx = _format_context(WITH_RESULT)
    assert "shipped" in ctx
    assert "Aramex" in ctx
    assert "returned:" in ctx


def test_context_includes_the_call_and_its_arguments():
    ctx = _format_context(WITH_RESULT)
    assert "lookup_order" in ctx
    assert "ORD-1041" in ctx


def test_no_tool_calls_is_stated_explicitly():
    """An empty context must read as 'nothing was looked up', not as blank."""
    ctx = _format_context(AgentOutput(reply="Could you send me the order number?"))
    assert "no tools called" in ctx


def test_missing_result_is_flagged_not_hidden():
    """Older recordings have no result field; say so rather than implying none."""
    out = AgentOutput(
        reply="It shipped.",
        tool_calls=[{"name": "lookup_order", "args": {"order_id": "ORD-1041"}}],
    )
    assert "(not recorded)" in _format_context(out)


def test_prompt_carries_message_reply_context_and_case_rubric():
    p = _prompt(CASE, WITH_RESULT)
    assert CASE.message in p
    assert WITH_RESULT.reply in p
    assert "Aramex" in p                      # the context
    assert CASE.rubric in p                   # the per-case expectation
    assert "must be supported by the context" in p


def test_prompt_does_not_leak_deterministic_verdicts():
    """The judge forms its own view; it never sees the graders' pass/fail."""
    p = _prompt(CASE, WITH_RESULT).lower()
    for leaked in ("graderesult", "no_card_numbers", "language_matches", "passed=true"):
        assert leaked not in p
