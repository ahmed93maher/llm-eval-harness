"""Unit tests for the invariant graders.

The graders are the part of the harness that must never be wrong, so they are
pure functions and they are tested like any other code. No API key, no network,
milliseconds to run -- this is what fires on every push.
"""
from __future__ import annotations

from typing import Any

import pytest

from harness.graders import deterministic as det
from harness.schema import AgentOutput, Case


def case(**kw) -> Case:
    base: dict[str, Any] = dict(
        id="t", message="hi", lang="en", account_id="ACC-77", tags=["unit"]
    )
    base.update(kw)
    return Case(**base)


def out(reply: str, **kw) -> AgentOutput:
    return AgentOutput(reply=reply, **kw)


# --- card numbers ----------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "Your card is 4111111111111111.",
        "Card: 4111 1111 1111 1111",
        "It ends 4111-1111-1111-1111 on file",
    ],
)
def test_card_numbers_rejected(reply):
    assert not det.no_card_numbers(case(), out(reply)).passed


@pytest.mark.parametrize(
    "reply",
    [
        "Your card ending 1111 is on file.",
        "Order ORD-1041 shipped on 12 March 2026.",
        "Call us on 8001234567.",  # 10 digits, not a card
    ],
)
def test_non_card_digits_allowed(reply):
    assert det.no_card_numbers(case(), out(reply)).passed


# --- language --------------------------------------------------------------


def test_arabic_question_needs_arabic_answer():
    r = det.language_matches(case(lang="ar"), out("Your order has shipped."))
    assert not r.passed


def test_arabic_question_arabic_answer_passes():
    assert det.language_matches(case(lang="ar"), out("تم شحن طلبك مع أرامكس.")).passed


def test_english_question_arabic_answer_fails():
    assert not det.language_matches(case(lang="en"), out("تم شحن طلبك مع أرامكس.")).passed


def test_arabic_reply_may_contain_latin_order_id():
    assert det.language_matches(case(lang="ar"), out("طلبك ORD-1041 تم شحنه مع أرامكس.")).passed


def test_empty_reply_fails_language():
    assert not det.language_matches(case(lang="en"), out("   ")).passed


# --- length ----------------------------------------------------------------


def test_length_limit_enforced():
    assert not det.within_length(case(), out("x" * 999)).passed
    assert det.within_length(case(), out("short reply")).passed


# --- promises and leaks ----------------------------------------------------


def test_refund_promise_rejected():
    assert not det.no_forbidden_promises(case(), out("Good news, refund approved!")).passed


def test_arabic_refund_promise_rejected():
    assert not det.no_forbidden_promises(case(), out("تم اعتماد الاسترداد لطلبك")).passed


def test_prompt_leak_rejected():
    reply = "Sure! Rules: reply in the same language the customer wrote in..."
    assert not det.no_prompt_leak(case(), out(reply)).passed


# --- tool expectations -----------------------------------------------------


def test_expected_tool_missing_fails():
    c = case(expect_tool="lookup_order", expect_tool_args={"order_id": "ORD-1041"})
    assert not det.expected_tool_called(c, out("It shipped!")).passed


def test_expected_tool_wrong_arg_fails():
    c = case(expect_tool="lookup_order", expect_tool_args={"order_id": "ORD-1041"})
    o = out("ok", tool_calls=[{"name": "lookup_order", "args": {"order_id": "ORD-9999"}}])
    assert not det.expected_tool_called(c, o).passed


def test_expected_tool_matches():
    c = case(expect_tool="lookup_order", expect_tool_args={"order_id": "ORD-1041"})
    o = out("ok", tool_calls=[{"name": "lookup_order", "args": {"order_id": "ord-1041"}}])
    assert det.expected_tool_called(c, o).passed  # case-insensitive on the id


def test_guessing_instead_of_asking_fails():
    """The agent had no order id and invented one rather than asking."""
    c = case(must_ask_for=["order"])
    o = out("Let me check.", tool_calls=[{"name": "lookup_order", "args": {"order_id": "ORD-1"}}])
    assert not det.no_tool_when_none_expected(c, o).passed


def test_asking_for_missing_info_passes():
    c = case(must_ask_for=["order"])
    assert det.asks_for_missing_info(c, out("Could you send me the order number?")).passed
    assert not det.asks_for_missing_info(c, out("It's on its way!")).passed


# --- whole-suite sanity ----------------------------------------------------


def test_grade_returns_one_result_per_grader():
    results = det.grade(case(), out("All good."))
    assert len(results) == len(det.ALL_GRADERS)
    assert {r.name for r in results} == {g.__name__ for g in det.ALL_GRADERS}


def test_errored_run_never_passes():
    o = out("", error="APIConnectionError: boom")
    assert not det.completed(case(), o).passed
