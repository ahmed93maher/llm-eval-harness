"""The shop agent's graders: did it use the tools it should, does every
figure it states trace back to the data, and did it keep quiet about figures
the data cannot give?

Each is (case, out) -> GraderResult.
"""
from __future__ import annotations

import re

from agents.shop import backoffice
from agents.shop.cases import ShopCase as Case
from harness import figures
from harness.graders import completed, language_matches
from harness.schema import AgentOutput, GraderResult

# A money amount: $12, 12.50, 1,234, "USD 40", "40 dollars". Dates ("20 August",
# "2026-09-01") deliberately do not match. Arabic-Indic digits are normalised first.
MONEY_RE = re.compile(
    r"[$€£]\s?\d|\d[\d,]*\.\d{2}\b|\b\d{1,3}(?:,\d{3})+\b|\bUSD\s?\d|\d\s?(?:USD|dollars?)\b|\d\s?دولار",
    re.IGNORECASE,
)
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩٬٫", "0123456789,.")


def expected_tools_called(case: Case, out: AgentOutput) -> GraderResult:
    """Every tool the case expects was called at least once."""
    name = "expected_tools_called"
    if not case.expect_tools:
        return GraderResult(name=name, passed=True, applicable=False, detail="no tool expected")
    called = {c.get("name") for c in out.tool_calls}
    missing = [t for t in case.expect_tools if t not in called]
    if missing:
        return GraderResult(name=name, passed=False, detail=f"not called: {missing}; called: {sorted(called)}")
    return GraderResult(name=name, passed=True)


def no_tools_when_none_expected(case: Case, out: AgentOutput) -> GraderResult:
    """Asking for a missing period, or declining an off-topic question, needs no data."""
    name = "no_tools_when_none_expected"
    if not case.expect_no_tools:
        return GraderResult(name=name, passed=True, applicable=False, detail="tools allowed")
    if out.tool_calls:
        return GraderResult(name=name, passed=False, detail=f"called: {[c.get('name') for c in out.tool_calls]}")
    return GraderResult(name=name, passed=True)


def no_figures_stated(case: Case, out: AgentOutput) -> GraderResult:
    """When the data cannot answer, any money figure in the answer is invented."""
    name = "no_figures_stated"
    if not case.must_not_state_figures:
        return GraderResult(name=name, passed=True, applicable=False, detail="figures expected")
    match = MONEY_RE.search(out.reply.translate(ARABIC_DIGITS))
    if match:
        return GraderResult(name=name, passed=False, detail=f"states a figure: {match.group()!r}")
    return GraderResult(name=name, passed=True)


def figures_grounded(case: Case, out: AgentOutput) -> GraderResult:
    """Every figure in the answer is a tool result, or plain arithmetic on tool
    results (see harness/figures.py). The check for the confident wrong number."""
    name = "figures_grounded"
    stated = figures.extract(out.reply)
    if not stated:
        return GraderResult(name=name, passed=True, applicable=False, detail="no figures stated")
    # The prompt tells the agent the shop has three stores; it may say so.
    bad = figures.ungrounded(out.reply, out.tool_calls, case.message, known=(len(backoffice.STORES),))
    if bad:
        return GraderResult(name=name, passed=False, detail=f"not in the tool results: {bad}")
    return GraderResult(name=name, passed=True, detail=f"{len(stated)} figures grounded")


ALL_GRADERS = (
    completed,
    language_matches,
    expected_tools_called,
    no_tools_when_none_expected,
    no_figures_stated,
    figures_grounded,
)
