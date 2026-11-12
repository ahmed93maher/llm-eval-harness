"""The shop agent's graders: did it ask its tools for the right thing, does
every figure it states trace back to the data, and did it keep quiet about
figures the data cannot give?

Each is (case, out) -> GraderResult.
"""
from __future__ import annotations

import re
from datetime import date

from agents.shop import backoffice
from agents.shop.cases import ShopCase as Case
from harness import figures, path
from harness.graders import completed, language_matches
from harness.schema import AgentOutput, GraderResult

# A money amount: $12, 12.50, 1,234, "USD 40", "40 dollars". Dates ("20 August",
# "2026-09-01") deliberately do not match. Arabic-Indic digits are normalised first.
MONEY_RE = re.compile(
    r"[$€£]\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*\.\d{2}\b|\b\d{1,3}(?:,\d{3})+\b|\bUSD\s?\d|\d\s?(?:USD|dollars?)\b|\d\s?دولار",
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


def right_calls(case: Case, out: AgentOutput) -> GraderResult:
    """The calls made cover the calls the right answer needs: the right store,
    the right dates. Failed calls do not count."""
    name = "right_calls"
    if not case.expect_calls:
        return GraderResult(name=name, passed=True, applicable=False, detail="no calls specified")
    missing = path.unmet(case.expect_calls, out.tool_calls)
    if not missing:
        return GraderResult(name=name, passed=True)
    made = [path.describe_call(c) for c in out.tool_calls]
    return GraderResult(name=name, passed=False, detail=f"missing {' and '.join(missing)}; made {made or 'none'}")


def no_repeated_calls(case: Case, out: AgentOutput) -> GraderResult:
    """The same call twice is wasted work; after an error, it is a retry loop."""
    name = "no_repeated_calls"
    if not out.tool_calls:
        return GraderResult(name=name, passed=True, applicable=False, detail="no calls")
    dupes = path.repeated(out.tool_calls)
    if dupes:
        return GraderResult(name=name, passed=False, detail=f"repeated: {dupes}")
    return GraderResult(name=name, passed=True)


def states_period(case: Case, out: AgentOutput) -> GraderResult:
    """A relative period ("last week") is ambiguous; the answer must name the
    dates it used, and they must be the dates of a call it actually made."""
    name = "states_period"
    if not case.must_state_period:
        return GraderResult(name=name, passed=True, applicable=False, detail="no relative period")
    named = path.mentioned_dates(out.reply, backoffice.TODAY.year)
    for c in out.tool_calls:
        args = c.get("args", {})
        if c.get("is_error") or "start_date" not in args:
            continue
        start, end = date.fromisoformat(args["start_date"]), date.fromisoformat(args["end_date"])
        if start in named and end in named:
            return GraderResult(name=name, passed=True, detail=f"{start} to {end}")
    said = sorted(d.isoformat() for d in named)
    return GraderResult(name=name, passed=False, detail=f"answer names {said or 'no dates'}, not the range it queried")


ALL_GRADERS = (
    completed,
    language_matches,
    expected_tools_called,
    no_tools_when_none_expected,
    no_figures_stated,
    figures_grounded,
    right_calls,
    no_repeated_calls,
    states_period,
)
