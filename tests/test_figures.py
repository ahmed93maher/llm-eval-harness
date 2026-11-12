"""The figure check, proven with planted faults.

No live model runs in this repo. A grader is shown to work the way a test
suite is in mutation testing: correct answers must pass, and answers with a
deliberately planted fault -- each one a realistic way a model gets a figure
wrong -- must fail. The tool results are the real ones from the seeded shop.
"""
from __future__ import annotations

import json

import pytest

from agents.shop import graders, tools
from agents.shop.cases import ShopCase
from harness import figures
from harness.schema import AgentOutput


def call(name: str, **args) -> dict:
    result, is_error = tools.execute(name, args)
    return {"name": name, "args": args, "result": result if is_error else json.dumps(result), "is_error": is_error}


YESTERDAY = [call("get_sales", store_id="all", start_date="2026-10-14", end_date="2026-10-14")]
SEPTEMBER = [call("get_sales", store_id=s, start_date="2026-09-01", end_date="2026-09-30") for s in ("harbor", "mill", "online")]
TOP3 = [call("get_top_products", store_id="online", start_date="2026-10-01", end_date="2026-10-07", limit=3)]
# Real figures: yesterday all stores $2,195.00 / 46 orders / 71 units.
# September: harbor $25,650.75, mill $15,837.50, online $41,891.00 (total $83,379.25).
# Top 3 online, 1-7 Oct: blanket $3,916.00 (44), teapot $1,566.00 (29), print $1,155.00 (33).


def check(reply: str, calls: list[dict], message: str = "") -> list[str]:
    return figures.ungrounded(reply, calls, message)


# --- correct answers pass, in the forms a model actually writes ---------------

CORRECT = [
    ("Yesterday all stores took $2,195.00 from 46 orders (71 units).", YESTERDAY),
    ("Revenue yesterday was 2195 USD.", YESTERDAY),
    ("About $2.2k yesterday.", YESTERDAY),  # rounding the answer said it was doing
    ("Roughly $2,200 yesterday.", YESTERDAY),
    ("Online made about $42k in September.", SEPTEMBER),
    ("حقق المتجر أمس ٢٬١٩٥ دولار من ٤٦ طلبًا.", YESTERDAY),  # Arabic-Indic digits
    ("Online led September with $41,891.00, ahead of Harbor Street ($25,650.75) and Mill Lane ($15,837.50).", SEPTEMBER),
    ("Together the three stores made $83,379.25 in September.", SEPTEMBER),  # sum across calls
    ("Online beat Harbor Street by $16,240.25.", SEPTEMBER),  # a difference
    ("Online was 50.2% of September revenue.", SEPTEMBER),  # a share
    ("Online made 63% more than Harbor Street.", SEPTEMBER),  # a change
    ("1. Wool Throw Blanket: $3,916.00 (44 units)\n2. Cast Iron Teapot: $1,566.00\n3. Harbor Art Print: $1,155.00", TOP3),
    ("The top 3 products made $6,637.00 together from 1 to 7 October.", TOP3),  # list total; "3" from the question
    ("Data starts on 2026-09-01, so nothing for 20 August.", []),  # dates are not figures
    ("On 14/10 the shop made $2,195.", YESTERDAY),
    ("Average order value yesterday was $47.72.", YESTERDAY),  # 2,195 / 46
    ("About 1.5 units per order yesterday.", YESTERDAY),  # 71 / 46
]


@pytest.mark.parametrize("reply,calls", CORRECT)
def test_correct_figures_are_grounded(reply, calls):
    assert check(reply, calls, message="What were the top 3 products?") == []


# --- planted faults are caught --------------------------------------------------

FAULTS = [
    ("off by one cent", "Yesterday all stores took $2,195.01.", YESTERDAY, "$2,195.01"),
    ("off by one order", "Yesterday: 47 orders.", YESTERDAY, "47"),
    ("transposed digits", "Yesterday all stores took $2,159.00.", YESTERDAY, "$2,159.00"),
    ("rounded too far", "About $2.5k yesterday.", YESTERDAY, "$2.5k"),
    ("wrong Arabic digit", "حقق المتجر أمس ٢٬١٩٦ دولار.", YESTERDAY, "2,196"),
    ("sum done wrong", "Together the stores made $83,379.52 in September.", SEPTEMBER, "$83,379.52"),
    ("share done wrong", "Online was 52% of September revenue.", SEPTEMBER, "52%"),
    ("invented with no tool call", "Revenue on 20 August was $1,240.", [], "$1,240"),
    ("figure from nowhere", "Harbor Art Print sold 3,000 units.", TOP3, "3,000"),
    # Coincidence traps: each matched by accident before the pools were separated.
    ("a percentage that is really a count", "Online was 55% of September revenue.", SEPTEMBER, "55%"),
    ("a round number near some sum", "Mill Lane made about $20,000 in September.", SEPTEMBER, "$20,000"),
    ("a count that is really an average", "Yesterday: 48 orders.", YESTERDAY, "48"),  # 2,195 / 46 = 47.7
]


@pytest.mark.parametrize("fault,reply,calls,caught", FAULTS, ids=[f[0] for f in FAULTS])
def test_planted_faults_are_caught(fault, reply, calls, caught):
    assert caught in check(reply, calls)


def test_a_failed_tool_call_grounds_nothing():
    """Numbers in a tool's error message are not data the agent can quote."""
    failed = [call("get_sales", store_id="all", start_date="2026-08-20", end_date="2026-08-20")]
    assert failed[0]["is_error"]
    assert check("Revenue was 14 dollars.", failed) == ["14"]


# --- the grader -----------------------------------------------------------------

CASE = ShopCase(id="c", message="What was our total revenue yesterday?", lang="en", tags=["figures"])


def test_grader_fails_and_names_the_figure():
    result = graders.figures_grounded(CASE, AgentOutput(reply="We took $2,915.00.", tool_calls=YESTERDAY))
    assert not result.passed and "$2,915.00" in result.detail


def test_grader_accepts_facts_from_the_prompt():
    reply = "Across all 3 stores, yesterday took $2,195.00."
    assert graders.figures_grounded(CASE, AgentOutput(reply=reply, tool_calls=YESTERDAY)).passed


def test_grader_is_not_applicable_without_figures():
    result = graders.figures_grounded(CASE, AgentOutput(reply="Which period do you mean?"))
    assert result.passed and not result.applicable
