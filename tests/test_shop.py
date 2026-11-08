"""The shop agent: its tools are the ground truth, its loop records what the
model was told, and its graders catch the failures they exist for."""
from __future__ import annotations

import json

from agents.shop import backoffice, graders, tools
from agents.shop.agent import run_agent
from agents.shop.cases import ShopCase
from harness.schema import AgentOutput
from tests.scripted import ScriptedClient, message, text, tool_use

# --- the tools: if they were wrong, every grade built on them would be too ---


def sales(store: str, start: str, end: str) -> dict:
    result, is_error = tools.execute("get_sales", {"store_id": store, "start_date": start, "end_date": end})
    assert not is_error, result
    return result


def test_data_is_deterministic():
    backoffice.unit_sales.cache_clear()
    first = dict(backoffice.unit_sales())
    backoffice.unit_sales.cache_clear()
    assert dict(backoffice.unit_sales()) == first


def test_all_stores_is_the_sum_of_each_store():
    total = sales("all", "2026-09-01", "2026-09-30")["totals"]
    parts = [sales(s, "2026-09-01", "2026-09-30")["totals"] for s in backoffice.STORES]
    assert total["units"] == sum(p["units"] for p in parts)
    assert total["orders"] == sum(p["orders"] for p in parts)
    assert abs(total["revenue"] - sum(p["revenue"] for p in parts)) < 0.01


def test_totals_are_the_sum_of_days():
    result = sales("harbor", "2026-10-01", "2026-10-07")
    assert len(result["days"]) == 7
    assert result["totals"]["units"] == sum(d["units"] for d in result["days"])


def test_out_of_range_is_an_error_not_an_empty_result():
    result, is_error = tools.execute("get_sales", {"store_id": "all", "start_date": "2026-08-20", "end_date": "2026-08-20"})
    assert is_error and "No data" in result


def test_top_products_are_ranked_by_revenue():
    result, is_error = tools.execute(
        "get_top_products", {"store_id": "online", "start_date": "2026-10-01", "end_date": "2026-10-07", "limit": 3}
    )
    assert not is_error
    revenues = [p["revenue"] for p in result["products"]]
    assert len(revenues) == 3 and revenues == sorted(revenues, reverse=True)


def test_unknown_tool_is_an_error():
    assert tools.execute("delete_everything", {}) == ("Unknown tool: delete_everything", True)


# --- the loop ---------------------------------------------------------------

YESTERDAY = {"store_id": "harbor", "start_date": "2026-10-14", "end_date": "2026-10-14"}


def test_records_each_call_with_the_result_the_model_saw():
    client = ScriptedClient([
        message([tool_use("t1", "get_sales", YESTERDAY)], "tool_use"),
        message([text("Harbor made $500.")], "end_turn"),
    ])
    out = run_agent("revenue yesterday at harbor?", client=client)  # type: ignore[arg-type]

    assert out.reply == "Harbor made $500." and out.turns == 2
    [call] = out.tool_calls
    assert call["args"] == YESTERDAY and not call["is_error"]
    sent = client.requests[1]["messages"][-1]["content"][0]
    # What was recorded is exactly what went back to the model.
    assert sent["content"] == call["result"] and json.loads(call["result"])["totals"]["revenue"] > 0


def test_tool_error_is_recorded_and_returned_to_the_model():
    bad = {"store_id": "all", "start_date": "2026-08-20", "end_date": "2026-08-20"}
    client = ScriptedClient([
        message([tool_use("t1", "get_sales", bad)], "tool_use"),
        message([text("No data before September.")], "end_turn"),
    ])
    out = run_agent("revenue on 20 August?", client=client)  # type: ignore[arg-type]
    assert out.tool_calls[0]["is_error"]
    assert client.requests[1]["messages"][-1]["content"][0]["is_error"] is True


# --- the graders --------------------------------------------------------------


def case(**kw) -> ShopCase:
    return ShopCase(id="c", message="m", lang="en", tags=["x"], **kw)


def test_expected_tools_called():
    c = case(expect_tools=["get_sales"])
    assert graders.expected_tools_called(c, AgentOutput(reply="x", tool_calls=[{"name": "get_sales"}])).passed
    result = graders.expected_tools_called(c, AgentOutput(reply="x", tool_calls=[{"name": "list_stores"}]))
    assert not result.passed and "get_sales" in result.detail
    assert not graders.expected_tools_called(case(), AgentOutput(reply="x")).applicable


def test_no_tools_when_none_expected():
    c = case(expect_no_tools=True)
    assert graders.no_tools_when_none_expected(c, AgentOutput(reply="Which period?")).passed
    assert not graders.no_tools_when_none_expected(c, AgentOutput(reply="x", tool_calls=[{"name": "get_sales"}])).passed


def test_no_figures_stated_catches_an_invented_figure():
    c = case(must_not_state_figures=True)
    for invented in ("Revenue on 20 August was $1,240.", "About 1,240.50 that day.", "كانت ١٬٢٤٠ دولار"):
        assert not graders.no_figures_stated(c, AgentOutput(reply=invented)).passed, invented


def test_no_figures_stated_allows_dates():
    c = case(must_not_state_figures=True)
    honest = "Our data starts on 1 September 2026 (2026-09-01), so I have nothing for 20 August."
    assert graders.no_figures_stated(c, AgentOutput(reply=honest)).passed


def test_unknown_case_fields_are_rejected():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ShopCase(id="c", message="m", lang="en", expect_tool=["get_sales"])  # typo: expect_tools
