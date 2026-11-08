"""System under test: a sales assistant answering a store manager's questions
over seeded sales data, with three tools (list_stores, get_sales,
get_top_products)."""
from __future__ import annotations

from pathlib import Path

from agents.shop import cases, graders
from agents.shop.rubric import RUBRIC
from harness.schema import AgentOutput, Case
from harness.system import SystemUnderTest


def _run(case: Case) -> AgentOutput:
    from agents.shop.agent import run_agent  # lazily: offline runs need no API client

    return run_agent(case.message)


SYSTEM = SystemUnderTest(
    name="shop",
    case_model=cases.ShopCase,
    data_dir=Path(__file__).resolve().parent / "data",
    run=_run,
    graders=graders.ALL_GRADERS,  # type: ignore[arg-type]
    rubric=RUBRIC,
    dataset_checks=cases.CHECKS,
)
