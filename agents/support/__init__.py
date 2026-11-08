"""System under test: a customer-support agent replying over SMS and WhatsApp,
with one tool (lookup_order) and a seeded order database."""
from __future__ import annotations

from pathlib import Path

from harness.schema import AgentOutput, Case
from harness.system import SystemUnderTest

from agents.support import cases, graders
from agents.support.rubric import RUBRIC


def _run(case: Case) -> AgentOutput:
    from agents.support.agent import run_agent  # lazily: offline runs need no API client

    assert isinstance(case, cases.SupportCase)
    return run_agent(case.message, case.account_id)


SYSTEM = SystemUnderTest(
    name="support",
    case_model=cases.SupportCase,
    data_dir=Path(__file__).resolve().parent / "data",
    run=_run,
    graders=graders.ALL_GRADERS,  # type: ignore[arg-type]
    rubric=RUBRIC,
    dataset_checks=cases.CHECKS,
)
