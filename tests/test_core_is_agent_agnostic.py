"""The harness core must run any agent that describes itself as a SystemUnderTest.

A toy agent defined here, sharing nothing with agents/support: if this test
needs anything support-specific to pass, the split has leaked.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

import agents
from harness import baseline, config, dataset, runner
from harness.schema import AgentOutput, Case, GraderResult
from harness.system import SystemUnderTest


class EchoCase(Case):
    must_echo: bool = True


def echoes(case: Case, out: AgentOutput) -> GraderResult:
    return GraderResult(name="echoes", passed=out.reply == case.message)


def toy(tmp_path: Path, replies: list[str]) -> SystemUnderTest:
    script = iter(replies)
    return SystemUnderTest(
        name="echo",
        case_model=EchoCase,
        data_dir=tmp_path,
        run=lambda case: AgentOutput(reply=next(script)),
        graders=(echoes,),
        rubric="(unused)",
        min_arabic_share=0.0,
    )


def test_the_runner_runs_an_agent_it_has_never_heard_of(tmp_path):
    system = toy(tmp_path, ["hi", "hello", "hi"])
    case = EchoCase(id="e", message="hi", lang="en", tags=["x"])
    report, runs = runner.run_suite(system, [case], repetitions=3, use_judge=False)
    assert report.agent == "echo"
    [summary] = report.cases
    assert (summary.passes, summary.runs, summary.flaky) == (2, 3, True)
    [rate] = report.graders
    assert (rate.name, rate.passed, rate.applicable) == ("echoes", 2, 3)


def test_dataset_rules_come_from_the_agent(tmp_path):
    def no_secrets(df):
        return ["a case mentions a secret"] if df["message"].str.contains("secret").any() else []

    system = toy(tmp_path, [])
    system = SystemUnderTest(**{**system.__dict__, "dataset_checks": (no_secrets,)})
    with pytest.raises(dataset.DatasetError, match="secret"):
        dataset.validate(system, [EchoCase(id="e", message="the secret", lang="en", tags=["x"])])


def test_each_agent_has_its_own_baseline():
    assert baseline.path_for("support") != baseline.path_for("shop")
    assert baseline.path_for("support").parent == config.BASELINES


def test_the_core_never_imports_an_agent():
    for path in Path("harness").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(n == "agents" or n.startswith("agents.") for n in names), path


def test_every_registered_agent_has_a_valid_dataset():
    for system in agents.registry().values():
        dataset.validate(system, dataset.load_cases(system))
