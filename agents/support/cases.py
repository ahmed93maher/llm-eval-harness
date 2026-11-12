"""The support agent's cases: what each one expects, and what makes the dataset sound."""
from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import Field

from harness.schema import Case


class SupportCase(Case):
    """One support conversation opener. Mirrors an entry in data/cases.yaml."""

    account_id: str
    expect_tool: str | None = None
    expect_tool_args: dict[str, Any] = Field(default_factory=dict)
    must_contain: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    must_ask_for: list[str] = Field(
        default_factory=list,
        description="agent must request this missing information rather than guess",
    )
    must_refuse: bool = False


def falsifiable(df: pd.DataFrame) -> list[str]:
    """Every case needs at least one expectation, or it always passes and proves nothing."""

    def has_expectation(row) -> bool:
        return bool(
            row["expect_tool"]
            or row["must_contain"]
            or row["must_not_contain"]
            or row["must_ask_for"]
            or row["must_refuse"]
            or row["rubric"]
        )

    toothless = df.loc[~df.apply(has_expectation, axis=1), "id"].tolist()
    return [f"cases with no expectation (always pass, prove nothing): {toothless}"] if toothless else []


def no_orphan_tool_args(df: pd.DataFrame) -> list[str]:
    """Tool args only make sense alongside an expected tool."""
    orphan = df.loc[(df["expect_tool"].isna()) & (df["expect_tool_args"].apply(len) > 0), "id"].tolist()
    return [f"expect_tool_args without expect_tool: {orphan}"] if orphan else []


def adversarial_coverage(df: pd.DataFrame) -> list[str]:
    """A suite of happy paths measures nothing useful."""
    tags = {t for row in df["tags"] for t in row}
    return [f"no case tagged {t!r}" for t in ("injection", "missing-data", "pii") if t not in tags]


CHECKS = (falsifiable, no_orphan_tool_args, adversarial_coverage)
