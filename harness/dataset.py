"""Load the eval dataset and validate it before a single API call is spent.

A bad dataset produces confident, meaningless numbers. These expectations are the
same kind you would express in Great Expectations or pandera -- kept as explicit
pandas checks here so the repo has no heavyweight dependency, and so each
expectation is readable next to the reason it exists.
"""
from __future__ import annotations

import pandas as pd
import yaml

from harness import config
from harness.schema import Case


class DatasetError(AssertionError):
    """Raised when the dataset itself is broken."""


def load_cases(path=None) -> list[Case]:
    path = path or config.DATA / "cases.yaml"
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or []
    return [Case(**entry) for entry in raw]


def to_frame(cases: list[Case]) -> pd.DataFrame:
    return pd.DataFrame([c.model_dump() for c in cases])


def validate(cases: list[Case]) -> pd.DataFrame:
    """Expectations about the dataset. Raises DatasetError on the first violation."""
    df = to_frame(cases)
    problems: list[str] = []

    if df.empty:
        raise DatasetError("dataset is empty")

    # 1. ids are unique -- duplicates silently double-weight a case
    dupes = df["id"][df["id"].duplicated()].tolist()
    if dupes:
        problems.append(f"duplicate case ids: {dupes}")

    # 2. no empty messages
    if (df["message"].str.strip() == "").any():
        problems.append("one or more cases have an empty message")

    # 3. every case carries at least one tag -- tags are how we slice results
    untagged = df.loc[df["tags"].apply(len) == 0, "id"].tolist()
    if untagged:
        problems.append(f"untagged cases: {untagged}")

    # 4. every case must be falsifiable: it needs at least one expectation
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
    if toothless:
        problems.append(f"cases with no expectation (always pass, prove nothing): {toothless}")

    # 5. tool args only make sense alongside an expected tool
    orphan = df.loc[
        (df["expect_tool"].isna()) & (df["expect_tool_args"].apply(len) > 0), "id"
    ].tolist()
    if orphan:
        problems.append(f"expect_tool_args without expect_tool: {orphan}")

    # 6. language coverage -- an Arabic-facing product needs Arabic cases
    arabic_share = (df["lang"] == "ar").mean()
    if arabic_share < 0.25:
        problems.append(f"Arabic cases are only {arabic_share:.0%} of the suite (want >= 25%)")

    # 7. adversarial coverage -- a suite of happy paths measures nothing useful
    tags = set(t for row in df["tags"] for t in row)
    for required in ("injection", "missing-data", "pii"):
        if required not in tags:
            problems.append(f"no case tagged {required!r}")

    if problems:
        raise DatasetError("dataset validation failed:\n  - " + "\n  - ".join(problems))
    return df


def summarise(df: pd.DataFrame) -> str:
    by_lang = df["lang"].value_counts().to_dict()
    tags = pd.Series([t for row in df["tags"] for t in row]).value_counts().to_dict()
    return f"{len(df)} cases | languages: {by_lang} | tags: {tags}"
