"""Recorded agent runs for offline replay: one file per (case, repetition).

Why per repetition: the offline suite exists to grade real model behaviour
without an API key. With one sample per case, every repetition replays the same
output, so an offline pass rate can only be 0% or 100% and a flaky case can
never show up. Recording each repetition keeps the run-to-run variance the live
model actually produced.

Why keyed by case id, not by message: editing a case's message must not
silently orphan its recording, and two cases may send the same message with
different expectations.

A missing recording is a failed run, not a skipped one: it fails `completed`
loudly instead of letting a new case pass by having nothing to grade.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from harness import config
from harness.schema import AgentOutput

FIXTURES = config.DATA / "fixtures"

# "hand-authored" marks a test double written by a person; "recorded" marks
# genuine model output. The report says which one it graded.
Source = Literal["recorded", "hand-authored"]


def path(case_id: str, repetition: int) -> Path:
    return FIXTURES / case_id / f"{repetition}.json"


def save(case_id: str, repetition: int, out: AgentOutput, source: Source = "recorded") -> Path:
    p = path(case_id, repetition)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source,
        "agent_model": config.AGENT_MODEL if source == "recorded" else None,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output": out.model_dump(),
    }
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def load(case_id: str, repetition: int) -> AgentOutput:
    p = path(case_id, repetition)
    if not p.exists():
        return AgentOutput(reply="", error=f"no fixture recorded for {case_id}#{repetition}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    return AgentOutput.model_validate(payload["output"])


def clear(case_id: str) -> None:
    """Drop a case's old recordings before re-recording it, so a smaller
    re-recording never leaves repetitions from an older model behind."""
    for p in (FIXTURES / case_id).glob("*.json"):
        p.unlink()


def missing(case_ids: list[str], repetitions: int) -> list[str]:
    """Recordings an offline run would need but does not have, as "case#rep".

    Checked before an offline run starts: graded, a gap would fail some
    repetitions and pass others, and look exactly like a flaky agent.
    """
    return [f"{c}#{r}" for c in case_ids for r in range(repetitions) if not path(c, r).exists()]
