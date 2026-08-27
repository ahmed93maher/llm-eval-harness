"""Record real agent outputs into data/fixtures.json for offline runs.

The shipped fixtures are hand-authored placeholders. Once you have an API key,
run this to replace them with genuine model output so the offline CI job is
grading something real.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["EVAL_FAKE"] = "0"  # always record live

from harness import config, dataset  # noqa: E402
from harness.agent import run_agent  # noqa: E402


def main() -> int:
    cases = dataset.load_cases()
    fixtures: dict[str, object] = {
        "_comment": "Recorded from live runs by scripts/record_fixtures.py. One "
        "sample per case -- these are test doubles for offline grading, not a "
        "substitute for a judged run."
    }
    for case in cases:
        out = run_agent(case.message, case.account_id)
        status = "ok" if not out.error else out.error
        print(f"  {case.id:32} {status} ({out.latency_ms} ms)")
        fixtures[case.message] = {
            "reply": out.reply,
            "tool_calls": out.tool_calls,
            "latency_ms": out.latency_ms,
        }
    path = config.DATA / "fixtures.json"
    path.write_text(json.dumps(fixtures, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
