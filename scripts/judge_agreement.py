"""Validate the judge against human labels.

This is the check that makes every other number in the harness meaningful. An
LLM judge is a measuring instrument; an instrument nobody calibrated produces
confident readings that mean nothing.

Workflow:
  1. python scripts/run_eval.py                    -> results/runs-<stamp>.json
  2. python scripts/judge_agreement.py --init results/runs-<stamp>.json
       writes data/human_labels.yaml with each reply, the tool calls and results
       the judge will see, and a blank verdict field
  3. label them yourself: pass / fail, no peeking at the judge's score
  4. python scripts/judge_agreement.py             -> agreement %, disagreements

Below ~85% agreement, fix the rubric before trusting any suite pass rate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import agreement, config, dataset  # noqa: E402
from harness.graders import judge as judge_mod  # noqa: E402

LABELS = config.DATA / "human_labels.yaml"


def init_labels(runs_path: str, limit: int) -> None:
    runs = json.loads(Path(runs_path).read_text(encoding="utf-8"))
    existing = {}
    if LABELS.exists():
        for row in yaml.safe_load(LABELS.read_text(encoding="utf-8")) or []:
            existing[(row["case_id"], row["repetition"])] = row.get("human_verdict", "")

    rows = agreement.label_rows(runs, existing, limit)
    LABELS.write_text(
        yaml.safe_dump(rows, allow_unicode=True, sort_keys=False, width=100), encoding="utf-8"
    )
    print(f"wrote {len(rows)} rows to {LABELS}")
    print("label each row's human_verdict as 'pass' or 'fail', then re-run without --init")


def measure() -> int:
    if not LABELS.exists():
        print(f"no labels at {LABELS} -- run with --init first")
        return 1
    rows = yaml.safe_load(LABELS.read_text(encoding="utf-8")) or []
    cases = {c.id: c for c in dataset.load_cases()}
    result = agreement.measure(rows, cases, judge_mod.judge, config.JUDGE_MIN_SCORE)

    if result.stale:
        print(f"{len(result.stale)} labelled rows have no tool calls recorded, so the judge")
        print("cannot score their faithfulness; re-run --init to refresh them (verdicts are kept):")
        print("  " + ", ".join(result.stale))
    if not result.total:
        print("no rows labelled yet")
        return 1

    print(f"\njudge agreement: {result.agreed}/{result.total} = {result.rate:.0%}  (judge={config.JUDGE_MODEL})")
    if result.disagreements:
        print("\ndisagreements -- read these before touching the suite:")
        for case_id, human, jv, why in result.disagreements:
            print(f"  {case_id}: human={human} judge={jv}")
            print(f"    judge said: {why}")
    if result.rate < 0.85:
        print("\nAgreement is too low to trust the suite. Sharpen the rubric.")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", metavar="RUNS_JSON", help="seed human_labels.yaml from a results file")
    ap.add_argument("--limit", type=int, default=30, help="how many rows to label")
    args = ap.parse_args()
    if args.init:
        init_labels(args.init, args.limit)
        return 0
    return measure()


if __name__ == "__main__":
    raise SystemExit(main())
