"""Validate the judge against human labels.

This is the check that makes every other number in the harness meaningful. An
LLM judge is a measuring instrument; an instrument nobody calibrated produces
confident readings that mean nothing.

Workflow:
  1. python scripts/run_eval.py                    -> results/runs-<stamp>.json
  2. python scripts/judge_agreement.py --init results/runs-<stamp>.json
       writes agents/<agent>/data/human_labels.yaml with each reply, the tool calls and results
       the judge will see, and a blank verdict field
  3. label them yourself: pass / fail, no peeking at the judge's score
  4. python scripts/judge_agreement.py             -> agreement %, disagreements

The judge is trusted only with agreement >= 85%, Cohen's kappa >= 0.6, and no
lenient disagreements (see harness/agreement.py for why each).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agents  # noqa: E402
from harness import agreement, config, dataset, judge as judge_mod  # noqa: E402
from harness.system import SystemUnderTest  # noqa: E402


def init_labels(system: SystemUnderTest, runs_path: str, limit: int) -> None:
    labels = system.labels_path
    runs = json.loads(Path(runs_path).read_text(encoding="utf-8"))
    existing = {}
    if labels.exists():
        for row in yaml.safe_load(labels.read_text(encoding="utf-8")) or []:
            existing[(row["case_id"], row["repetition"])] = row.get("human_verdict", "")

    rows = agreement.label_rows(runs, existing, limit)
    labels.write_text(
        yaml.safe_dump(rows, allow_unicode=True, sort_keys=False, width=100), encoding="utf-8"
    )
    print(f"wrote {len(rows)} rows to {labels}")
    print("label each row's human_verdict as 'pass' or 'fail', then re-run without --init")


def measure(system: SystemUnderTest) -> int:
    labels = system.labels_path
    if not labels.exists():
        print(f"no labels at {labels} -- run with --init first")
        return 1
    rows = yaml.safe_load(labels.read_text(encoding="utf-8")) or []
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("no ANTHROPIC_API_KEY set: measuring agreement runs the judge, which needs a key")
        return 1
    cases = {c.id: c for c in dataset.load_cases(system)}

    def judge(case, out):
        return judge_mod.judge(system.rubric, case, out)

    result = agreement.measure(rows, cases, judge, config.JUDGE_MIN_SCORE)

    if result.stale:
        print(f"{len(result.stale)} labelled rows have no tool calls recorded, so the judge")
        print("cannot score their faithfulness; re-run --init to refresh them (verdicts are kept):")
        print("  " + ", ".join(result.stale))
    if not result.total:
        print("no rows labelled yet")
        return 1

    kappa = "undefined" if result.kappa is None else f"{result.kappa:.2f}"
    print(f"\njudge agreement: {result.agreed}/{result.total} = {result.rate:.0%}, kappa {kappa}  (judge={config.JUDGE_MODEL})")
    print(f"disagreements: {result.lenient} lenient (judge passed, human failed), {result.strict} strict")
    if result.unjudged:
        print(f"{len(result.unjudged)} rows got no verdict (refusal or truncation): {', '.join(result.unjudged)}")
    if result.disagreements:
        print("\ndisagreements -- read these before touching the suite:")
        for case_id, human, jv, why in result.disagreements:
            print(f"  {case_id}: human={human} judge={jv}")
            print(f"    judge said: {why}")
    ok, reasons = agreement.trusted(result)
    if not ok:
        print("\nThe judge is not trusted yet: " + "; ".join(reasons) + ". Sharpen the rubric.")
        return 1
    print("\nThe judge clears the bar.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="support", help="which system under test (agents/<name>)")
    ap.add_argument("--init", metavar="RUNS_JSON", help="seed human_labels.yaml from a results file")
    ap.add_argument("--limit", type=int, default=30, help="how many rows to label")
    args = ap.parse_args()
    system = agents.get(args.agent)
    if args.init:
        init_labels(system, args.init, args.limit)
        return 0
    return measure(system)


if __name__ == "__main__":
    raise SystemExit(main())
