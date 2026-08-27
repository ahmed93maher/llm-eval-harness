"""Validate the judge against human labels.

This is the check that makes every other number in the harness meaningful. An
LLM judge is a measuring instrument; an instrument nobody calibrated produces
confident readings that mean nothing.

Workflow:
  1. python scripts/run_eval.py                    -> results/runs-<stamp>.json
  2. python scripts/judge_agreement.py --init results/runs-<stamp>.json
       writes data/human_labels.yaml with the replies and a blank verdict field
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

from harness import config, dataset  # noqa: E402
from harness.graders import judge as judge_mod  # noqa: E402
from harness.schema import AgentOutput  # noqa: E402

LABELS = config.DATA / "human_labels.yaml"


def init_labels(runs_path: str, limit: int) -> None:
    runs = json.loads(Path(runs_path).read_text(encoding="utf-8"))
    existing = {}
    if LABELS.exists():
        for row in yaml.safe_load(LABELS.read_text(encoding="utf-8")) or []:
            existing[(row["case_id"], row["repetition"])] = row.get("human_verdict", "")

    rows = []
    for r in runs[:limit]:
        key = (r["case_id"], r["repetition"])
        rows.append(
            {
                "case_id": r["case_id"],
                "repetition": r["repetition"],
                "reply": r["output"]["reply"],
                # fill in "pass" or "fail" by hand
                "human_verdict": existing.get(key, ""),
            }
        )
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
    labelled = [r for r in rows if r.get("human_verdict") in ("pass", "fail")]
    if not labelled:
        print("no rows labelled yet")
        return 1

    cases = {c.id: c for c in dataset.load_cases()}
    agree = 0
    disagreements = []

    for row in labelled:
        case = cases.get(row["case_id"])
        if case is None:
            continue
        verdict = judge_mod.judge(case, AgentOutput(reply=row["reply"]))
        if verdict is None:
            continue
        low = min(verdict.helpfulness, verdict.faithfulness, verdict.tone)
        judge_verdict = "pass" if low >= config.JUDGE_MIN_SCORE else "fail"
        if judge_verdict == row["human_verdict"]:
            agree += 1
        else:
            disagreements.append((row["case_id"], row["human_verdict"], judge_verdict, verdict.reasoning))

    total = len(labelled)
    print(f"\njudge agreement: {agree}/{total} = {agree / total:.0%}  (judge={config.JUDGE_MODEL})")
    if disagreements:
        print("\ndisagreements -- read these before touching the suite:")
        for case_id, human, jv, why in disagreements:
            print(f"  {case_id}: human={human} judge={jv}")
            print(f"    judge said: {why}")
    if agree / total < 0.85:
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
