"""Run the eval suite.

  python scripts/run_eval.py                 # full judged run
  python scripts/run_eval.py --no-judge      # deterministic invariants only
  EVAL_FAKE=1 python scripts/run_eval.py     # offline, from fixtures, no API key
  python scripts/run_eval.py --tag arabic    # slice by tag
  python scripts/run_eval.py --reps 5        # more repetitions = better flake signal

Exit code is non-zero on a regression against the committed baseline, so this is
safe to use directly as a CI gate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import baseline, config, dataset, runner  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judge (free, fast)")
    ap.add_argument("--reps", type=int, default=None, help="runs per case")
    ap.add_argument("--tag", action="append", default=[], help="only cases with this tag")
    ap.add_argument("--case", action="append", default=[], help="only these case ids")
    ap.add_argument("--no-gate", action="store_true", help="report only, never fail")
    args = ap.parse_args()

    cases = dataset.load_cases()
    df = dataset.validate(cases)
    print(dataset.summarise(df))

    if args.tag:
        cases = [c for c in cases if set(args.tag) & set(c.tags)]
    if args.case:
        cases = [c for c in cases if c.id in args.case]
    if not cases:
        print("no cases matched the filter")
        return 1

    use_judge = not args.no_judge and not config.FAKE
    mode = "judged" if use_judge else "deterministic-only"
    reps = args.reps or config.REPETITIONS
    print(
        f"\nrunning {len(cases)} cases x {reps} reps ({mode}) "
        f"| agent={config.AGENT_MODEL}{' [FIXTURES]' if config.FAKE else ''}\n"
    )

    report, runs = runner.run_suite(cases, repetitions=reps, use_judge=use_judge)
    report_path, runs_path = runner.save(report, runs)

    print(f"\nsuite pass rate: {report.suite_pass_rate:.1%}")
    if report.flaky_cases:
        print("\nflaky (passed some runs, failed others) -- the most useful signal here:")
        for c in report.flaky_cases:
            print(f"  {c.case_id}: {c.passes}/{c.runs}")
    print(f"\nwrote {report_path}\n      {runs_path}")

    ok, messages = baseline.compare(report)
    print("\nbaseline comparison:")
    for m in messages:
        print(f"  {m}")

    if args.no_gate:
        return 0
    if not ok:
        print("\nFAIL: quality regressed against the baseline.")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
