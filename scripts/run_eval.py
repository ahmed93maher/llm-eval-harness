"""Run the eval suite.

  python scripts/run_eval.py                 # full judged run of the support agent
  python scripts/run_eval.py --agent support # pick the system under test
  python scripts/run_eval.py --no-judge      # deterministic invariants only
  EVAL_FAKE=1 python scripts/run_eval.py     # offline, from fixtures, no API key
  python scripts/run_eval.py --tag arabic    # slice by tag
  python scripts/run_eval.py --reps 5        # more repetitions = better flake signal

Exit code is non-zero on a regression against the committed baseline, so this is
safe to use directly as a CI gate.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agents  # noqa: E402
from harness import baseline, config, dataset, fixtures, runner  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="support", help="which system under test (agents/<name>)")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judge (free, fast)")
    ap.add_argument("--reps", type=int, default=None, help="runs per case")
    ap.add_argument("--tag", action="append", default=[], help="only cases with this tag")
    ap.add_argument("--case", action="append", default=[], help="only these case ids")
    ap.add_argument("--no-gate", action="store_true", help="report only, never fail")
    args = ap.parse_args()

    system = agents.get(args.agent)
    cases = dataset.load_cases(system)
    df = dataset.validate(system, cases)
    print(dataset.summarise(df))

    if args.tag:
        cases = [c for c in cases if set(args.tag) & set(c.tags)]
    if args.case:
        cases = [c for c in cases if c.id in args.case]
    if not cases:
        print("no cases matched the filter")
        return 1

    reps = args.reps or config.REPETITIONS
    if not config.FAKE and not os.getenv("ANTHROPIC_API_KEY"):
        # Without a key every run fails at the API; graded, that is a suite of
        # errors that can still pass a gate. Stop before it starts.
        print("\nno ANTHROPIC_API_KEY set: add it to .env, or run offline with EVAL_FAKE=1")
        return 1
    if config.FAKE:
        gaps = fixtures.missing(system.fixtures_dir, [c.id for c in cases], reps)
        if gaps:
            print(f"\nno recording for {len(gaps)} run(s): {', '.join(gaps[:8])}"
                  f"{' ...' if len(gaps) > 8 else ''}")
            print(f"record them (scripts/record_fixtures.py --agent {system.name} --reps N) or lower --reps.")
            return 1

    use_judge = not args.no_judge and not config.FAKE
    mode = "judged" if use_judge else "deterministic-only"
    print(
        f"\nrunning {len(cases)} cases x {reps} reps ({mode}) "
        f"| agent={system.name} on {config.AGENT_MODEL}{' [FIXTURES]' if config.FAKE else ''}\n"
    )

    report, runs = runner.run_suite(system, cases, repetitions=reps, use_judge=use_judge)
    report_path, runs_path = runner.save(report, runs)

    print(f"\nsuite pass rate: {report.suite_pass_rate:.1%}")
    if config.FAKE:
        print(f"graded fixtures: {report.source}")
    print("\nper grader (runs it applied to):")
    for g in sorted(report.graders, key=lambda g: (g.rate, g.name)):
        rate = f"{g.rate:6.1%}" if g.applicable else "   n/a"
        print(f"  {rate}  {g.name:28} {g.passed}/{g.applicable}")
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
