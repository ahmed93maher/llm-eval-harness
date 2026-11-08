"""Record live agent runs into data/fixtures/<case>/<repetition>.json.

  python scripts/record_fixtures.py               # every case, EVAL_REPETITIONS times
  python scripts/record_fixtures.py --reps 5      # more recordings = offline flake signal
  python scripts/record_fixtures.py --case multi_intent

Each repetition is a separate live run, so offline replay keeps the variance the
model really showed (see harness/fixtures.py). A run that failed at the API
(rate limit, network) is not saved: replaying it would grade the network, not
the agent. Re-run the case instead. Turn limits and refusals are the agent's own
behaviour, so they are saved.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["EVAL_FAKE"] = "0"  # always record live

from harness import config, dataset, fixtures  # noqa: E402
from harness.agent import run_agent  # noqa: E402
from harness.schema import AgentOutput  # noqa: E402


def is_infrastructure_error(out: AgentOutput) -> bool:
    return bool(out.error) and not out.refused and not out.error.startswith("turn limit")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=config.REPETITIONS, help="recordings per case")
    ap.add_argument("--case", action="append", default=[], help="only these case ids")
    args = ap.parse_args()

    cases = dataset.load_cases()
    if args.case:
        cases = [c for c in cases if c.id in args.case]

    unsaved = 0
    for case in cases:
        fixtures.clear(case.id)
        for rep in range(args.reps):
            out = run_agent(case.message, case.account_id)
            if is_infrastructure_error(out):
                unsaved += 1
                print(f"  {case.id:32} #{rep} NOT SAVED: {out.error}")
                continue
            fixtures.save(case.id, rep, out)
            status = out.error or "ok"
            print(f"  {case.id:32} #{rep} {status} ({out.latency_ms} ms, {out.turns} turns)")

    print(f"\nwrote {fixtures.FIXTURES}")
    if unsaved:
        print(f"{unsaved} run(s) failed at the API and were not saved; re-run those cases.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
