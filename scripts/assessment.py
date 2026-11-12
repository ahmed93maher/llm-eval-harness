"""Write the assessment for a saved run: the report you hand to someone.

  python scripts/assessment.py --agent shop          # newest shop run
  python scripts/assessment.py results/report-<stamp>-shop.json
  AGENT_PRICE_PER_MTOK=1.0,5.0 python scripts/assessment.py --agent shop

Writes results/assessment-<stamp>-<agent>.md next to the run. The price is
input,output in dollars per million tokens; without it, cost is left out
rather than guessed.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agents  # noqa: E402
from harness import assessment, dataset, reporting  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?", help="path to a report-*.json (default: newest of --agent)")
    ap.add_argument("--agent", default=None, help="newest report of this agent")
    ap.add_argument("-o", "--out", default=None, help="output path")
    args = ap.parse_args()

    path = Path(args.report) if args.report else reporting.newest_report(args.agent)
    if path is None:
        print("no results yet -- run scripts/run_eval.py first")
        return 1
    report, runs = reporting.load_results(path)
    system = agents.get(report.get("agent") or "support")
    cases = {c.id: c for c in dataset.load_cases(system)}
    descriptions = {g.__name__: assessment.describe(g.__doc__) for g in system.graders}

    price = None
    if os.getenv("AGENT_PRICE_PER_MTOK"):
        price_in, price_out = (float(x) for x in os.environ["AGENT_PRICE_PER_MTOK"].split(","))
        price = (price_in, price_out)

    out = Path(args.out) if args.out else path.with_name(path.name.replace("report-", "assessment-")).with_suffix(".md")
    out.write_text(assessment.build(report, runs, cases, descriptions, price), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
