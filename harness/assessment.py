"""The assessment: one run, written up for someone who has to act on it.

The other reports serve tools (JUnit), the CI page (Markdown summary) or a
tester triaging cases (HTML). This one answers a reader who did not run the
eval: how good is the agent, where does it fail, how sure are we, and what does
it cost. So it leads with the answer, says what was NOT measured before any
number can be over-read, and backs every failure with the evidence: the
question, the answer, and the check's own words.
"""
from __future__ import annotations

import re
from collections import defaultdict
from statistics import median
from typing import Any

from harness.schema import Case

THIN = 3  # a check applied to fewer runs than this is reported as thinly tested
EXAMPLES_PER_CHECK = 2
EXCERPT = 240


def describe(doc: str | None) -> str:
    """A check's description for a reader: the first paragraph of its docstring,
    on one line, without pointers into the code."""
    first = (doc or "").strip().split("\n\n")[0]
    return re.sub(r"\s*\(see [^)]*\)", "", " ".join(first.split()))


def _rate(passed: int, total: int) -> str:
    return f"{passed / total:.0%} ({passed}/{total})" if total else "n/a"


def _status(passes: int, runs: int) -> str:
    return "PASS" if passes == runs else "FAIL" if passes == 0 else "FLAKY"


def _excerpt(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= EXCERPT else text[: EXCERPT - 1] + "…"


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]


def build(
    report: dict[str, Any],
    runs: list[dict[str, Any]],
    cases: dict[str, Case],
    descriptions: dict[str, str],
    price_per_mtok: tuple[float, float] | None = None,
) -> str:
    agent = report.get("agent") or "agent"
    summaries = report["cases"]
    run_pass = {(r["case_id"], r["repetition"]): _run_passed(r) for r in runs}
    total_runs, passed_runs = len(run_pass), sum(run_pass.values())
    statuses = {c["case_id"]: _status(c["passes"], c["runs"]) for c in summaries}
    count = {s: sum(1 for v in statuses.values() if v == s) for s in ("PASS", "FLAKY", "FAIL")}
    graders = sorted(report.get("graders") or [], key=lambda g: (g["passed"] / g["applicable"] if g["applicable"] else 2, g["name"]))
    failing = [g for g in graders if g["applicable"] and g["passed"] < g["applicable"]]

    out: list[str] = [f"# Assessment: the `{agent}` agent", ""]
    out += [
        f"Model `{report['agent_model']}` · {len(summaries)} cases × {report['repetitions']} repetitions "
        f"= {total_runs} runs · outputs: {report.get('source', 'live')} · "
        f"{'judged' if report.get('judged') else 'not judged'}",
        "",
        "## Summary",
        "",
        f"**{_rate(passed_runs, total_runs)} of runs passed every check.** "
        f"Of {len(summaries)} cases, {count['PASS']} passed every repetition, "
        f"{count['FLAKY']} passed only some (flaky) and {count['FAIL']} never passed.",
        "",
    ]
    if failing:
        out.append("Where it fails, most often first:")
        out.append("")
        for g in sorted(failing, key=lambda g: g["passed"] - g["applicable"])[:3]:
            fails = g["applicable"] - g["passed"]
            out.append(f"- **{g['name']}** — {descriptions.get(g['name'], '')} Failed {fails} of {g['applicable']} runs.")
        out.append("")
    else:
        out += ["No check failed.", ""]

    # Before any number can be over-read: what this run does not show.
    caveats = []
    source = report.get("source", "live")
    if source != "live":
        caveats.append(
            f"The outputs were replayed from fixtures ({source}), not produced by the model in this run. "
            "This shows what the checks catch, not how the model behaves."
        )
    if not report.get("judged"):
        caveats.append(
            "No LLM judge scored these answers. Whether an answer is helpful and clear, beyond what the "
            "checks test, is unmeasured."
        )
    if report["repetitions"] < 5:
        caveats.append(
            f"{report['repetitions']} repetitions per case: a case that fails 1 run in 10 will usually look "
            "like a pass. Read PASS as 'no failure seen', not 'never fails'."
        )
    thin = [g["name"] for g in graders if 0 < g["applicable"] < THIN]
    if thin:
        caveats.append(f"Thinly tested (fewer than {THIN} runs): {', '.join(f'`{t}`' for t in thin)}. Their rates say little.")
    unused = [g["name"] for g in graders if not g["applicable"]]
    if unused:
        caveats.append(f"Never applied in this run: {', '.join(f'`{t}`' for t in unused)}.")
    out += ["## What this does not show", ""] + [f"- {c}" for c in caveats] + [""]

    out += ["## By check", "", "Each rate counts only the runs the check applied to.", "",
            "| Check | What it checks | Passed |", "|---|---|---|"]
    for g in graders:
        out.append(f"| `{g['name']}` | {descriptions.get(g['name'], '')} | {_rate(g['passed'], g['applicable'])} |")
    out.append("")

    by_tag: dict[str, list[bool]] = defaultdict(list)
    for (case_id, _), ok in run_pass.items():
        for tag in cases[case_id].tags if case_id in cases else []:
            by_tag[tag].append(ok)
    out += ["## By category", "", "| Category | Cases | Runs passed |", "|---|---|---|"]
    case_count = defaultdict(int)
    for c in cases.values():
        for t in c.tags:
            case_count[t] += 1
    for tag, oks in sorted(by_tag.items(), key=lambda kv: (sum(kv[1]) / len(kv[1]), kv[0])):
        out.append(f"| {tag} | {case_count[tag]} | {_rate(sum(oks), len(oks))} |")
    out.append("")

    out += ["## By case", "", "| Case | Result | Runs passed | Question |", "|---|---|---|---|"]
    for c in sorted(summaries, key=lambda c: ({"FAIL": 0, "FLAKY": 1, "PASS": 2}[statuses[c["case_id"]]], c["case_id"])):
        question = cases[c["case_id"]].message if c["case_id"] in cases else ""
        out.append(f"| `{c['case_id']}` | {statuses[c['case_id']]} | {c['passes']}/{c['runs']} | {_excerpt(question)} |")
    out.append("")

    out += ["## Failures, with evidence", ""]
    if not failing:
        out += ["None.", ""]
    for g in sorted(failing, key=lambda g: g["passed"] - g["applicable"]):
        examples = [
            (r, res)
            for r in runs
            for res in r.get("deterministic", [])
            if res["name"] == g["name"] and not res["passed"] and res.get("applicable", True)
        ]
        out += [f"### `{g['name']}` — failed {len(examples)} run(s)", ""]
        for r, res in examples[:EXAMPLES_PER_CHECK]:
            case = cases.get(r["case_id"])
            out += [
                f"**`{r['case_id']}`, run {r['repetition'] + 1}.**",
                "",
                f"- Asked: {_excerpt(case.message) if case else '?'}",
                f"- Answered: {_excerpt(r['output'].get('reply', '')) or '(nothing)'}",
                f"- Why it failed: {res['detail']}",
                "",
            ]
        if len(examples) > EXAMPLES_PER_CHECK:
            out += [f"…and {len(examples) - EXAMPLES_PER_CHECK} more in the runs file.", ""]

    out += ["## Cost and latency per task", ""]
    out += _cost(runs, price_per_mtok)
    return "\n".join(out).rstrip() + "\n"


def _run_passed(run: dict[str, Any]) -> bool:
    if run["output"].get("error"):
        return False
    return all(g["passed"] for g in run.get("deterministic", []))


def _cost(runs: list[dict[str, Any]], price: tuple[float, float] | None) -> list[str]:
    measured = [r["output"] for r in runs if r["output"].get("input_tokens") or r["output"].get("output_tokens")]
    timed = [r["output"]["latency_ms"] for r in runs if r["output"].get("latency_ms")]
    if not measured and not timed:
        return [
            "Not measured: these outputs carry no token counts or timings (they were not produced by a "
            "live run). A live run records both per task.",
            "",
        ]
    lines = ["| Measure | Per task |", "|---|---|"]
    if measured:
        ins = [o["input_tokens"] for o in measured]
        outs = [o["output_tokens"] for o in measured]
        turns = [o.get("turns", 0) for o in measured]
        lines.append(f"| Model requests | median {median(turns):g}, max {max(turns)} |")
        lines.append(f"| Input tokens | median {median(ins):,.0f} |")
        lines.append(f"| Output tokens (incl. thinking) | median {median(outs):,.0f} |")
        if price:
            costs = [(i * price[0] + o * price[1]) / 1e6 for i, o in zip(ins, outs)]
            lines.append(f"| Cost | median ${median(costs):.4f}, total ${sum(costs):.2f} for {len(costs)} runs |")
    if timed:
        lines.append(f"| Latency | median {median(timed) / 1000:.1f} s, p95 {_p95(timed) / 1000:.1f} s |")
    note = "" if price or not measured else " Cost is not shown: set the model's price to compute it."
    return lines + ["", f"Measured over {len(measured or timed)} runs.{note}", ""]
