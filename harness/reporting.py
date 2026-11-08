"""Turn a saved run into something someone (or something) else can read.

Three renderers over the same data:
  to_html     -- for a human, opened locally or attached to a CI artifact
  to_junit    -- for machines: Jira/Xray, Jenkins, GitHub test reporters
  to_markdown -- for the CI run page, via $GITHUB_STEP_SUMMARY

They are here rather than in the scripts so they can be unit-tested; the scripts
in scripts/report_*.py are thin command-line wrappers.
"""
from __future__ import annotations

import glob
import html
import json
from pathlib import Path
from xml.etree import ElementTree as ET

from harness import config

# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def newest_report() -> Path | None:
    found = sorted(glob.glob(str(config.RESULTS / "report-*.json")))
    return Path(found[-1]) if found else None


def load_results(report_path: Path) -> tuple[dict, list[dict]]:
    """A report and its matching runs file (runs may be absent; that is fine)."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    runs_path = report_path.with_name(report_path.name.replace("report-", "runs-"))
    runs = json.loads(runs_path.read_text(encoding="utf-8")) if runs_path.exists() else []
    return report, runs


def status_of(case: dict) -> str:
    """PASS / FAIL / FLAKY for one case summary."""
    if 0 < case["passes"] < case["runs"]:
        return "FLAKY"
    return "PASS" if case["passes"] == case["runs"] else "FAIL"


def suite_rate(cases: list[dict]) -> float:
    if not cases:
        return 0.0
    return sum(c["passes"] / c["runs"] for c in cases) / len(cases)


def _esc(text) -> str:
    return html.escape(str(text) if text is not None else "")


def _runs_by_case(runs: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in runs:
        out.setdefault(r["case_id"], []).append(r)
    return out


# --------------------------------------------------------------------------
# JUnit XML -- the interchange format
# --------------------------------------------------------------------------


def to_junit(report: dict, runs: list[dict], suite_name: str = "llm-eval") -> str:
    """JUnit XML, the format Xray, Jenkins and GitHub test reporters all ingest.

    Mapping decisions worth knowing, because JUnit has no vocabulary for any of
    this:
      * one <testcase> per eval case, not per repetition -- the unit of meaning
        is "does this case hold", and repetitions are how we decide that
      * classname is "<suite>.<first tag>", so tools that group by class give
        you a breakdown by tag (arabic, injection, pii, ...)
      * FLAKY becomes a <failure type="flaky">, since JUnit has no flaky state
        and silently passing an intermittent case is the exact failure mode
        this harness exists to prevent
      * the reply and grader verdicts go in <system-out> so triage does not
        require opening the JSON
    """
    by_case = _runs_by_case(runs)
    cases = report["cases"]
    failures = sum(1 for c in cases if status_of(c) != "PASS")
    total_ms = sum(r.get("output", {}).get("latency_ms", 0) for r in runs)

    suites = ET.Element(
        "testsuites",
        name=suite_name,
        tests=str(len(cases)),
        failures=str(failures),
        time=f"{total_ms / 1000:.3f}",
    )
    suite = ET.SubElement(
        suites,
        "testsuite",
        name=suite_name,
        tests=str(len(cases)),
        failures=str(failures),
        errors="0",
        skipped="0",
        time=f"{total_ms / 1000:.3f}",
    )
    props = ET.SubElement(suite, "properties")
    for key in ("agent_model", "judge_model", "repetitions", "judged"):
        if report.get(key) not in (None, ""):
            ET.SubElement(props, "property", name=key, value=str(report[key]))

    for case in cases:
        case_runs = by_case.get(case["case_id"], [])
        case_ms = sum(r.get("output", {}).get("latency_ms", 0) for r in case_runs)
        tag = case["tags"][0] if case["tags"] else "untagged"
        node = ET.SubElement(
            suite,
            "testcase",
            classname=f"{suite_name}.{tag}",
            name=case["case_id"],
            time=f"{case_ms / 1000:.3f}",
        )

        status = status_of(case)
        if status != "PASS":
            kind = "flaky" if status == "FLAKY" else "assertion"
            headline = (
                f"{status}: passed {case['passes']} of {case['runs']} runs"
                if status == "FLAKY"
                else f"FAILED: passed {case['passes']} of {case['runs']} runs"
            )
            failure = ET.SubElement(node, "failure", message=headline, type=kind)
            failure.text = "\n".join(case["failures"]) or headline

        # Replies and verdicts, so a failure can be triaged from the XML alone.
        lines: list[str] = []
        for r in case_runs:
            out = r.get("output", {})
            lines.append(f"--- run {r.get('repetition', 0) + 1} ({out.get('latency_ms', 0)} ms)")
            lines.append(out.get("reply") or "(no reply)")
            bad = [g["name"] for g in r.get("deterministic", []) if not g["passed"]]
            if bad:
                lines.append(f"failed graders: {', '.join(bad)}")
            if r.get("judge"):
                j = r["judge"]
                lines.append(
                    f"judge: helpfulness={j['helpfulness']} faithfulness={j['faithfulness']} "
                    f"tone={j['tone']} -- {j['reasoning']}"
                )
        if lines:
            ET.SubElement(node, "system-out").text = "\n".join(lines)

    ET.indent(suites, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(suites, encoding="unicode")


# --------------------------------------------------------------------------
# Markdown -- for the CI run page
# --------------------------------------------------------------------------


def to_markdown(report: dict, runs: list[dict], baseline_messages: list[str] | None = None) -> str:
    cases = report["cases"]
    rate = suite_rate(cases)
    counts = {s: sum(1 for c in cases if status_of(c) == s) for s in ("PASS", "FAIL", "FLAKY")}
    icon = {"PASS": "✅", "FAIL": "❌", "FLAKY": "⚠️"}

    judged_note = (
        "Judged run — invariants and rubric scoring."
        if report.get("judged")
        else "**Deterministic-only run** — the judge did not score these replies, "
        "so subjective quality is unmeasured."
    )
    if "hand-authored" in report.get("source", ""):
        judged_note += (
            " Some fixtures are **hand-authored test doubles**: this run shows the "
            "graders work, not how the model behaves."
        )

    out = [
        "## LLM eval",
        "",
        f"**{rate:.0%} suite pass rate** — {counts['PASS']} passed, "
        f"{counts['FAIL']} failed, {counts['FLAKY']} flaky "
        f"across {len(cases)} cases × {report['repetitions']} run(s).",
        "",
        judged_note,
        "",
        f"agent `{report['agent_model']}`"
        + (f" · judge `{report['judge_model']}`" if report.get("judge_model") else ""),
        "",
        "| | Case | Tags | Passed | Why it failed |",
        "|---|---|---|---|---|",
    ]

    # Failures first: the point of a summary is what needs attention.
    for case in sorted(cases, key=lambda c: (status_of(c) == "PASS", c["case_id"])):
        status = status_of(case)
        why = case["failures"][0] if case["failures"] else ""
        if len(case["failures"]) > 1:
            why += f" (+{len(case['failures']) - 1} more)"
        why = why.replace("|", "\\|")[:160]
        out.append(
            f"| {icon[status]} | `{case['case_id']}` | {' '.join(case['tags'])} | "
            f"{case['passes']}/{case['runs']} | {why} |"
        )

    graders = report.get("graders") or []
    if graders:
        out += [
            "",
            "### Graders",
            "",
            "Pass rate over the runs each grader applied to; n/a runs are left out.",
            "",
            "| Grader | Passed | Rate |",
            "|---|---|---|",
        ]
        def rate(g: dict) -> float:
            return g["passed"] / g["applicable"] if g["applicable"] else 1.0
        for g in sorted(graders, key=lambda g: (rate(g), g["name"])):
            shown = f"{rate(g):.0%}" if g["applicable"] else "n/a"
            out.append(f"| `{g['name']}` | {g['passed']}/{g['applicable']} | {shown} |")

    if baseline_messages:
        out += ["", "### Baseline", "", "```"] + baseline_messages + ["```"]

    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# HTML -- for a human
# --------------------------------------------------------------------------

CSS = """
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #6b7280; --line: #e5e7eb;
  --pass: #15803d; --pass-bg: #dcfce7; --fail: #b91c1c; --fail-bg: #fee2e2;
  --flaky: #b45309; --flaky-bg: #fef3c7; --chip: #f3f4f6; --card: #fafafa;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115; --fg: #e5e7eb; --muted: #9ca3af; --line: #262b33;
    --pass: #4ade80; --pass-bg: #14321f; --fail: #f87171; --fail-bg: #3b1618;
    --flaky: #fbbf24; --flaky-bg: #3a2c0a; --chip: #1b2028; --card: #151922;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 24px 64px; background: var(--bg); color: var(--fg);
  font: 15px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif;
}
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; }
.sub { color: var(--muted); font-size: 13px; margin-bottom: 24px; }
.sub code { background: var(--chip); padding: 1px 6px; border-radius: 4px; }
.cards { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 28px; }
.card {
  background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 14px 18px; min-width: 120px; flex: 1;
}
.card .n { font-size: 26px; font-weight: 650; letter-spacing: -0.5px; }
.card .l { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .5px; }
.banner {
  border-radius: 10px; padding: 14px 18px; margin-bottom: 24px; font-weight: 600;
  border: 1px solid var(--line);
}
.banner.ok { background: var(--pass-bg); color: var(--pass); }
.banner.bad { background: var(--fail-bg); color: var(--fail); }
details {
  border: 1px solid var(--line); border-radius: 10px; margin-bottom: 10px;
  background: var(--card); overflow: hidden;
}
summary {
  cursor: pointer; padding: 12px 16px; display: grid; align-items: center; gap: 12px;
  grid-template-columns: 90px 1fr auto auto; list-style: none;
}
summary::-webkit-details-marker { display: none; }
summary:hover { background: var(--chip); }
.badge {
  font-size: 11px; font-weight: 700; letter-spacing: .6px; padding: 4px 8px;
  border-radius: 5px; text-align: center;
}
.badge.PASS { background: var(--pass-bg); color: var(--pass); }
.badge.FAIL { background: var(--fail-bg); color: var(--fail); }
.badge.FLAKY { background: var(--flaky-bg); color: var(--flaky); }
.cid { font-weight: 600; font-family: ui-monospace, "Cascadia Mono", monospace; font-size: 13px; }
.tags { display: flex; gap: 5px; flex-wrap: wrap; }
.tag { background: var(--chip); color: var(--muted); font-size: 11px; padding: 2px 7px; border-radius: 999px; }
.rate { font-variant-numeric: tabular-nums; color: var(--muted); font-size: 13px; }
.body { padding: 4px 16px 16px; border-top: 1px solid var(--line); }
.fail-line {
  background: var(--fail-bg); color: var(--fail); border-radius: 6px;
  padding: 8px 12px; margin: 10px 0 0; font-size: 13px;
  font-family: ui-monospace, "Cascadia Mono", monospace;
}
.run { border-top: 1px dashed var(--line); padding: 14px 0 4px; }
.run h4 { margin: 0 0 8px; font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; }
.reply {
  background: var(--bg); border: 1px solid var(--line); border-radius: 8px;
  padding: 10px 14px; white-space: pre-wrap; margin-bottom: 10px;
}
.graders { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }
.g { font-size: 11px; padding: 3px 8px; border-radius: 5px; background: var(--pass-bg); color: var(--pass); }
.g.no { background: var(--fail-bg); color: var(--fail); }
.g.na { background: var(--chip); color: var(--muted); }
.judge { font-size: 13px; color: var(--muted); }
.judge b { color: var(--fg); }
.meta { color: var(--muted); font-size: 12px; }
"""


def _chip(g: dict) -> str:
    if not g.get("applicable", True):
        return " na"  # nothing to check on this case: shown muted, not green
    return "" if g["passed"] else " no"


def _run_block(run: dict) -> str:
    out = run.get("output", {})
    graders = "".join(
        f'<span class="g{_chip(g)}">{_esc(g["name"])}</span>'
        for g in run.get("deterministic", [])
    )
    judge = run.get("judge")
    judge_html = ""
    if judge:
        judge_html = (
            f'<div class="judge">judge &mdash; helpfulness <b>{judge["helpfulness"]}</b>, '
            f'faithfulness <b>{judge["faithfulness"]}</b>, tone <b>{judge["tone"]}</b>: '
            f'{_esc(judge["reasoning"])}</div>'
        )
    tools = out.get("tool_calls") or []
    tool_html = ""
    if tools:
        names = ", ".join(
            f'{_esc(t.get("name", "?"))}({_esc(json.dumps(t.get("args", {})))})' for t in tools
        )
        tool_html = f'<div class="meta">tools: {names}</div>'
    err = f'<div class="fail-line">{_esc(out.get("error"))}</div>' if out.get("error") else ""
    # dir="auto" lets the browser pick RTL for Arabic replies and LTR for English.
    return (
        f'<div class="run"><h4>run {run.get("repetition", 0) + 1}'
        f' &middot; {out.get("latency_ms", 0)} ms</h4>'
        f'<div class="reply" dir="auto">{_esc(out.get("reply"))}</div>'
        f'<div class="graders">{graders}</div>{tool_html}{judge_html}{err}</div>'
    )


def to_html(report: dict, runs: list[dict], stamp: str) -> str:
    cases = report["cases"]
    by_case = _runs_by_case(runs)
    counts = {s: sum(1 for c in cases if status_of(c) == s) for s in ("PASS", "FAIL", "FLAKY")}
    rate = suite_rate(cases)

    rows = []
    for case in sorted(cases, key=lambda c: (status_of(c) == "PASS", c["case_id"])):
        status = status_of(case)
        tags = "".join(f'<span class="tag">{_esc(t)}</span>' for t in case["tags"])
        fails = "".join(f'<div class="fail-line">{_esc(f)}</div>' for f in case["failures"])
        detail = "".join(_run_block(r) for r in by_case.get(case["case_id"], []))
        rows.append(
            f"<details{' open' if status != 'PASS' else ''}>"
            f'<summary><span class="badge {status}">{status}</span>'
            f'<span class="cid">{_esc(case["case_id"])}</span>'
            f'<span class="tags">{tags}</span>'
            f'<span class="rate">{case["passes"]}/{case["runs"]}</span></summary>'
            f'<div class="body">{fails}{detail}</div></details>'
        )

    banner = (
        '<div class="banner ok">Judged run &mdash; deterministic invariants and rubric scoring.</div>'
        if report.get("judged")
        else '<div class="banner bad">Deterministic-only run &mdash; the judge did not score these '
        "replies, so subjective quality is unmeasured.</div>"
    )
    judge_bit = (
        f'&middot; judge <code>{_esc(report["judge_model"])}</code>'
        if report.get("judge_model")
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Eval report {stamp}</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>LLM eval report</h1>
<div class="sub">{stamp} &middot; agent <code>{_esc(report["agent_model"])}</code>
{judge_bit} &middot; {report["repetitions"]} repetition(s) per case</div>
{banner}
<div class="cards">
  <div class="card"><div class="n">{rate:.0%}</div><div class="l">suite pass rate</div></div>
  <div class="card"><div class="n">{len(cases)}</div><div class="l">cases</div></div>
  <div class="card"><div class="n">{counts["PASS"]}</div><div class="l">passed</div></div>
  <div class="card"><div class="n">{counts["FAIL"]}</div><div class="l">failed</div></div>
  <div class="card"><div class="n">{counts["FLAKY"]}</div><div class="l">flaky</div></div>
</div>
{"".join(rows)}
</div></body></html>
"""
