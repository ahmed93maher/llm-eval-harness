"""The figure check: every number in an answer must trace back to the data.

The technique, independent of any agent:

1. Extract every figure the answer states, across the formats a model writes:
   "$2,195.00", "2195", "2.2k", "62%", and Arabic-Indic digits ("١٬٠٠٠٫٢٥").
   Dates, years and list markers are numbers but not figures, and are skipped.
2. Collect what the agent was told: every number in its tool results (and in
   the arguments it sent and the question it was asked, which it may repeat).
3. Ground each figure: it must equal one of those values, a count of a list,
   or a plain derivation of the headline values -- a sum, a difference, a
   ratio or a percentage change.
4. The tolerance comes from the precision the answer stated: "41.9k" claims
   less than "41,891.00", so it is allowed to be off by more.

A figure that grounds nowhere was invented, or computed wrongly. That is the
failure this exists for: a figure that reads plausibly, in a confident answer.

What it cannot do: grounded is not the same as correct. "46 orders" grounds if
46 appears anywhere in the results, even against the wrong day or store. That
gap is closed by the path graders (the right arguments) and by the judge.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import permutations
from typing import Any

# Arabic-Indic and Eastern Arabic-Indic digits, plus the Arabic thousands (٬)
# and decimal (٫) separators.
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٬٫", "01234567890123456789,.")

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
    "|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
    "|يناير|فبراير|مارس|أبريل|ابريل|مايو|يونيو|يوليو|أغسطس|اغسطس|سبتمبر|أكتوبر|اكتوبر|نوفمبر|ديسمبر"
)
_DAY = r"\d{1,2}(?:st|nd|rd|th)?"
_NOT_FIGURES = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),  # 2026-10-14
    re.compile(rf"\b{_DAY}\s*(?:to|-|–|and|و|إلى|الى)\s*{_DAY}\s+(?:of\s+)?(?:{_MONTHS})\b", re.I),  # 1 to 7 October
    re.compile(rf"\b{_DAY}\s+(?:of\s+)?(?:{_MONTHS})\b", re.I),  # 14 October
    re.compile(rf"\b(?:{_MONTHS})\s+{_DAY}\b", re.I),  # October 14
    re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b"),  # 14/10, 14/10/2026
    re.compile(r"\b20\d{2}\b"),  # a year
    re.compile(r"^\s*\d{1,2}[.)]\s", re.M),  # a numbered list marker
]

_FIGURE = re.compile(
    r"(?:[$€£]\s?)?(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(k|m|thousand|million|ألف|مليون)?(?!\w)\s*(%|٪|percent)?",
    re.I,
)
_SCALE = {"k": 1e3, "thousand": 1e3, "ألف": 1e3, "m": 1e6, "million": 1e6, "مليون": 1e6}


@dataclass(frozen=True)
class Figure:
    text: str  # as written, for the failure message
    value: float
    tolerance: float  # half a unit of the last digit stated
    percent: bool
    rounded: bool = False  # "2,200": trailing zeros widened the tolerance
    decimals: bool = False  # written with a fraction: "47.72", "1.5"


def normalise(text: str) -> str:
    return text.translate(_DIGITS)


def extract(text: str) -> list[Figure]:
    """Every figure an answer states, with the tolerance its precision implies."""
    text = normalise(text)
    for pattern in _NOT_FIGURES:
        text = pattern.sub(" ", text)
    figures = []
    for m in _FIGURE.finditer(text):
        whole, frac, scale_word, pct = m.groups()
        digits = whole.replace(",", "")
        decimals = len(frac) - 1 if frac else 0
        scale = _SCALE.get((scale_word or "").lower(), 1.0)
        value = float(digits + (frac or "")) * scale
        tolerance = 0.5 * 10 ** -decimals * scale
        # "2,200" for 2,195 is rounding, not a wrong figure: trailing zeros in a
        # large whole number mean less precision. Two limits, both policy:
        # small counts stay exact ("50 orders" for 46 is wrong), and a rounded
        # figure keeps at least 2 significant figures ("$20,000" for $15,837
        # misleads, even if it is a 1-significant-figure rounding).
        rounded = False
        if not frac and scale == 1.0 and value >= 1000:
            zeros = min(len(digits) - len(digits.rstrip("0")), len(digits) - 2)
            rounded = zeros > 0
            tolerance = max(tolerance, 0.5 * 10**zeros)
        figures.append(Figure(m.group().strip(), value, tolerance, bool(pct), rounded or scale > 1, bool(frac)))
    return figures


# --- what the agent was told -------------------------------------------------


def _walk(node: Any, values: list[float], headline: list[float], counts: list[float], in_big_list: bool) -> None:
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        values.append(float(node))
        if not in_big_list:
            headline.append(float(node))
    elif isinstance(node, dict):
        for v in node.values():
            _walk(v, values, headline, counts, in_big_list)
    elif isinstance(node, list):
        counts.append(float(len(node)))
        big = in_big_list or len(node) > 10
        for item in node:
            _walk(item, values, headline, counts, big)
        # A total over a list's items, per field: "the top three made $6,637".
        if node and all(isinstance(i, dict) for i in node):
            for key in node[0]:
                nums = [i.get(key) for i in node]
                if all(isinstance(n, (int, float)) and not isinstance(n, bool) for n in nums):
                    values.append(float(sum(nums)))


def _parse(result: Any) -> Any:
    if isinstance(result, str):
        try:
            return json.loads(result)
        except ValueError:
            return None  # an error message: words, not data
    return result


@dataclass
class Ground:
    values: set[float]  # stated as-is: raw values, counts, list totals, known facts
    percentages: set[float]  # ratios and changes, as percentages
    derived: set[float]  # sums and differences of headline values
    ratios: set[float]  # plain ratios: averages such as revenue per order


def ground(tool_calls: list[dict[str, Any]], message: str = "", known: tuple[float, ...] = ()) -> Ground:
    """Everything the agent can honestly state. `known` holds facts it was given
    outside any tool call, such as the number of stores in its prompt."""
    values: list[float] = []
    headline: list[float] = []
    counts: list[float] = []
    per_call_totals: dict[str, list[float]] = {}

    for call in tool_calls:
        if call.get("is_error"):
            continue
        data = _parse(call.get("result"))
        if data is None:
            continue
        _walk(data, values, headline, counts, False)
        _walk(call.get("args", {}), values, headline, counts, False)
        # The same field across several calls, summed: harbor + mill + online.
        if isinstance(data, dict):
            for key, v in (data.get("totals") or {}).items():
                if isinstance(v, (int, float)):
                    per_call_totals.setdefault(key, []).append(float(v))

    values += [f.value for f in extract(message)] + counts + [float(k) for k in known]
    for totals in per_call_totals.values():
        if len(totals) > 1:
            values.append(sum(totals))
            headline.append(sum(totals))  # so "Online was 50% of the total" grounds too

    heads = sorted(set(headline))[:200]  # bound the pairwise work
    derived, percentages, ratios = set(), set(), set()
    for a, b in permutations(heads, 2):
        derived.add(a + b)
        derived.add(a - b)
        if b:
            ratios.add(a / b)
            percentages.add(a / b * 100)
            percentages.add((a - b) / b * 100)
    return Ground(set(values), percentages, derived, ratios)


def grounded(fig: Figure, g: Ground) -> bool:
    """Match a figure against what the agent was told, guarding against coincidence.

    The chance of a false match grows with the size of the pool times the width
    of the tolerance. Hundreds of derived sums and a tolerance of +/-500 will
    "ground" almost any number, so each kind of figure is only compared with
    the pool it can honestly come from:
      - a percentage only with computed percentages (55% is not the 55 orders
        some day had);
      - a rounded figure ("2,200", "2.2k") only with raw values, not with
        derivations, because its tolerance is wide;
      - a ratio (an average) only for a figure written with decimals, so that
        "47 orders" cannot ground on an average of 47.7.
    """

    def near(pool: set[float]) -> bool:
        return any(abs(fig.value - v) <= fig.tolerance + 1e-9 for v in pool)

    if fig.percent:
        return near(g.percentages)
    if fig.rounded:
        return near(g.values)
    return near(g.values) or near(g.derived) or (fig.decimals and near(g.ratios))


def ungrounded(
    answer: str, tool_calls: list[dict[str, Any]], message: str = "", known: tuple[float, ...] = ()
) -> list[str]:
    """The figures in an answer that trace to nothing the agent was told."""
    g = ground(tool_calls, message, known)
    return [f.text for f in extract(answer) if not grounded(f, g)]
