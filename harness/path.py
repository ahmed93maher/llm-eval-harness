"""Path checks: did the agent ask its tools for the right thing?

The figure check proves a number came from the data; it cannot tell whether it
came from the right slice of it. "46 orders" grounds just as well against the
wrong day. The path is where that error happens, so the path gets its own
checks.

The hard part is that most questions have more than one valid path. "Last
week" may mean Monday to Sunday or the last seven days; "revenue yesterday" may
be one call for all stores, or one per store, added up. A path expectation is
therefore a small tree of requirements:

    {"tool": "get_sales", "store_id": "mill"}   a call with these arguments
    {"any_of": [req, req, ...]}                 one of these paths
    {"all_of": [req, req, ...]}                 every one of these

A list at the top level means all_of. A call spec lists only the arguments that
matter; any argument it leaves out may take any value.
"""
from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

Requirement = dict[str, Any]

_MONTH_NUMBERS = {
    **{m: i for i, m in enumerate(
        "january february march april may june july august september october november december".split(), 1)},
    **{m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)},
    "sept": 9,
    **{m: i for i, m in enumerate(
        "يناير فبراير مارس أبريل مايو يونيو يوليو أغسطس سبتمبر أكتوبر نوفمبر ديسمبر".split(), 1)},
    "ابريل": 4, "اغسطس": 8, "اكتوبر": 10,
}
_MONTH = "|".join(sorted(_MONTH_NUMBERS, key=len, reverse=True))
_WEEKDAY = r"(?:(?:mon|tues?|wed(?:nes)?|thu(?:rs)?|fri|sat(?:ur)?|sun)(?:day)?\.?,?\s+)?"
_DAY = r"(\d{1,2})(?:st|nd|rd|th)?"
_TO = r"\s*(?:-|–|—|to|until|through|and|إلى|الى|حتى)\s*"
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _key(call: dict[str, Any]) -> str:
    return json.dumps([call.get("name"), call.get("args", {})], sort_keys=True, default=str)


def _matches(call: dict[str, Any], spec: Requirement) -> bool:
    if call.get("is_error") or call.get("name") != spec["tool"]:
        return False
    args = call.get("args", {})
    return all(str(args.get(k)) == str(v) for k, v in spec.items() if k != "tool")


def satisfied(req: Requirement | list, calls: list[dict[str, Any]]) -> bool:
    """Whether the calls made satisfy a requirement tree. Failed calls never count."""
    if isinstance(req, list):
        return all(satisfied(r, calls) for r in req)
    if "any_of" in req:
        return any(satisfied(r, calls) for r in req["any_of"])
    if "all_of" in req:
        return all(satisfied(r, calls) for r in req["all_of"])
    return any(_matches(c, req) for c in calls)


def describe(req: Requirement | list) -> str:
    if isinstance(req, list):
        return " and ".join(describe(r) for r in req)
    if "any_of" in req:
        return "(" + " or ".join(describe(r) for r in req["any_of"]) + ")"
    if "all_of" in req:
        return "(" + " and ".join(describe(r) for r in req["all_of"]) + ")"
    args = ", ".join(f"{k}={v}" for k, v in req.items() if k != "tool")
    return f"{req['tool']}({args})"


def tools_named(req: Requirement | list) -> set[str]:
    if isinstance(req, list):
        return set().union(*(tools_named(r) for r in req)) if req else set()
    for key in ("any_of", "all_of"):
        if key in req:
            return set().union(*(tools_named(r) for r in req[key]))
    return {req["tool"]}


def repeated(calls: list[dict[str, Any]]) -> list[str]:
    """Calls made more than once with identical arguments: wasted work, or a retry loop."""
    seen: set[str] = set()
    dupes = []
    for c in calls:
        k = _key(c)
        if k in seen:
            dupes.append(f"{c.get('name')}({json.dumps(c.get('args', {}), sort_keys=True, default=str)})")
        seen.add(k)
    return dupes


def mentioned_dates(text: str, year: int) -> set[date]:
    """The calendar dates an answer names: "5 October", "Oct 5", "5-11 October",
    "Monday 5 to Sunday 11 October", "2026-10-05", and the Arabic forms."""
    text = text.translate(_DIGITS).lower()
    found: set[date] = set()

    def add(day: str, month: str) -> None:
        try:
            found.add(date(year, _MONTH_NUMBERS[month], int(day)))
        except (KeyError, ValueError):
            pass

    for y, m, d in re.findall(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        try:
            found.add(date(int(y), int(m), int(d)))
        except ValueError:
            pass
    for a, b, month in re.findall(rf"{_DAY}{_TO}{_WEEKDAY}{_DAY}\s+(?:of\s+)?({_MONTH})\b", text):
        add(a, month)
        add(b, month)
    for month, a, b in re.findall(rf"\b({_MONTH})\s+{_DAY}{_TO}{_DAY}\b", text):
        add(a, month)
        add(b, month)
    for day, month in re.findall(rf"{_DAY}\s+(?:of\s+)?({_MONTH})\b", text):
        add(day, month)
    for month, day in re.findall(rf"\b({_MONTH})\s+{_DAY}\b", text):
        add(day, month)
    return found
