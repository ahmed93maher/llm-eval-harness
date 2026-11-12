"""The agent's tools: their definitions (what the model sees) and their
implementations (what actually runs).

The tools are deliberately strict and return errors as data. An agent's
handling of a tool error (does it retry sensibly, or invent a number?) is one
of the behaviours worth measuring.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from agents.shop import backoffice as data

_DATE = {"type": "string", "description": "ISO date, YYYY-MM-DD"}
_STORE = {
    "type": "string",
    "enum": [*data.STORES, "all"],
    "description": "A store id from list_stores, or 'all' for every store combined.",
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_stores",
        "description": "List the shop's stores with their ids and names.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "get_sales",
        "description": (
            "Daily sales of one store, or all stores combined, over a date range (inclusive). "
            "Returns each day's orders, units and revenue, plus the totals for the range. "
            f"Data exists from {data.FIRST_DAY} to {data.LAST_DAY}."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"store_id": _STORE, "start_date": _DATE, "end_date": _DATE},
            "required": ["store_id", "start_date", "end_date"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "get_top_products",
        "description": (
            "The best-selling products of one store, or all stores, over a date range "
            "(inclusive), ranked by revenue. Returns units and revenue per product."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": _STORE,
                "start_date": _DATE,
                "end_date": _DATE,
                "limit": {"type": "integer", "description": "How many products, 1 to 8."},
            },
            "required": ["store_id", "start_date", "end_date", "limit"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


class ToolError(Exception):
    """A tool refused its input. Sent back to the model as an error result."""


def _range(args: dict[str, Any]) -> tuple[list[str], list[date]]:
    try:
        start = date.fromisoformat(args["start_date"])
        end = date.fromisoformat(args["end_date"])
    except ValueError as exc:
        raise ToolError(f"Invalid date: {exc}") from exc
    if start > end:
        raise ToolError("start_date is after end_date.")
    if start < data.FIRST_DAY or end > data.LAST_DAY:
        raise ToolError(f"No data outside {data.FIRST_DAY} to {data.LAST_DAY}.")
    stores = list(data.STORES) if args["store_id"] == "all" else [args["store_id"]]
    return stores, [d for d in data.days() if start <= d <= end]


def _revenue(store: str, day: date, product: str) -> float:
    return data.unit_sales()[(store, day, product)] * data.PRODUCTS[product][1]


def list_stores(_: dict[str, Any]) -> list[dict[str, str]]:
    return [{"id": sid, "name": name} for sid, name in data.STORES.items()]


def get_sales(args: dict[str, Any]) -> dict[str, Any]:
    stores, days = _range(args)
    rows = []
    for day in days:
        rows.append({
            "date": day.isoformat(),
            "orders": sum(data.orders()[(s, day)] for s in stores),
            "units": sum(data.unit_sales()[(s, day, p)] for s in stores for p in data.PRODUCTS),
            "revenue": round(sum(_revenue(s, day, p) for s in stores for p in data.PRODUCTS), 2),
        })
    totals = {
        "orders": sum(r["orders"] for r in rows),
        "units": sum(r["units"] for r in rows),
        "revenue": round(sum(r["revenue"] for r in rows), 2),
    }
    return {"store_id": args["store_id"], "currency": "USD", "days": rows, "totals": totals}


def get_top_products(args: dict[str, Any]) -> dict[str, Any]:
    stores, days = _range(args)
    limit = args["limit"]
    if not 1 <= limit <= len(data.PRODUCTS):
        raise ToolError(f"limit must be between 1 and {len(data.PRODUCTS)}.")
    ranked = []
    for pid, (name, _) in data.PRODUCTS.items():
        units = sum(data.unit_sales()[(s, d, pid)] for s in stores for d in days)
        revenue = round(sum(_revenue(s, d, pid) for s in stores for d in days), 2)
        ranked.append({"product": name, "units": units, "revenue": revenue})
    ranked.sort(key=lambda r: r["revenue"], reverse=True)
    return {"store_id": args["store_id"], "currency": "USD", "products": ranked[:limit]}


_IMPLEMENTATIONS = {"list_stores": list_stores, "get_sales": get_sales, "get_top_products": get_top_products}


def execute(name: str, args: dict[str, Any]) -> tuple[Any, bool]:
    """Run a tool. Returns (result, is_error); never raises for bad input."""
    impl = _IMPLEMENTATIONS.get(name)
    if impl is None:
        return f"Unknown tool: {name}", True
    try:
        return impl(args), False
    except ToolError as exc:
        return str(exc), True
