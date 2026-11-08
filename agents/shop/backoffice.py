"""The fake back office of "Tidewater Goods", a fictional three-store shop.

The data is generated from a fixed seed, so every run, every machine and every
test sees exactly the same figures. That is what lets a grader say "this
number is wrong": the ground truth never moves.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from functools import lru_cache

SEED = 20261015
TODAY = date(2026, 10, 15)  # the agent is told this is "today"
FIRST_DAY = date(2026, 9, 1)
LAST_DAY = date(2026, 10, 14)  # data runs through yesterday

STORES = {
    "harbor": "Harbor Street",
    "mill": "Mill Lane",
    "online": "Online Store",
}

# product id -> (name, unit price)
PRODUCTS = {
    "mug": ("Stoneware Mug", 14.00),
    "tote": ("Canvas Tote", 22.50),
    "candle": ("Cedar Candle", 18.00),
    "notebook": ("Linen Notebook", 9.75),
    "blanket": ("Wool Throw Blanket", 89.00),
    "teapot": ("Cast Iron Teapot", 54.00),
    "socks": ("Merino Socks", 16.50),
    "print": ("Harbor Art Print", 35.00),
}

# Rough daily demand per store, so stores differ in a way an answer can get wrong.
_STORE_SCALE = {"harbor": 1.0, "mill": 0.6, "online": 1.6}


def days() -> list[date]:
    n = (LAST_DAY - FIRST_DAY).days + 1
    return [FIRST_DAY + timedelta(d) for d in range(n)]


@lru_cache(maxsize=1)
def unit_sales() -> dict[tuple[str, date, str], int]:
    """Units sold per (store, day, product)."""
    rng = random.Random(SEED)
    sales: dict[tuple[str, date, str], int] = {}
    for store, scale in _STORE_SCALE.items():
        for day in days():
            weekend = 1.4 if day.weekday() >= 5 else 1.0
            for product in PRODUCTS:
                mean = 3.0 * scale * weekend
                sales[(store, day, product)] = max(0, round(rng.gauss(mean, mean / 2)))
    return sales


@lru_cache(maxsize=1)
def orders() -> dict[tuple[str, date], int]:
    """Orders per (store, day): each order holds one or more units."""
    rng = random.Random(SEED + 1)
    result: dict[tuple[str, date], int] = {}
    for store in STORES:
        for day in days():
            units = sum(unit_sales()[(store, day, p)] for p in PRODUCTS)
            result[(store, day)] = min(units, max(1, round(units / rng.uniform(1.3, 2.1)))) if units else 0
    return result
