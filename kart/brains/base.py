"""What every brain receives and returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from ..config import RiskCfg
from ..markets import Market
from ..portfolio import Account
from ..risk import Intent


@dataclass
class Context:
    now: datetime
    market: Market
    account: Account
    bars: dict[str, pd.DataFrame]
    prices: dict[str, float]
    minutes_to_close: float
    interval_minutes: int
    risk: RiskCfg
    round_trip_cost_pct: float
    headlines: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Decision:
    intents: list[Intent] = field(default_factory=list)
    note: str = ""
    error: str = ""


class Brain:
    kind = "base"
    benchmark = False      # benchmarks skip risk limits and the profit target
    news_per_symbol = 0    # >0 asks the arena for public headlines (live mode only)

    def __init__(self, name: str, params: dict):
        self.name = name
        self.params = params

    def decide(self, ctx: Context) -> Decision:  # pragma: no cover
        raise NotImplementedError
