"""Benchmark: buy the whole watchlist in equal amounts once, then never trade."""

from __future__ import annotations

from ..risk import Intent
from .base import Brain, Context, Decision


class BuyAndHold(Brain):
    kind = "hold"
    benchmark = True

    def decide(self, ctx: Context) -> Decision:
        if ctx.account.meta.get("bought"):
            return Decision()
        symbols = [s for s in ctx.market.symbols if s in ctx.prices]
        if not symbols:
            return Decision()
        # Wait (up to a few ticks) for prices on the whole watchlist.
        waited = ctx.account.meta.get("waited", 0)
        if len(symbols) < len(ctx.market.symbols) and waited < 3:
            ctx.account.meta["waited"] = waited + 1
            return Decision(note="waiting for prices on the whole watchlist")
        ctx.account.meta["bought"] = True
        pct = 99.0 / len(symbols)  # leave ~1% for costs
        return Decision(
            intents=[Intent(s, "buy", pct, "benchmark: equal-weight buy and hold") for s in symbols],
            note="buying the watchlist once",
        )
