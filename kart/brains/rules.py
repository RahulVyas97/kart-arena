"""Classic trend-following bot: no AI, just moving averages and RSI."""

from __future__ import annotations

from ..indicators import rsi, sma
from ..risk import Intent
from .base import Brain, Context, Decision


class TrendRules(Brain):
    kind = "rules"

    def __init__(self, name: str, params: dict):
        super().__init__(name, params)
        self.fast = int(params.get("fast", 8))
        self.slow = int(params.get("slow", 21))
        self.rsi_max = float(params.get("rsi_max", 70))
        self.stop_loss = float(params.get("stop_loss_pct", 2.0)) / 100
        self.take_profit = float(params.get("take_profit_pct", 4.0)) / 100
        self.position_pct = float(params.get("position_pct", 15))
        if self.fast >= self.slow:
            raise ValueError(f"{name}: fast ({self.fast}) must be smaller than slow ({self.slow})")

    def decide(self, ctx: Context) -> Decision:
        intents: list[Intent] = []
        for sym, df in ctx.bars.items():
            if len(df) < self.slow + 2 or sym not in ctx.prices:
                continue
            close = df["close"]
            f, s = sma(close, self.fast), sma(close, self.slow)
            price = ctx.prices[sym]
            pos = ctx.account.positions.get(sym)
            if pos:
                change = price / pos.avg_cost - 1
                if change <= -self.stop_loss:
                    intents.append(Intent(sym, "sell", 100, f"stop-loss ({change * 100:.1f}%)"))
                elif change >= self.take_profit:
                    intents.append(Intent(sym, "sell", 100, f"take-profit (+{change * 100:.1f}%)"))
                elif f.iloc[-1] < s.iloc[-1]:
                    intents.append(Intent(sym, "sell", 100, "trend turned down (fast MA below slow MA)"))
            else:
                crossed_up = f.iloc[-2] <= s.iloc[-2] and f.iloc[-1] > s.iloc[-1]
                r = rsi(close).iloc[-1]
                if crossed_up and (r != r or r < self.rsi_max):  # r != r means NaN
                    intents.append(Intent(sym, "buy", self.position_pct, "fast MA crossed above slow MA"))
        return Decision(intents=intents, note=f"{len(intents)} signal(s)" if intents else "")
