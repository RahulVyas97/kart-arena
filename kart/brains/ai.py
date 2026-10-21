"""The AI contestant: a model reads a market snapshot and proposes orders.

It gets public data only (prices, indicators, its own account, and optional
public headlines). It never sees the profit target, and the risk manager
decides what actually trades. Where the answers come from is pluggable:
an API key, your Claude subscription, or a free local model (see backends.py).
"""

from __future__ import annotations

from ..indicators import snapshot
from ..risk import Intent
from .backends import BackendError, make_backend
from .base import Brain, Context, Decision

# Plain-text descriptions (no % signs) so the schema passes safely on any command line.
ORDER_SCHEMA = {
    "type": "object",
    "properties": {
        "market_view": {
            "type": "string",
            "description": "One or two sentences on what you see and why you're acting (or not).",
        },
        "orders": {
            "type": "array",
            "description": "Orders for this tick. Empty list = hold everything.",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker exactly as shown in the snapshot."},
                    "action": {"type": "string", "enum": ["buy", "sell"]},
                    "percent": {
                        "type": "number",
                        "description": "buy: percent of total account equity to spend, 1 to 100. "
                        "sell: percent of the current holding to sell, 1 to 100.",
                    },
                    "reason": {"type": "string", "description": "Short evidence for this order."},
                },
                "required": ["symbol", "action", "percent", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["market_view", "orders"],
    "additionalProperties": False,
}


def _f(value, fmt: str = "{:+.2f}", none: str = "n/a") -> str:
    return none if value is None else fmt.format(value)


class AITrader(Brain):
    kind = "ai"

    def __init__(self, name: str, params: dict, client=None, usage: dict | None = None,
                 runner=None, opener=None, workdir: str | None = None):
        super().__init__(name, params)
        self.news_per_symbol = int(params.get("news_per_symbol", 2))
        self.usage = usage if usage is not None else {}
        self.backend = make_backend(params, client=client, runner=runner, opener=opener, workdir=workdir)
        self.model = getattr(self.backend, "model", "")
        self.disabled_reason = self.backend.reason

    @property
    def available(self) -> bool:
        return not self.backend.reason

    @property
    def describe(self) -> str:
        return f"{self.backend.label}, {self.model}"

    # ------------------------------------------------------------------
    def decide(self, ctx: Context) -> Decision:
        if not self.available:
            return Decision(error=self.disabled_reason)
        try:
            data, usage = self.backend.ask(self.system_prompt(ctx), self.snapshot_prompt(ctx), ORDER_SCHEMA)
        except BackendError as exc:
            if len(exc.args) > 1 and isinstance(exc.args[1], dict):
                self._track_usage(exc.args[1])
            return Decision(error=f"{exc.args[0]}; holding")
        self._track_usage(usage)

        intents = []
        for o in data.get("orders") or []:
            try:
                intents.append(
                    Intent(
                        symbol=str(o["symbol"]),
                        action=str(o["action"]).lower(),
                        pct=float(o["percent"]),
                        reason=str(o.get("reason", ""))[:300],
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        note = str(data.get("market_view", ""))[:500]
        self._remember(ctx, note, intents)
        return Decision(intents=intents, note=note)

    # ------------------------------------------------------------------
    def system_prompt(self, ctx: Context) -> str:
        r = ctx.risk
        return f"""You manage a PAPER-TRADING account (simulated money) in the {ctx.market.name} stock market, \
in {ctx.market.currency}. You compete against a rules-based bot and a buy-and-hold benchmark. \
Every {ctx.interval_minutes} minutes while the market is open you get a snapshot and decide: buy, sell, or hold.

How the account works:
- Long only, cash only: no shorting, no leverage, whole shares only.
- Trading costs about {ctx.round_trip_cost_pct:.2f}% per round trip (fees, taxes, slippage). Small frequent \
trades usually lose to costs. Holding is a perfectly good decision.
- A risk manager you can't override enforces: max {r.max_position_pct:g}% of equity per stock, max \
{r.max_open_positions} stocks, max {r.max_trades_per_day} trades per day, no new buys in the last \
{r.no_new_buys_before_close_min} minutes before close, a daily loss limit and a drawdown stop. Orders \
that break a limit are clamped or rejected.
- Aim for good returns after costs over weeks, not today. Avoid big bets on noise.
- Use only the data in the snapshot. Don't assume prices or news you weren't given.

Reply in JSON with market_view (1-2 sentences) and orders (empty list to hold). For buys, percent is % of \
total account equity to spend. For sells, percent is % of the current holding to sell."""

    def snapshot_prompt(self, ctx: Context) -> str:
        acct, prices = ctx.account, ctx.prices
        equity = acct.equity(prices)
        local = ctx.now.astimezone(ctx.market.tz)
        lines = [
            f"Time: {local:%a %Y-%m-%d %H:%M} {ctx.market.tz.key} (market closes in {ctx.minutes_to_close:.0f} min)",
            f"Account ({acct.currency}): equity {equity:,.2f} | cash {acct.cash:,.2f} | "
            f"positions {len(acct.positions)}/{ctx.risk.max_open_positions} | "
            f"trades today {acct.trades_today}/{ctx.risk.max_trades_per_day}",
        ]
        if acct.paused_today:
            lines.append("Status: paused for new buys today (daily loss limit). Sells still allowed.")
        lines.append("Positions:" if acct.positions else "Positions: none")
        for sym, pos in acct.positions.items():
            px = prices.get(sym, pos.avg_cost)
            lines.append(
                f"- {sym}: {pos.qty} sh, avg cost {pos.avg_cost:,.2f}, now {px:,.2f} "
                f"({(px / pos.avg_cost - 1) * 100:+.2f}%), {pos.qty * px / equity * 100:.1f}% of equity"
            )
        lines.append(f"\nWatchlist ({ctx.interval_minutes}-min bars; trend = 8-bar vs 21-bar moving average):")
        lines.append("symbol | price | 1h % | vs prev close % | trend | RSI14 | volatility % | volume vs avg")
        for sym in ctx.market.symbols:
            df = ctx.bars.get(sym)
            if df is None or len(df) < 2:
                lines.append(f"{sym} | no data")
                continue
            s = snapshot(df, ctx.market.tz, ctx.interval_minutes)
            lines.append(
                f"{sym} | {s['price']:,.2f} | {_f(s['chg_1h'])} | {_f(s['chg_day'])} | {s['trend'] or 'n/a'} | "
                f"{_f(s['rsi'], '{:.0f}')} | {_f(s['volatility'], '{:.2f}')} | {_f(s['volume_x'], '{:.1f}x')}"
            )
        news = {k: v for k, v in (ctx.headlines or {}).items() if v}
        if news:
            lines.append("\nRecent public headlines:")
            for sym, items in news.items():
                for h in items:
                    lines.append(f"- {sym}: {h}")
        notes = acct.meta.get("ai_notes", [])
        if notes:
            lines.append("\nYour recent notes:")
            for n in notes:
                lines.append(f"- {n['t']}: {n['note']} (orders: {n['orders'] or 'none'})")
        lines.append("\nDecide now.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _remember(self, ctx: Context, note: str, intents: list[Intent]) -> None:
        notes = ctx.account.meta.setdefault("ai_notes", [])
        local = ctx.now.astimezone(ctx.market.tz)
        notes.append(
            {
                "t": f"{local:%a %H:%M}",
                "note": note[:200],
                "orders": ", ".join(f"{i.action} {i.symbol} {i.pct:g}%" for i in intents),
            }
        )
        del notes[:-3]

    def _track_usage(self, usage: dict) -> None:
        stats = self.usage.setdefault(self.name, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0})
        stats["backend"] = self.backend.name
        stats["calls"] += 1
        stats["input_tokens"] += int(usage.get("in", 0))
        stats["output_tokens"] += int(usage.get("out", 0))
        stats["usd"] = round(stats["usd"] + float(usage.get("usd") or 0.0), 6)


ClaudeTrader = AITrader  # old name, kept for compatibility
