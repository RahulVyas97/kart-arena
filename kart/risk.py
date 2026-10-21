"""Hard risk limits. Brains only propose; this decides what actually trades."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import RiskCfg
from .markets import Market
from .portfolio import Account, PaperBroker


@dataclass
class Intent:
    """What a brain wants: buy `pct`% of equity, or sell `pct`% of a holding."""

    symbol: str
    action: str
    pct: float
    reason: str = ""


@dataclass
class Order:
    symbol: str
    side: str
    qty: int
    reason: str


class RiskManager:
    def __init__(self, cfg: RiskCfg):
        self.cfg = cfg

    def roll_day(self, acct: Account, local_day: str, total: float) -> None:
        if acct.day != local_day:
            acct.day = local_day
            acct.day_start_total = total
            acct.trades_today = 0
            if acct.paused_today:
                acct.paused_today = False
                acct.status_note = ""
        acct.peak_total = max(acct.peak_total or total, total)

    def check_stops(self, acct: Account, total: float) -> None:
        """Updates paused/stopped flags from the drawdown and daily-loss limits."""
        if acct.stopped:
            return
        drawdown = (1 - total / acct.peak_total) * 100 if acct.peak_total else 0
        if drawdown >= self.cfg.max_drawdown_pct:
            acct.stopped = True
            acct.status_note = f"stopped: {drawdown:.1f}% below peak (limit {self.cfg.max_drawdown_pct}%)"
            return
        day_loss = (1 - total / acct.day_start_total) * 100 if acct.day_start_total else 0
        if not acct.paused_today and day_loss >= self.cfg.daily_loss_limit_pct:
            acct.paused_today = True
            acct.status_note = f"paused today: down {day_loss:.1f}% (limit {self.cfg.daily_loss_limit_pct}%)"

    def target_hit(self, acct: Account, equity: float) -> bool:
        if self.cfg.on_target == "ignore" or self.cfg.target_return_pct <= 0:
            return False
        return equity >= acct.start_capital * (1 + self.cfg.target_return_pct / 100)

    def vet(
        self,
        intents: list[Intent],
        acct: Account,
        prices: dict[str, float],
        market: Market,
        minutes_to_close: float,
        broker: PaperBroker,
        benchmark: bool = False,
    ) -> tuple[list[Order], list[dict]]:
        cfg = self.cfg
        canon = {s.upper(): s for s in market.symbols}
        orders: list[Order] = []
        rejected: list[dict] = []
        seen: set[str] = set()
        equity = acct.equity(prices)
        cash = acct.cash
        open_positions = len(acct.positions)
        trades_left = math.inf if benchmark else cfg.max_trades_per_day - acct.trades_today

        def reject(it: Intent, why: str) -> None:
            rejected.append({"symbol": it.symbol, "action": it.action, "pct": it.pct, "why": why})

        # Sells first: they free up cash and slots for buys in the same tick.
        for it in sorted(intents, key=lambda i: 0 if str(i.action).lower() == "sell" else 1):
            action = str(it.action).lower()
            sym = canon.get(str(it.symbol).strip().upper())
            if sym is None:
                reject(it, "not on this market's watchlist")
                continue
            if sym in seen:
                reject(it, "only one order per stock per tick")
                continue
            seen.add(sym)
            price = prices.get(sym)
            if not price or price <= 0:
                reject(it, "no current price")
                continue
            try:
                pct = float(it.pct)
            except (TypeError, ValueError):
                pct = float("nan")
            if math.isnan(pct) or pct <= 0:
                reject(it, "percent must be above 0")
                continue
            if trades_left <= 0:
                reject(it, f"daily trade limit ({cfg.max_trades_per_day}) reached")
                continue

            if action == "sell":
                pos = acct.positions.get(sym)
                if not pos:
                    reject(it, "nothing to sell")
                    continue
                qty = pos.qty if pct >= 99.5 else max(1, int(pos.qty * min(pct, 100) / 100))
                orders.append(Order(sym, "sell", qty, it.reason))
                trades_left -= 1
                cash += qty * price * (1 - broker.slip) * (1 - broker.sell_cost)
                if qty == pos.qty:
                    open_positions -= 1
                continue

            if action != "buy":
                reject(it, f"unknown action '{it.action}'")
                continue

            note = ""
            if not benchmark:
                if acct.stopped:
                    reject(it, "account stopped")
                    continue
                if acct.paused_today:
                    reject(it, "paused for the day (daily loss limit)")
                    continue
                if minutes_to_close <= cfg.no_new_buys_before_close_min:
                    reject(it, f"no new buys in the last {cfg.no_new_buys_before_close_min} min before close")
                    continue
                pos = acct.positions.get(sym)
                held_pct = (pos.qty * price / equity * 100) if (pos and equity > 0) else 0.0
                room = cfg.max_position_pct - held_pct
                if room <= 0.1:
                    reject(it, f"already at the {cfg.max_position_pct}% position limit")
                    continue
                if pos is None and open_positions >= cfg.max_open_positions:
                    reject(it, f"already holding the max {cfg.max_open_positions} stocks")
                    continue
                if pct > room:
                    note = f" [clamped from {pct:.0f}% to {room:.0f}%]"
                    pct = room

            spend = min(equity * pct / 100, cash)
            qty = int(spend // broker.unit_buy_cost(price))
            if qty < 1:
                reject(it, "not enough for one whole share")
                continue
            orders.append(Order(sym, "buy", qty, (it.reason or "") + note))
            trades_left -= 1
            cash -= qty * broker.unit_buy_cost(price)
            if sym not in acct.positions:
                open_positions += 1

        return orders, rejected
