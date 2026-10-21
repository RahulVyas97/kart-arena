"""Paper accounts and the paper broker that fills orders with realistic costs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime

from .config import MarketCfg


@dataclass
class Position:
    qty: int
    avg_cost: float  # per share, including buy costs


@dataclass
class Account:
    contestant: str
    market: str
    currency: str
    start_capital: float
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    banked: float = 0.0          # profit set aside by the target rule
    times_banked: int = 0
    fees: float = 0.0            # commission + taxes paid
    slippage: float = 0.0        # cost of filling worse than the quoted price
    realized_pnl: float = 0.0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    peak_total: float = 0.0
    day: str = ""
    day_start_total: float = 0.0
    trades_today: int = 0
    paused_today: bool = False
    stopped: bool = False
    status_note: str = ""
    last_prices: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def market_value(self, prices: dict[str, float] | None = None) -> float:
        prices = prices or self.last_prices
        total = 0.0
        for sym, pos in self.positions.items():
            price = prices.get(sym) or self.last_prices.get(sym) or pos.avg_cost
            total += pos.qty * price
        return total

    def equity(self, prices: dict[str, float] | None = None) -> float:
        return self.cash + self.market_value(prices)

    def total(self, prices: dict[str, float] | None = None) -> float:
        return self.equity(prices) + self.banked

    def to_dict(self) -> dict:
        d = asdict(self)
        d["positions"] = {s: asdict(p) for s, p in self.positions.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Account":
        d = dict(d)
        d["positions"] = {s: Position(**p) for s, p in d.get("positions", {}).items()}
        return cls(**d)


class PaperBroker:
    """Fills market orders at the latest price, plus slippage and costs.
    Whole shares only, long only, cash only."""

    def __init__(self, cfg: MarketCfg):
        self.slip = cfg.slippage_bps / 10_000
        self.buy_cost = (cfg.commission_bps + cfg.buy_tax_bps) / 10_000
        self.sell_cost = (cfg.commission_bps + cfg.sell_tax_bps) / 10_000

    def round_trip_pct(self) -> float:
        return (2 * self.slip + self.buy_cost + self.sell_cost) * 100

    def unit_buy_cost(self, price: float) -> float:
        return price * (1 + self.slip) * (1 + self.buy_cost)

    def buy(self, acct: Account, symbol: str, qty: int, price: float, ts: datetime, reason: str) -> dict | None:
        unit = self.unit_buy_cost(price)
        qty = min(int(qty), int(acct.cash // unit))
        if qty < 1:
            return None
        fill_px = price * (1 + self.slip)
        gross = qty * fill_px
        fee = gross * self.buy_cost
        total_cost = gross + fee
        pos = acct.positions.get(symbol)
        if pos:
            new_qty = pos.qty + qty
            pos.avg_cost = (pos.qty * pos.avg_cost + total_cost) / new_qty
            pos.qty = new_qty
        else:
            acct.positions[symbol] = Position(qty=qty, avg_cost=total_cost / qty)
        acct.cash -= total_cost
        acct.fees += fee
        acct.slippage += qty * price * self.slip
        acct.trades += 1
        acct.trades_today += 1
        return self._fill(acct, ts, symbol, "buy", qty, fill_px, fee, None, reason)

    def sell(self, acct: Account, symbol: str, qty: int, price: float, ts: datetime, reason: str) -> dict | None:
        pos = acct.positions.get(symbol)
        if not pos:
            return None
        qty = min(int(qty), pos.qty)
        if qty < 1:
            return None
        fill_px = price * (1 - self.slip)
        gross = qty * fill_px
        fee = gross * self.sell_cost
        proceeds = gross - fee
        pnl = proceeds - qty * pos.avg_cost
        acct.cash += proceeds
        acct.fees += fee
        acct.slippage += qty * price * self.slip
        acct.realized_pnl += pnl
        acct.trades += 1
        acct.trades_today += 1
        if pnl > 0:
            acct.wins += 1
        elif pnl < 0:
            acct.losses += 1
        pos.qty -= qty
        if pos.qty == 0:
            del acct.positions[symbol]
        return self._fill(acct, ts, symbol, "sell", qty, fill_px, fee, pnl, reason)

    def liquidate(self, acct: Account, prices: dict[str, float], ts: datetime, reason: str) -> list[dict]:
        fills = []
        for sym in list(acct.positions):
            price = prices.get(sym) or acct.last_prices.get(sym)
            if price:
                fill = self.sell(acct, sym, acct.positions[sym].qty, price, ts, reason)
                if fill:
                    fills.append(fill)
        return fills

    @staticmethod
    def _fill(acct, ts, symbol, side, qty, px, fee, pnl, reason) -> dict:
        return {
            "ts": ts.isoformat(),
            "contestant": acct.contestant,
            "market": acct.market,
            "currency": acct.currency,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": round(px, 4),
            "fees": round(fee, 4),
            "pnl": None if pnl is None else round(pnl, 4),
            "cash_after": round(acct.cash, 2),
            "reason": reason,
        }
