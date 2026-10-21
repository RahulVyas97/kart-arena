"""The arena: runs every contestant on every open market, each tick."""

from __future__ import annotations

import time as _time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

from .brains import Context, Decision, build_brain
from .config import Config
from .data import utcnow
from .markets import Market, utc
from .portfolio import Account, PaperBroker
from .report import leaderboard, one_line
from .risk import RiskManager
from .store import Store

LOCAL_TZ = datetime.now().astimezone().tzinfo


def fmt_local(dt: datetime, with_day: bool = True) -> str:
    local = utc(dt).astimezone(LOCAL_TZ)
    return local.strftime("%a %d %b %H:%M" if with_day else "%H:%M")


def human(delta: timedelta) -> str:
    mins = max(0, int(delta.total_seconds() // 60))
    days, rem = divmod(mins, 1440)
    hours, minutes = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


class Arena:
    LOOKBACK_BARS = 160

    def __init__(
        self,
        cfg: Config,
        mode: str,
        source,
        fx,
        *,
        include_ai: bool = True,
        ai_client=None,
        fresh: bool = False,
        verbose: bool = True,
        log=print,
    ):
        self.cfg = cfg
        self.mode = mode
        self.source = source
        self.fx = fx
        self.verbose = verbose
        self.log = log
        self.interval = cfg.arena.interval_minutes
        self.markets = [Market(m) for m in cfg.enabled_markets]
        runs = Path(cfg.arena.runs_dir)
        if not runs.is_absolute():
            runs = cfg.path.parent / runs
        self.store = Store(runs / mode)
        if fresh:
            self.store.reset()
        self.state = self.store.load_state() or {
            "version": 1,
            "mode": mode,
            "created": utcnow().isoformat(),
            "base_currency": cfg.arena.base_currency,
            "capital_per_market": cfg.arena.capital_per_market,
            "accounts": {},
            "ai_usage": {},
        }
        if self.state.get("base_currency") != cfg.arena.base_currency:
            self.log(f"Note: this run started in {self.state.get('base_currency')}; "
                     f"use --fresh to restart in {cfg.arena.base_currency}.")
        usage = self.state.setdefault("ai_usage", {})

        self.brains = []
        self.disabled: list[tuple[str, str]] = []
        for c in cfg.contestants:
            if c.brain == "ai" and not include_ai:
                self.disabled.append((c.name, "AI is off in this mode (add --with-ai to include it)"))
                continue
            brain = build_brain(c, ai_client=ai_client, usage=usage, workdir=str(self.store.dir))
            if brain.kind == "ai" and not brain.available:
                self.disabled.append((c.name, brain.disabled_reason))
                continue
            self.brains.append(brain)
        self.order = [c.name for c in cfg.contestants]  # colour order follows config, not rank

        self.risk = RiskManager(cfg.risk)
        self.brokers = {m.key: PaperBroker(m.cfg) for m in self.markets}
        self.accounts: dict[tuple[str, str], Account] = {}
        for name, per_market in self.state.get("accounts", {}).items():
            for key, d in per_market.items():
                self.accounts[(name, key)] = Account.from_dict(d)

    # ------------------------------------------------------------------
    def account(self, name: str, market: Market) -> Account:
        key = (name, market.key)
        acct = self.accounts.get(key)
        if acct is None:
            start = round(self.fx.from_base(self.cfg.arena.capital_per_market, market.currency), 2)
            acct = Account(
                contestant=name, market=market.key, currency=market.currency,
                start_capital=start, cash=start, peak_total=start, day_start_total=start,
            )
            self.accounts[key] = acct
        return acct

    def save(self) -> None:
        accounts: dict[str, dict] = {}
        for (name, key), acct in self.accounts.items():
            accounts.setdefault(name, {})[key] = acct.to_dict()
        self.state["accounts"] = accounts
        self.store.save_state(self.state)

    # ------------------------------------------------------------------
    def tick(self, now: datetime) -> list[str]:
        """Run one decision round on every market that's open at `now`."""
        now = utc(now)
        ticked = []
        for market in self.markets:
            if market.is_open(now) and self._tick_market(market, now):
                ticked.append(market.key)
        if ticked:
            self.state["last_tick"] = now.isoformat()
            self.save()
            if self.verbose:
                self.log("  " + one_line(leaderboard(self)))
        return ticked

    def _tick_market(self, market: Market, now: datetime) -> bool:
        try:
            bars = self.source.bars(market, now, self.LOOKBACK_BARS)
        except Exception as exc:  # noqa: BLE001
            self.log(f"[{fmt_local(now)}] {market.key}: price data error ({type(exc).__name__}: {str(exc)[:120]})")
            return False
        prices = {s: float(df["close"].iloc[-1]) for s, df in bars.items() if len(df)}
        if not prices:
            if self.verbose:
                self.log(f"[{fmt_local(now)}] {market.key}: no prices came back (internet down, or Yahoo didn't "
                         "answer). Will retry next round.")
            return False

        minutes_to_close = market.minutes_to_close(now)
        day = market.local_date(now).isoformat()
        broker = self.brokers[market.key]
        news_n = max((b.news_per_symbol for b in self.brains), default=0)
        headlines = {}
        if news_n > 0 and getattr(self.source, "live", False):
            headlines = {s: self.source.headlines(s, news_n) for s in market.symbols}

        parts = []
        for brain in self.brains:
            acct = self.account(brain.name, market)
            acct.last_prices.update(prices)
            equity = acct.equity(prices)
            self.risk.roll_day(acct, day, equity + acct.banked)

            fills: list[dict] = []
            rejected: list[dict] = []
            decision = Decision()
            act = True
            if not brain.benchmark:
                was_stopped = acct.stopped
                self.risk.check_stops(acct, equity + acct.banked)
                if acct.stopped:
                    fills += broker.liquidate(acct, prices, now, acct.status_note)
                    if not was_stopped:
                        self.log(f"[{fmt_local(now)}] {market.key} {brain.name}: {acct.status_note}")
                    act = False
                elif self.risk.target_hit(acct, equity):
                    fills += self._hit_target(acct, broker, prices, now, market)
                    act = False
                elif self._flatten_now(minutes_to_close):
                    fills += broker.liquidate(acct, prices, now, "end-of-day flatten")
                    act = False

            if act:
                ctx = Context(
                    now=now, market=market, account=acct, bars=bars, prices=prices,
                    minutes_to_close=minutes_to_close, interval_minutes=self.interval,
                    risk=self.cfg.risk, round_trip_cost_pct=broker.round_trip_pct(),
                    headlines=headlines if brain.news_per_symbol else {},
                )
                try:
                    decision = brain.decide(ctx)
                except Exception as exc:  # noqa: BLE001
                    decision = Decision(error=f"brain crashed: {type(exc).__name__}: {exc}")
                orders, rejected = self.risk.vet(
                    decision.intents, acct, prices, market, minutes_to_close, broker, benchmark=brain.benchmark
                )
                for o in orders:
                    trade = broker.buy if o.side == "buy" else broker.sell
                    fill = trade(acct, o.symbol, o.qty, prices[o.symbol], now, o.reason)
                    if fill:
                        fills.append(fill)

            self._record(brain, market, now, acct, decision, rejected, fills)
            parts.append(f"{brain.name}: " + (", ".join(f"{f['side']} {f['symbol']} x{f['qty']}" for f in fills) or "-"))
            if decision.error and self.verbose:
                self.log(f"  ! {market.key} {brain.name}: {decision.error}")

        if self.verbose:
            self.log(f"[{fmt_local(now)}] {market.key:<3}| " + " | ".join(parts))
        return True

    def _flatten_now(self, minutes_to_close: float) -> bool:
        window = self.cfg.risk.flatten_before_close_min
        return window > 0 and minutes_to_close <= window

    def _hit_target(self, acct: Account, broker: PaperBroker, prices, now, market) -> list[dict]:
        pct = self.cfg.risk.target_return_pct
        fills = broker.liquidate(acct, prices, now, f"target +{pct:g}% reached")
        profit = acct.cash - acct.start_capital
        if self.cfg.risk.on_target == "bank" and profit > 0:
            acct.banked += profit
            acct.cash -= profit
            acct.times_banked += 1
            self.log(f"[{fmt_local(now)}] {market.key} {acct.contestant}: target hit, banked "
                     f"{profit:,.2f} {acct.currency}; restarting with {acct.start_capital:,.2f}")
        elif self.cfg.risk.on_target == "stop":
            acct.stopped = True
            acct.status_note = f"stopped: +{pct:g}% target reached"
            self.log(f"[{fmt_local(now)}] {market.key} {acct.contestant}: {acct.status_note}")
        return fills

    def _record(self, brain, market, now, acct, decision, rejected, fills) -> None:
        for f in fills:
            self.store.append_jsonl(self.store.trades_path, f)
        interesting = decision.intents or decision.error or rejected or fills or (decision.note and brain.kind == "ai")
        if interesting:
            self.store.append_jsonl(self.store.decisions_path, {
                "ts": now.isoformat(),
                "contestant": brain.name,
                "kind": brain.kind,
                "market": market.key,
                "note": decision.note,
                "error": decision.error,
                "intents": [asdict(i) for i in decision.intents],
                "rejected": rejected,
                "fills": len(fills),
            })
        total = acct.total()
        self.store.append_equity({
            "ts": now.isoformat(),
            "contestant": brain.name,
            "market": market.key,
            "currency": acct.currency,
            "equity": round(acct.equity(), 2),
            "banked": round(acct.banked, 2),
            "total_local": round(total, 2),
            "start_local": acct.start_capital,
            "return_pct": round((total / acct.start_capital - 1) * 100, 4),
            "total_base": round(self.fx.to_base(total, acct.currency), 2),
        })


# ----------------------------------------------------------------------
# Runners
# ----------------------------------------------------------------------
def _next_boundary(now: datetime, interval: int) -> datetime:
    now = utc(now)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    mins = (now - midnight).total_seconds() / 60
    return midnight + timedelta(minutes=(int(mins // interval) + 1) * interval)


def _sleep_until(t: datetime) -> None:
    while True:
        remaining = (t - utcnow()).total_seconds()
        if remaining <= 0:
            return
        _time.sleep(min(remaining, 30))


def quiet_stretches(markets: list[Market], start: datetime, days: int = 8) -> list[tuple[datetime, datetime]]:
    """Stretches in the next `days` when no enabled market is open."""
    step = timedelta(minutes=15)
    t = utc(start)
    end = t + timedelta(days=days)
    out: list[tuple[datetime, datetime]] = []
    prev_open = any(m.is_open(t) for m in markets)
    quiet_from = None if prev_open else t
    while t < end:
        t += step
        now_open = any(m.is_open(t) for m in markets)
        if prev_open and not now_open:
            prev = t - step
            closes = [c for c in (m.session_close(prev) for m in markets if m.is_open(prev)) if c]
            quiet_from = max(closes) if closes else t
        elif not prev_open and now_open and quiet_from is not None:
            out.append((quiet_from, min(m.next_open(quiet_from) for m in markets)))
            quiet_from = None
        prev_open = now_open
    return out


def run_live(arena: Arena, once: bool = False) -> None:
    log = arena.log
    names = ", ".join(f"{b.name} ({b.describe})" if b.kind == "ai" else b.name for b in arena.brains) or "nobody"
    log(f"Kart Arena LIVE (paper money). Contestants: {names}. Ctrl+C to stop.")
    for name, why in arena.disabled:
        log(f"  {name} is sitting out: {why}")
    while True:
        now = utcnow()
        if any(m.is_open(now) for m in arena.markets):
            arena.tick(now)
            if once:
                return
            wake = _next_boundary(now, arena.interval) + timedelta(seconds=20)
        else:
            nxt_market, nxt_time = min(((m, m.next_open(now)) for m in arena.markets), key=lambda x: x[1])
            log(f"[{fmt_local(now)}] All markets closed. Next: {nxt_market.name} opens "
                f"{fmt_local(nxt_time)} (in {human(nxt_time - now)}).")
            if once:
                return
            wake = min(nxt_time + timedelta(seconds=30), now + timedelta(hours=6))
        _sleep_until(wake)


def run_clock(arena: Arena, start: datetime, end: datetime, progress=None) -> int:
    """Fast-forward a simulated clock from start to end (sim and replay modes)."""
    t = _next_boundary(start - timedelta(seconds=1), arena.interval)
    end = utc(end)
    ticks = 0
    last_day = None
    while t <= end:
        if arena.tick(t):
            ticks += 1
        day = t.date()
        if progress and day != last_day:
            progress(t)
            last_day = day
        t += timedelta(minutes=arena.interval)
    return ticks


def fresh_start_end(days: int, end: datetime | None = None) -> tuple[datetime, datetime]:
    end = utc(end or utcnow()).replace(second=0, microsecond=0)
    return end - timedelta(days=days), end


__all__ = ["Arena", "run_live", "run_clock", "quiet_stretches", "fmt_local", "human", "fresh_start_end", "LOCAL_TZ"]
