"""Offline tests: no internet, no API key needed.  Run with:  python -m pytest -q"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kart.brains import ClaudeTrader
from kart.brains.base import Context
from kart.config import RiskCfg, load_config
from kart.data import SyntheticSource, normalise_download
from kart.engine import Arena, run_clock
from kart.fx import FX
from kart.markets import Market
from kart.portfolio import Account, PaperBroker, Position
from kart.report import leaderboard, write_html
from kart.risk import Intent, RiskManager

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- helpers
class _Block:
    def __init__(self, type_, text=None):
        self.type = type_
        self.text = text


class _Usage:
    input_tokens = 1500
    output_tokens = 300


class _Resp:
    def __init__(self, text, stop="end_turn"):
        self.content = [_Block("thinking"), _Block("text", text)]
        self.stop_reason = stop
        self.usage = _Usage()


class FakeClient:
    """Stands in for anthropic.Anthropic(); `fn(request) -> _Resp`."""

    def __init__(self, fn):
        outer = self

        class _Messages:
            def create(self, **kw):
                outer.calls.append(kw)
                return fn(kw)

        self.calls: list[dict] = []
        self.messages = _Messages()


def toy_trader(request):
    """Buys the first watchlist stock when flat, sells it once it's held."""
    prompt = request["messages"][0]["content"]
    lines = prompt.splitlines()
    header = next(i for i, ln in enumerate(lines) if ln.startswith("symbol |"))
    first = lines[header + 1].split(" | ")[0].strip()
    held = "Positions: none" not in prompt
    orders = [{"symbol": first, "action": "sell" if held else "buy", "percent": 100 if held else 50,
               "reason": "toy"}]
    return _Resp(json.dumps({"market_view": "toy view", "orders": orders}))


def make_config(tmp_path: Path, extra_risk: str = "") -> Path:
    text = (ROOT / "config.toml").read_text(encoding="utf-8")
    text = text.replace('runs_dir = "runs"', f'runs_dir = "{(tmp_path / "runs").as_posix()}"')
    if extra_risk:
        text = text.replace("[risk]\n", "[risk]\n" + extra_risk + "\n", 1)
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def ctx_for(market, acct, prices, bars=None, minutes_to_close=120, risk=None):
    return Context(
        now=datetime(2026, 10, 16, 14, 0, tzinfo=timezone.utc), market=market, account=acct,
        bars=bars or {}, prices=prices, minutes_to_close=minutes_to_close, interval_minutes=15,
        risk=risk or RiskCfg(), round_trip_cost_pct=0.6,
    )


# ---------------------------------------------------------------- data
def test_normalise_both_yfinance_layouts():
    idx = pd.date_range("2026-10-16 13:30", periods=3, freq="15min", tz="America/New_York")
    fields = ["Open", "High", "Low", "Close", "Volume"]
    data = np.arange(30, dtype=float).reshape(3, 10) + 1
    by_ticker = pd.DataFrame(data, index=idx, columns=pd.MultiIndex.from_product([["AAPL", "MSFT"], fields]))
    by_field = pd.DataFrame(data, index=idx, columns=pd.MultiIndex.from_product([fields, ["AAPL", "MSFT"]]))
    for raw in (by_ticker, by_field):
        out = normalise_download(raw, ["AAPL", "MSFT"])
        assert set(out) == {"AAPL", "MSFT"}
        assert list(out["AAPL"].columns) == ["open", "high", "low", "close", "volume"]
        assert str(out["AAPL"].index.tz) == "UTC"
    flat = pd.DataFrame(data[:, :5], index=idx, columns=fields)
    assert len(normalise_download(flat, ["AAPL"])["AAPL"]) == 3


# ---------------------------------------------------------------- markets
def test_market_hours_and_calendar_fallback(tmp_path):
    cfg = load_config(make_config(tmp_path))
    by_key = {m.key: Market(m) for m in cfg.markets}
    us, uk, india = by_key["US"], by_key["UK"], by_key["IN"]
    fri_1500z = datetime(2026, 10, 16, 15, 0, tzinfo=timezone.utc)
    assert us.is_open(fri_1500z) and uk.is_open(fri_1500z) and not india.is_open(fri_1500z)
    assert not us.is_open(datetime(2026, 10, 17, 15, 0, tzinfo=timezone.utc))  # Saturday
    assert us.next_open(datetime(2026, 10, 17, 15, 0, tzinfo=timezone.utc)) == datetime(
        2026, 10, 19, 13, 30, tzinfo=timezone.utc)
    # The India calendar's data stops at the end of 2026: the fallback must take over.
    assert india.is_open(datetime(2027, 1, 4, 5, 0, tzinfo=timezone.utc))      # Mon 10:30 IST
    assert not india.is_open(datetime(2027, 1, 4, 11, 0, tzinfo=timezone.utc))  # Mon 16:30 IST
    assert india.next_open(datetime(2027, 1, 2, 12, 0, tzinfo=timezone.utc)) == datetime(
        2027, 1, 4, 3, 45, tzinfo=timezone.utc)
    assert 0 < us.minutes_to_close(fri_1500z) <= 300


# ---------------------------------------------------------------- broker & risk
def test_costs_are_charged(tmp_path):
    cfg = load_config(make_config(tmp_path))
    uk = next(m for m in cfg.markets if m.key == "UK")
    broker = PaperBroker(uk)
    acct = Account("t", "UK", "GBP", 10_000, 10_000)
    now = datetime(2026, 10, 16, 10, tzinfo=timezone.utc)
    broker.buy(acct, "AZN.L", 10, 100.0, now, "test")
    # 10 shares at 100 + 0.05% slippage + 0.5% stamp duty
    assert acct.cash == pytest.approx(10_000 - 10 * 100.05 * 1.005)
    broker.sell(acct, "AZN.L", 10, 100.0, now, "test")
    assert acct.cash < 10_000 and acct.losses == 1 and not acct.positions


def test_risk_clamps_and_blocks():
    rm = RiskManager(RiskCfg(max_position_pct=20, max_open_positions=2, no_new_buys_before_close_min=15))
    cfg = load_config(ROOT / "config.toml")
    market = Market(next(m for m in cfg.markets if m.key == "US"))
    broker = PaperBroker(market.cfg)
    acct = Account("t", "US", "USD", 10_000, 10_000)
    prices = {s: 100.0 for s in market.symbols}

    orders, rejected = rm.vet([Intent("aapl", "buy", 80, "big bet")], acct, prices, market, 120, broker)
    assert orders[0].symbol == "AAPL" and orders[0].qty <= 20 and "clamped" in orders[0].reason

    orders, rejected = rm.vet([Intent("AAPL", "buy", 10, "")], acct, prices, market, 10, broker)
    assert not orders and "last 15 min" in rejected[0]["why"]

    orders, rejected = rm.vet([Intent("FAKE", "buy", 10, "")], acct, prices, market, 120, broker)
    assert not orders and "watchlist" in rejected[0]["why"]

    acct.positions = {"AAPL": Position(5, 100.0), "MSFT": Position(5, 100.0)}
    acct.paused_today = True
    orders, rejected = rm.vet(
        [Intent("NVDA", "buy", 10, ""), Intent("AAPL", "sell", 100, "")], acct, prices, market, 120, broker)
    assert [o.side for o in orders] == ["sell"]  # sells always allowed; buys blocked while paused


def test_daily_loss_and_drawdown_limits():
    rm = RiskManager(RiskCfg(daily_loss_limit_pct=3, max_drawdown_pct=15))
    acct = Account("t", "US", "USD", 10_000, 10_000)
    rm.roll_day(acct, "2026-10-16", 10_000)
    rm.check_stops(acct, 9_650)
    assert acct.paused_today and not acct.stopped
    rm.roll_day(acct, "2026-10-19", 9_650)
    assert not acct.paused_today
    rm.check_stops(acct, 8_400)
    assert acct.stopped


# ---------------------------------------------------------------- AI brain
def test_ai_parses_structured_output_and_fails_safe(tmp_path):
    cfg = load_config(make_config(tmp_path))
    market = Market(next(m for m in cfg.markets if m.key == "US"))
    acct = Account("Claude", "US", "USD", 10_000, 10_000)
    prices = {s: 100.0 for s in market.symbols}

    good = FakeClient(lambda kw: _Resp(json.dumps({"market_view": "calm", "orders": [
        {"symbol": "MSFT", "action": "BUY", "percent": 12.5, "reason": "trend up"}]})))
    brain = ClaudeTrader("Claude", {"model": "claude-haiku-5-5"}, client=good, usage={})
    d = brain.decide(ctx_for(market, acct, prices))
    assert d.note == "calm" and d.intents[0].action == "buy" and d.intents[0].pct == 12.5
    req = good.calls[0]
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert "tool_choice" not in req
    assert "target" not in req["system"].lower()  # the AI never sees the profit target
    assert brain.usage["Claude"]["calls"] == 1 and brain.usage["Claude"]["usd"] > 0

    for bad in (lambda kw: _Resp("not json"), lambda kw: _Resp("{}", stop="max_tokens")):
        d = ClaudeTrader("Claude", {}, client=FakeClient(bad), usage={}).decide(ctx_for(market, acct, prices))
        assert d.intents == [] and d.error

    def boom(kw):
        raise ConnectionError("offline")

    d = ClaudeTrader("Claude", {}, client=FakeClient(boom), usage={}).decide(ctx_for(market, acct, prices))
    assert d.intents == [] and "offline" in d.error


# ---------------------------------------------------------------- end to end
def test_sim_end_to_end_with_fake_ai(tmp_path):
    cfg = load_config(make_config(tmp_path))
    markets = [Market(m) for m in cfg.enabled_markets]
    end = datetime(2026, 10, 16, 21, 0, tzinfo=timezone.utc)
    start = end - timedelta(days=4)
    source = SyntheticSource(markets, 15, start, end, seed=3)
    fake = FakeClient(toy_trader)
    arena = Arena(cfg, "sim", source, FX("GBP", cfg.fx_usd_value), ai_client=fake, fresh=True, verbose=False)
    ticks = run_clock(arena, start, end)
    assert ticks > 50
    assert fake.calls, "the AI contestant should have been asked"
    rows = leaderboard(arena)
    assert {r["name"] for r in rows} == {"Claude", "Trend Rules", "Buy & Hold"}  # Local AI sits out: no Ollama here
    assert any(name == "Local AI" for name, _ in arena.disabled)
    assert all(r["trades"] > 0 for r in rows)
    keys = {k for (_, k) in arena.accounts}
    assert keys == {"IN", "UK", "US"}
    # no account ever goes below zero cash
    assert all(a.cash >= -1e-6 for a in arena.accounts.values())
    report = Path(write_html(arena))
    assert report.exists() and "Leaderboard" in report.read_text(encoding="utf-8")

    # the phone app version: installable page with manifest and icons
    page = Path(write_html(arena, tmp_path / "site" / "index.html", web=True))
    text = page.read_text(encoding="utf-8")
    assert 'rel="manifest"' in text and "noindex" in text and 'id="updated"' in text
    manifest = json.loads((page.parent / "manifest.webmanifest").read_text(encoding="utf-8"))
    assert manifest["display"] == "standalone"
    for icon in manifest["icons"]:
        assert (page.parent / icon["src"]).stat().st_size > 1000
    assert (page.parent / "apple-touch-icon.png").exists()
    assert 'rel="manifest"' not in report.read_text(encoding="utf-8")  # local report stays plain
    state = json.loads(arena.store.state_path.read_text(encoding="utf-8"))
    assert state["ai_usage"]["Claude"]["calls"] == len(fake.calls)


def test_target_banking_restarts_with_original_capital(tmp_path):
    cfg = load_config(make_config(tmp_path, extra_risk=""))
    cfg.risk.target_return_pct = 1  # tiny target so it triggers
    markets = [Market(m) for m in cfg.enabled_markets]
    end = datetime(2026, 10, 16, 21, 0, tzinfo=timezone.utc)
    start = end - timedelta(days=4)
    source = SyntheticSource(markets, 15, start, end, seed=11)
    arena = Arena(cfg, "sim", source, FX("GBP", cfg.fx_usd_value), include_ai=False, fresh=True, verbose=False,
                  log=lambda *_: None)
    run_clock(arena, start, end)
    banked = [a for a in arena.accounts.values() if a.times_banked]
    for a in banked:
        assert a.banked > 0
        assert a.equity() <= a.start_capital * 1.02 + 1  # restarted near the original capital
