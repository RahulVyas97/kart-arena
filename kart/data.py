"""Price data sources.

YahooSource        live prices from Yahoo Finance (free, unofficial, may be delayed)
YahooReplaySource  the last weeks of real 15-min bars, replayed fast
SyntheticSource    made-up random-walk prices, for testing the engine offline

Every source returns {symbol: DataFrame[open, high, low, close, volume]} with a
UTC index of bar start times. Replay and synthetic sources only ever return
bars that had finished by `now`, so no contestant can peek at the future.
"""

from __future__ import annotations

import time as _time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .markets import Market, utc

YF_INTERVALS = {1: "1m", 2: "2m", 5: "5m", 15: "15m", 30: "30m", 60: "60m"}
COLS = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Normalising yfinance output (its column layout varies by version/arguments)
# ---------------------------------------------------------------------------
def normalise_download(raw: pd.DataFrame | None, symbols: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = {str(v) for v in raw.columns.get_level_values(0)}
        level1 = {str(v) for v in raw.columns.get_level_values(1)}
        for sym in symbols:
            if sym in level0:
                sub = raw[sym]
            elif sym in level1:
                sub = raw.xs(sym, axis=1, level=1)
            else:
                continue
            out[sym] = _clean(sub)
    elif len(symbols) == 1:
        out[symbols[0]] = _clean(raw)
    return {k: v for k, v in out.items() if not v.empty}


def _clean(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.copy()
    sub.columns = [str(c).strip().lower().replace(" ", "_") for c in sub.columns]
    if "close" not in sub.columns and "adj_close" in sub.columns:
        sub["close"] = sub["adj_close"]
    if "close" not in sub.columns:
        return pd.DataFrame(columns=COLS)
    sub = sub.dropna(subset=["close"])
    for col in ("open", "high", "low"):
        if col not in sub.columns:
            sub[col] = sub["close"]
    if "volume" not in sub.columns:
        sub["volume"] = 0.0
    idx = pd.DatetimeIndex(sub.index)
    sub.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    sub = sub[~sub.index.duplicated(keep="last")].sort_index()
    return sub[COLS].astype(float)


def _divide_prices(df: pd.DataFrame, divisor: float) -> pd.DataFrame:
    if not divisor or divisor == 1:
        return df
    df = df.copy()
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]] / divisor
    return df


# ---------------------------------------------------------------------------
# Live Yahoo Finance
# ---------------------------------------------------------------------------
class YahooSource:
    live = True

    def __init__(self, interval_minutes: int, log=print):
        import yfinance as yf  # imported lazily so tests don't need network

        self.yf = yf
        self.log = log
        self.interval_minutes = interval_minutes
        self.interval = YF_INTERVALS[interval_minutes]
        self.period = "10d" if interval_minutes <= 30 else "60d"
        self._news: dict[str, tuple[float, list[str]]] = {}
        self._news_failed_logged = False

    def bars(self, market: Market, now: datetime, lookback: int) -> dict[str, pd.DataFrame]:
        raw = self.yf.download(
            market.symbols,
            period=self.period,
            interval=self.interval,
            group_by="ticker",
            auto_adjust=True,
            prepost=False,
            progress=False,
            threads=True,
            timeout=20,
            multi_level_index=True,
        )
        data = normalise_download(raw, market.symbols)
        now = utc(now)
        return {
            sym: _divide_prices(df[df.index <= now].tail(lookback), market.cfg.price_divisor)
            for sym, df in data.items()
        }

    def headlines(self, symbol: str, n: int) -> list[str]:
        """Recent public headlines for a stock. Cached for an hour. Never raises."""
        if n <= 0:
            return []
        cached = self._news.get(symbol)
        if cached and _time.time() - cached[0] < 3600:
            return cached[1]
        items: list[str] = []
        try:
            raw = self.yf.Ticker(symbol).get_news(count=n) or []
            for item in raw[:n]:
                content = item.get("content", item) if isinstance(item, dict) else {}
                title = content.get("title")
                provider = content.get("provider")
                source = provider.get("displayName") if isinstance(provider, dict) else content.get("publisher")
                when = content.get("pubDate") or content.get("displayTime") or ""
                if title:
                    items.append(f"{title} ({source or 'unknown source'}{', ' + str(when)[:16] if when else ''})")
        except Exception as exc:  # noqa: BLE001
            if not self._news_failed_logged:
                self.log(f"  (headlines unavailable: {type(exc).__name__}; continuing without news)")
                self._news_failed_logged = True
        self._news[symbol] = (_time.time(), items)
        return items


# ---------------------------------------------------------------------------
# Replay of recent real bars
# ---------------------------------------------------------------------------
class YahooReplaySource:
    live = False

    def __init__(self, markets: list[Market], interval_minutes: int, log=print):
        import yfinance as yf

        self.interval_minutes = interval_minutes
        period = "59d" if interval_minutes <= 30 else "720d"
        self.data: dict[str, dict[str, pd.DataFrame]] = {}
        for market in markets:
            log(f"  downloading {market.name} history...")
            raw = yf.download(
                market.symbols,
                period=period,
                interval=YF_INTERVALS[interval_minutes],
                group_by="ticker",
                auto_adjust=True,
                prepost=False,
                progress=False,
                threads=True,
                timeout=30,
                multi_level_index=True,
            )
            frames = normalise_download(raw, market.symbols)
            self.data[market.key] = {s: _divide_prices(df, market.cfg.price_divisor) for s, df in frames.items()}
            if not frames:
                log(f"  WARNING: no history for {market.name}")

    def span(self) -> tuple[datetime, datetime] | None:
        starts, ends = [], []
        for frames in self.data.values():
            for df in frames.values():
                if len(df):
                    starts.append(df.index[0])
                    ends.append(df.index[-1])
        if not starts:
            return None
        return min(starts).to_pydatetime(), max(ends).to_pydatetime()

    def bars(self, market: Market, now: datetime, lookback: int) -> dict[str, pd.DataFrame]:
        cutoff = utc(now) - timedelta(minutes=self.interval_minutes)
        return {s: df[df.index <= cutoff].tail(lookback) for s, df in self.data.get(market.key, {}).items()}

    def headlines(self, symbol: str, n: int) -> list[str]:
        return []  # past headlines aren't available point-in-time


# ---------------------------------------------------------------------------
# Synthetic prices for offline testing
# ---------------------------------------------------------------------------
BASE_PRICE = {"USD": 200.0, "GBP": 30.0, "INR": 1500.0, "JPY": 3000.0, "AUD": 50.0, "EUR": 60.0}


class SyntheticSource:
    """Random walks with changing trends. Only bars when each market is open.
    Results on this data mean nothing about real markets; it's for testing."""

    live = False

    def __init__(self, markets: list[Market], interval_minutes: int, start: datetime, end: datetime, seed: int = 7):
        self.interval_minutes = interval_minutes
        rng = np.random.default_rng(seed)
        step = f"{interval_minutes}min"
        warmup = utc(start) - timedelta(days=7)
        grid = pd.date_range(pd.Timestamp(warmup).floor(step), pd.Timestamp(utc(end)).ceil(step), freq=step)
        self.data: dict[str, dict[str, pd.DataFrame]] = {}
        for market in markets:
            open_idx = pd.DatetimeIndex([t for t in grid if market.is_open(t.to_pydatetime())])
            n = len(open_idx)
            frames = {}
            for sym in market.symbols:
                if n == 0:
                    frames[sym] = pd.DataFrame(columns=COLS)
                    continue
                base = BASE_PRICE.get(market.currency, 100.0) * rng.uniform(0.5, 2.0)
                regime = 30
                drift = np.repeat(rng.normal(0, 0.0012, size=n // regime + 1), regime)[:n]
                vol = rng.uniform(0.002, 0.006)
                rets = drift + vol * rng.standard_normal(n)
                close = base * np.exp(np.cumsum(rets))
                open_ = np.concatenate([[base], close[:-1]])
                wick = np.abs(rng.normal(0, vol / 2, n))
                frames[sym] = pd.DataFrame(
                    {
                        "open": open_,
                        "high": np.maximum(open_, close) * (1 + wick),
                        "low": np.minimum(open_, close) * (1 - wick),
                        "close": close,
                        "volume": rng.lognormal(12, 0.5, n),
                    },
                    index=open_idx,
                )
            self.data[market.key] = frames

    def bars(self, market: Market, now: datetime, lookback: int) -> dict[str, pd.DataFrame]:
        cutoff = utc(now) - timedelta(minutes=self.interval_minutes)
        return {s: df[df.index <= cutoff].tail(lookback) for s, df in self.data.get(market.key, {}).items()}

    def headlines(self, symbol: str, n: int) -> list[str]:
        return []


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
