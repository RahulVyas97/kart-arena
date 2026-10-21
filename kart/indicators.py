"""Small set of indicators, computed from the bars every contestant sees."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def _pct(now: float, then: float) -> float | None:
    if then is None or not then or math.isnan(then):
        return None
    return (now / then - 1) * 100


def snapshot(df: pd.DataFrame, tz, interval_minutes: int, fast: int = 8, slow: int = 21) -> dict:
    """Plain-number summary of one stock, used in the AI prompt and the logs."""
    close = df["close"]
    last = float(close.iloc[-1])
    bars_per_hour = max(1, 60 // interval_minutes)
    out: dict = {"price": last}
    out["chg_1h"] = _pct(last, float(close.iloc[-1 - bars_per_hour])) if len(close) > bars_per_hour else None

    local_dates = np.array(df.index.tz_convert(tz).date)
    today = local_dates[-1]
    prev = close[local_dates < today]
    out["chg_day"] = _pct(last, float(prev.iloc[-1])) if len(prev) else None
    today_bars = df[local_dates == today]
    out["day_high"] = float(today_bars["high"].max()) if len(today_bars) else None
    out["day_low"] = float(today_bars["low"].min()) if len(today_bars) else None

    f, s = sma(close, fast).iloc[-1], sma(close, slow).iloc[-1]
    out["trend"] = None if (pd.isna(f) or pd.isna(s)) else ("up" if f > s else "down" if f < s else "flat")
    r = rsi(close).iloc[-1]
    out["rsi"] = None if pd.isna(r) else float(r)
    rets = close.pct_change().tail(20)
    out["volatility"] = float(rets.std() * 100) if rets.notna().sum() > 2 else None
    vol_avg = df["volume"].tail(20).mean()
    out["volume_x"] = float(df["volume"].iloc[-1] / vol_avg) if vol_avg and vol_avg > 0 else None
    return out
