"""Currency conversion for the leaderboard and starting capital."""

from __future__ import annotations

import time as _time

from .data import normalise_download


class FX:
    def __init__(self, base: str, fallback_usd_value: dict[str, float], live: bool = False, log=print):
        self.base = base.upper()
        self.fallback = {k.upper(): float(v) for k, v in fallback_usd_value.items()}
        self.fallback.setdefault("USD", 1.0)
        self.live = live
        self.log = log
        self._cache: dict[str, tuple[float, float]] = {}
        self._warned: set[str] = set()

    def usd_value(self, ccy: str) -> float:
        """Value of 1 unit of `ccy` in USD."""
        ccy = ccy.upper()
        if ccy == "USD":
            return 1.0
        if not self.live:
            return self._fallback(ccy)
        cached = self._cache.get(ccy)
        if cached and _time.time() - cached[0] < 1800:
            return cached[1]
        value = self._fetch(ccy)
        if value is None:
            return self._fallback(ccy)
        self._cache[ccy] = (_time.time(), value)
        return value

    def rate_to_base(self, ccy: str) -> float:
        return self.usd_value(ccy) / self.usd_value(self.base)

    def to_base(self, amount: float, ccy: str) -> float:
        return amount * self.rate_to_base(ccy)

    def from_base(self, amount: float, ccy: str) -> float:
        return amount / self.rate_to_base(ccy)

    # -- internals ----------------------------------------------------------
    def _fallback(self, ccy: str) -> float:
        if ccy not in self.fallback:
            raise KeyError(f"No FX rate for {ccy}. Add it to [fx_fallback_usd] in config.toml")
        return self.fallback[ccy]

    def _fetch(self, ccy: str) -> float | None:
        try:
            import yfinance as yf

            for ticker, invert in ((f"{ccy}USD=X", False), (f"USD{ccy}=X", True)):
                raw = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=True, timeout=15)
                frames = normalise_download(raw, [ticker])
                df = frames.get(ticker)
                if df is not None and len(df) and df["close"].iloc[-1] > 0:
                    last = float(df["close"].iloc[-1])
                    return 1.0 / last if invert else last
        except Exception:  # noqa: BLE001
            pass
        if ccy not in self._warned:
            self.log(f"  (live FX for {ccy} unavailable, using the rough rate from config)")
            self._warned.add(ccy)
        return None
