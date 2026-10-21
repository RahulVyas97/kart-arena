"""Market hours. Uses exchange_calendars (holidays included) and falls back to
plain weekday hours from config when a calendar is missing or out of range."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from .config import MarketCfg

try:
    import exchange_calendars as xcals
except ImportError:  # pragma: no cover
    xcals = None


def utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class Market:
    def __init__(self, cfg: MarketCfg):
        self.cfg = cfg
        self.key = cfg.key
        self.name = cfg.name
        self.currency = cfg.currency
        self.symbols = list(cfg.symbols)
        self.tz = ZoneInfo(cfg.timezone)
        self.cal = None
        if xcals is not None:
            try:
                self.cal = xcals.get_calendar(cfg.calendar)
            except Exception:  # noqa: BLE001
                self.cal = None

    # -- public -------------------------------------------------------------
    def is_open(self, now: datetime) -> bool:
        if self.cal is not None:
            try:
                return bool(self.cal.is_open_on_minute(self._minute(now)))
            except Exception:  # noqa: BLE001  (e.g. date past the calendar's range)
                pass
        return self._fallback_is_open(now)

    def next_open(self, now: datetime) -> datetime:
        if self.cal is not None:
            try:
                return self.cal.next_open(self._minute(now)).to_pydatetime()
            except Exception:  # noqa: BLE001
                pass
        return self._fallback_next_open(now)

    def session_close(self, now: datetime) -> datetime | None:
        """Close time of the session that's open right now, else None."""
        if not self.is_open(now):
            return None
        if self.cal is not None:
            try:
                return self.cal.next_close(self._minute(now)).to_pydatetime()
            except Exception:  # noqa: BLE001
                pass
        local = utc(now).astimezone(self.tz)
        return datetime.combine(local.date(), self.cfg.close, self.tz).astimezone(timezone.utc)

    def minutes_to_close(self, now: datetime) -> float:
        close = self.session_close(now)
        return max(0.0, (close - utc(now)).total_seconds() / 60) if close else 0.0

    def local_date(self, now: datetime) -> date:
        return utc(now).astimezone(self.tz).date()

    # -- internals ----------------------------------------------------------
    @staticmethod
    def _minute(now: datetime) -> pd.Timestamp:
        return pd.Timestamp(utc(now)).floor("min")

    def _fallback_is_open(self, now: datetime) -> bool:
        local = utc(now).astimezone(self.tz)
        if local.weekday() >= 5:
            return False
        return self.cfg.open <= local.time().replace(tzinfo=None) < self.cfg.close

    def _fallback_next_open(self, now: datetime) -> datetime:
        local = utc(now).astimezone(self.tz)
        for days in range(0, 9):
            day = local.date() + timedelta(days=days)
            if day.weekday() >= 5:
                continue
            candidate = datetime.combine(day, self.cfg.open, self.tz)
            if candidate > local:
                return candidate.astimezone(timezone.utc)
        raise RuntimeError(f"Could not find the next open for {self.key}")
