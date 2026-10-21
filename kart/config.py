"""Load and validate config.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from datetime import time
from pathlib import Path

VALID_INTERVALS = (1, 2, 5, 15, 30, 60)
VALID_BRAINS = ("ai", "rules", "hold")
VALID_ON_TARGET = ("bank", "stop", "ignore")


class ConfigError(ValueError):
    pass


@dataclass
class ArenaCfg:
    base_currency: str = "GBP"
    capital_per_market: float = 10_000
    interval_minutes: int = 15
    runs_dir: str = "runs"


@dataclass
class RiskCfg:
    max_position_pct: float = 20
    max_open_positions: int = 5
    max_trades_per_day: int = 12
    daily_loss_limit_pct: float = 3
    max_drawdown_pct: float = 15
    no_new_buys_before_close_min: int = 15
    flatten_before_close_min: int = 0
    target_return_pct: float = 10
    on_target: str = "bank"


@dataclass
class MarketCfg:
    key: str
    name: str
    calendar: str
    timezone: str
    open: time
    close: time
    currency: str
    symbols: list[str]
    price_divisor: float = 1
    commission_bps: float = 0
    buy_tax_bps: float = 0
    sell_tax_bps: float = 0
    slippage_bps: float = 5
    enabled: bool = True


@dataclass
class ContestantCfg:
    name: str
    brain: str
    params: dict = field(default_factory=dict)


@dataclass
class Config:
    arena: ArenaCfg
    risk: RiskCfg
    markets: list[MarketCfg]
    contestants: list[ContestantCfg]
    fx_usd_value: dict[str, float]
    path: Path

    @property
    def enabled_markets(self) -> list[MarketCfg]:
        return [m for m in self.markets if m.enabled]


def _parse_time(value: str, where: str) -> time:
    try:
        hh, mm = str(value).split(":")
        return time(int(hh), int(mm))
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"{where}: expected HH:MM, got {value!r}") from exc


def _build(cls, raw: dict, where: str):
    allowed = {f.name for f in fields(cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"{where}: unknown setting(s) {sorted(unknown)}. Allowed: {sorted(allowed)}")
    return cls(**raw)


def load_config(path: str | Path = "config.toml") -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path.resolve()}")
    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    arena = _build(ArenaCfg, raw.get("arena", {}), "[arena]")
    risk = _build(RiskCfg, raw.get("risk", {}), "[risk]")

    if arena.interval_minutes not in VALID_INTERVALS:
        raise ConfigError(f"[arena] interval_minutes must be one of {VALID_INTERVALS}")
    if risk.on_target not in VALID_ON_TARGET:
        raise ConfigError(f"[risk] on_target must be one of {VALID_ON_TARGET}")
    arena.base_currency = arena.base_currency.upper()

    markets: list[MarketCfg] = []
    for i, m in enumerate(raw.get("markets", [])):
        where = f"[[markets]] #{i + 1}"
        m = dict(m)
        for key in ("open", "close"):
            if key not in m:
                raise ConfigError(f"{where}: missing '{key}'")
            m[key] = _parse_time(m[key], f"{where} {key}")
        mc = _build(MarketCfg, m, where)
        mc.currency = mc.currency.upper()
        if not mc.symbols:
            raise ConfigError(f"{where} ({mc.key}): symbols list is empty")
        markets.append(mc)

    keys = [m.key for m in markets]
    if len(keys) != len(set(keys)):
        raise ConfigError("Two [[markets]] share the same key")
    if not any(m.enabled for m in markets):
        raise ConfigError("No markets are enabled")

    contestants: list[ContestantCfg] = []
    for i, c in enumerate(raw.get("contestants", [])):
        c = dict(c)
        name, brain = c.pop("name", None), c.pop("brain", None)
        if not name or brain not in VALID_BRAINS:
            raise ConfigError(f"[[contestants]] #{i + 1}: needs a name and brain in {VALID_BRAINS}")
        contestants.append(ContestantCfg(name=name, brain=brain, params=c))
    names = [c.name for c in contestants]
    if len(names) != len(set(names)):
        raise ConfigError("Two [[contestants]] share the same name")
    if not contestants:
        raise ConfigError("No contestants defined")

    fx = {k.upper(): float(v) for k, v in raw.get("fx_usd_value", raw.get("fx_fallback_usd", {})).items()}
    fx.setdefault("USD", 1.0)
    needed = {arena.base_currency} | {m.currency for m in markets if m.enabled}
    missing = needed - set(fx)
    if missing:
        raise ConfigError(f"[fx_fallback_usd] needs a rough USD value for: {sorted(missing)}")

    return Config(arena=arena, risk=risk, markets=markets, contestants=contestants, fx_usd_value=fx, path=path)
