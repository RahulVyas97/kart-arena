"""Command line entry point:  python -m kart <command>

  run       live paper trading, following each market's hours
  sim       fast test on made-up prices (no internet needed)
  replay    replay the last few weeks of real prices, fast
  status    print the leaderboard
  report    write an HTML report
  markets   which markets are open now, and when the next one opens
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path


def _load_env(paths: list[Path]) -> None:
    """Minimal .env reader so you don't have to set environment variables by hand."""
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main(argv: list[str] | None = None) -> int:
    if sys.version_info < (3, 11):
        print("Kart Arena needs Python 3.11 or newer.")
        return 1

    from .config import ConfigError, load_config
    from .data import SyntheticSource, YahooReplaySource, YahooSource, utcnow
    from .engine import LOCAL_TZ, Arena, fmt_local, fresh_start_end, human, quiet_stretches, run_clock, run_live
    from .fx import FX
    from .markets import Market
    from .report import print_table, write_html

    parser = argparse.ArgumentParser(
        prog="python -m kart",
        description="Paper-trading arena: AI vs rules vs buy-and-hold. Fake money only.",
    )
    parser.add_argument("--config", default="config.toml", help="path to config.toml (default: ./config.toml)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="live paper trading, following each market's hours")
    p.add_argument("--fresh", action="store_true", help="wipe the live arena and start over")
    p.add_argument("--once", action="store_true", help="do one round now (if a market is open), then exit")

    p = sub.add_parser("sim", help="fast test on made-up prices (no internet needed)")
    p.add_argument("--days", type=int, default=5, help="days to simulate (default 5)")
    p.add_argument("--seed", type=int, default=7, help="random seed for the made-up prices")
    p.add_argument("--with-ai", action="store_true", help="include AI contestants (real API calls, costs money)")

    p = sub.add_parser("replay", help="replay recent real prices, fast")
    p.add_argument("--days", type=int, default=20, help="recent days to replay (max about 55)")
    p.add_argument("--with-ai", action="store_true", help="include AI contestants (real API calls, costs money)")

    for name, text in (("status", "print the leaderboard"), ("report", "write the HTML report")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--mode", choices=["live", "sim", "replay"], default="live")
        if name == "report":
            p.add_argument("--web", metavar="DIR", help="build the phone app page into DIR (index.html + icons)")

    sub.add_parser("markets", help="which markets are open now, and when the next one opens")
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    _load_env([Path.cwd() / ".env", cfg_path.parent / ".env"])
    try:
        cfg = load_config(cfg_path)
    except ConfigError as exc:
        print(f"Config problem: {exc}")
        return 2
    base = cfg.arena.base_currency
    interval = cfg.arena.interval_minutes

    if args.cmd == "markets":
        now = utcnow()
        print(f"Times are in your time zone ({LOCAL_TZ}).")
        enabled = []
        for mc in cfg.markets:
            m = Market(mc)
            if not mc.enabled:
                print(f"  {m.key:<3} {m.name:<20} off (enable it in config.toml)")
                continue
            enabled.append(m)
            if m.is_open(now):
                close = m.session_close(now)
                print(f"  {m.key:<3} {m.name:<20} OPEN    closes {fmt_local(close)} (in {human(close - now)})")
            else:
                nxt = m.next_open(now)
                print(f"  {m.key:<3} {m.name:<20} closed  opens  {fmt_local(nxt)} (in {human(nxt - now)})")
        stretches = quiet_stretches(enabled, now)
        if stretches:
            a, b = stretches[0]
            print(f"Next stretch with no market open: {fmt_local(a)} -> {fmt_local(b)} ({human(b - a)}).")
            nights = [s for s in stretches if s[1] - s[0] < timedelta(hours=20)]
            if nights:
                a, b = min(nights, key=lambda s: s[1] - s[0])  # holidays make some nights longer
                print(f"Usual weeknight gap: {fmt_local(a, with_day=False)} -> {fmt_local(b, with_day=False)} "
                      f"({human(b - a)}).")
        print("Stock markets are closed at weekends. Enable AU or JP in config.toml to shrink the overnight gap.")
        return 0

    if args.cmd in ("status", "report"):
        arena = Arena(cfg, args.mode, None, FX(base, cfg.fx_usd_value, live=args.mode == "live"),
                      include_ai=False, verbose=False)
        if args.cmd == "report" and args.web:
            print(f"Phone page written: {write_html(arena, Path(args.web) / 'index.html', web=True)}")
            return 0
        if not arena.accounts:
            print(f"No {args.mode} results yet.")
            return 0
        if args.cmd == "status":
            print_table(arena)
        else:
            print(f"Report written: {write_html(arena)}")
        return 0

    if args.cmd == "run":
        fx = FX(base, cfg.fx_usd_value, live=True)
        arena = Arena(cfg, "live", YahooSource(interval), fx, fresh=args.fresh)
        if not arena.brains:
            print("No contestants can run. Check config.toml.")
            return 1
        try:
            run_live(arena, once=args.once)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            arena.save()
            if arena.accounts:
                print_table(arena)
                print(f"Report: {write_html(arena)}")
        return 0

    markets = [Market(m) for m in cfg.enabled_markets]
    fx = FX(base, cfg.fx_usd_value, live=False)

    if args.cmd == "sim":
        start, end = fresh_start_end(max(1, args.days))
        print(f"Simulating {args.days} days on MADE-UP prices. This tests the machinery; "
              "the results say nothing about real markets.")
        source = SyntheticSource(markets, interval, start, end, seed=args.seed)
        arena = Arena(cfg, "sim", source, fx, include_ai=args.with_ai, fresh=True, verbose=False)
    else:  # replay
        print("Downloading recent prices from Yahoo Finance...")
        source = YahooReplaySource(markets, interval)
        span = source.span()
        if not span:
            print("No price history came back. Check your internet connection and the tickers in config.toml.")
            return 1
        first, last = span
        end = last + timedelta(minutes=interval)
        start = max(first + timedelta(days=3), end - timedelta(days=max(1, args.days)))
        arena = Arena(cfg, "replay", source, fx, include_ai=args.with_ai, fresh=True, verbose=False)
        print(f"Replaying {fmt_local(start)} -> {fmt_local(end)}.")

    for name, why in arena.disabled:
        print(f"  {name} is sitting out: {why}")
    ai = [b for b in arena.brains if b.kind == "ai"]
    if ai:
        rounds = _count_rounds(markets, start, end, interval)
        for b in ai:
            if b.backend.name == "api":
                cost = rounds * (4000 * b.backend.usd_in + 900 * b.backend.usd_out) / 1e6
                what = f"rough cost ${cost:.2f} on your API key"
            elif b.backend.name == "claude-code":
                what = "counts against your Claude plan's usage limits"
            else:
                what = "free, runs on this PC (can be slow without a GPU)"
            print(f"  {b.name} will be asked about {rounds} times: {what}.")
        if sys.stdin.isatty() and input("  Continue? [y/N] ").strip().lower() != "y":
            return 0
    run_clock(arena, start, end, progress=lambda t: print(f"  {fmt_local(t, with_day=True)[:10]}..."))
    print()
    print_table(arena)
    print(f"Report: {write_html(arena)}")
    return 0


def _count_rounds(markets, start, end, interval) -> int:
    t, n = start, 0
    while t <= end:
        n += sum(1 for m in markets if m.is_open(t))
        t += timedelta(minutes=interval)
    return n


if __name__ == "__main__":
    sys.exit(main())
