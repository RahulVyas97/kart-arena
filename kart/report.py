"""Leaderboard (console) and a self-contained HTML report (no internet needed)."""

from __future__ import annotations

import html
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Categorical palette (validated light/dark). Colour follows the contestant's
# position in config.toml, never its rank.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]


# ----------------------------------------------------------------------
# Leaderboard
# ----------------------------------------------------------------------
def leaderboard(arena) -> list[dict]:
    groups: dict[str, list] = {}
    for (name, _key), acct in arena.accounts.items():
        groups.setdefault(name, []).append(acct)
    usage = arena.state.get("ai_usage", {})
    to_base = arena.fx.to_base
    rows = []
    for name, accts in groups.items():
        markets = {a.market: (a.total() / a.start_capital - 1) * 100 for a in accts}
        wins, losses = sum(a.wins for a in accts), sum(a.losses for a in accts)
        rows.append({
            "name": name,
            "return_pct": sum(markets.values()) / len(markets),
            "value_base": sum(to_base(a.total(), a.currency) for a in accts),
            "trades": sum(a.trades for a in accts),
            "win_rate": (wins / (wins + losses) * 100) if (wins + losses) else None,
            "costs_base": sum(to_base(a.fees + a.slippage, a.currency) for a in accts),
            "banked_base": sum(to_base(a.banked, a.currency) for a in accts),
            "times_banked": sum(a.times_banked for a in accts),
            "markets": markets,
            "notes": [f"{a.market}: {a.status_note}" for a in accts if a.status_note],
            "ai_usd": usage.get(name, {}).get("usd"),
            "ai_calls": usage.get(name, {}).get("calls"),
            "ai_backend": usage.get(name, {}).get("backend", "api"),
        })
    rows.sort(key=lambda r: r["return_pct"], reverse=True)
    return rows


def one_line(rows: list[dict]) -> str:
    if not rows:
        return "Leaderboard: nothing yet"
    return "Leaderboard: " + " | ".join(f"{r['name']} {r['return_pct']:+.2f}%" for r in rows)


def print_table(arena, out=print) -> None:
    rows = leaderboard(arena)
    base = arena.cfg.arena.base_currency
    keys = [m.key for m in arena.markets]
    if not rows:
        out("No results yet. Run the arena first.")
        return
    out("Leaderboard (paper money, returns after fees, taxes and slippage)")
    head = f"  #  {'Contestant':<16}{'Return':>9}{'Value ' + base:>14}{'Trades':>8}{'Win%':>6}{'Costs ' + base:>12}{'Banked':>10}"
    head += "".join(f"{k:>9}" for k in keys)
    out(head)
    for i, r in enumerate(rows, 1):
        win = f"{r['win_rate']:.0f}%" if r["win_rate"] is not None else "-"
        line = (f"  {i:<3}{r['name'][:15]:<16}{r['return_pct']:>+8.2f}%{r['value_base']:>14,.2f}{r['trades']:>8}"
                f"{win:>6}{r['costs_base']:>12,.2f}{r['banked_base']:>10,.2f}")
        line += "".join(f"{r['markets'][k]:>+8.2f}%" if k in r["markets"] else f"{'-':>9}" for k in keys)
        out(line)
    for r in rows:
        if r["ai_calls"]:
            out(f"  {r['name']}: {r['ai_calls']} AI calls, {spend_text(r)}")
        for n in r["notes"]:
            out(f"  {r['name']} {n}")


def spend_text(row: dict) -> str:
    if not row.get("ai_calls"):
        return "-"
    backend = row.get("ai_backend")
    if backend == "claude-code":
        return "on your Claude plan"
    if backend == "ollama":
        return "free (local model)"
    return f"about ${row['ai_usd']:.2f} (estimate)"


# ----------------------------------------------------------------------
# HTML report
# ----------------------------------------------------------------------
def _series(arena) -> tuple[pd.DatetimeIndex, dict[str, pd.Series]]:
    path = arena.store.equity_path
    if not path.exists():
        return pd.DatetimeIndex([]), {}
    df = pd.read_csv(path)
    if df.empty:
        return pd.DatetimeIndex([]), {}
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    piv = df.pivot_table(index="ts", columns=["contestant", "market"], values="return_pct", aggfunc="last")
    piv = piv.sort_index().ffill()
    names = [n for n in arena.order if n in piv.columns.get_level_values(0)]
    names += [n for n in piv.columns.get_level_values(0).unique() if n not in names]
    series = {n: piv[n].mean(axis=1, skipna=True) for n in names}
    idx = piv.index
    if len(idx) > 500:  # keep the chart light
        step = math.ceil(len(idx) / 500)
        keep = list(range(0, len(idx), step))
        if keep[-1] != len(idx) - 1:
            keep.append(len(idx) - 1)
        idx = idx[keep]
        series = {n: s.iloc[keep] for n, s in series.items()}
    return idx, series


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    first = math.floor(lo / step + 1e-9)
    last = math.ceil(hi / step - 1e-9)
    return [round(i * step, 10) for i in range(first, last + 1)]  # always covers lo..hi


def _chart(arena, idx, series) -> str:
    """Legend plus two renders of the same chart: wide for computers, compact for phones."""
    if len(idx) < 2 or not series:
        return '<p class="muted">The chart appears after a few ticks.</p>'
    names = list(series)
    color_slot = {n: (arena.order.index(n) if n in arena.order else len(arena.order) + i) % 8 for i, n in enumerate(names)}
    legend = "".join(
        f'<span class="key"><span class="swatch" style="background:var(--series-{color_slot[n] + 1})"></span>'
        f"{html.escape(n)}</span>" for n in names
    )
    wide = _chart_svg(idx, series, names, color_slot, uid="wide", W=760, H=300, max_xticks=6, end_labels=True)
    narrow = _chart_svg(idx, series, names, color_slot, uid="narrow", W=400, H=320, max_xticks=3, end_labels=False)
    return f'<div class="legend">{legend}</div>{wide}{narrow}'


def _chart_svg(idx, series, names, color_slot, *, uid, W, H, max_xticks, end_labels) -> str:
    longest = max(len(n[:14]) for n in names)
    right = min(190, int((longest + 8) * 6.8) + 16) if end_labels else 14
    left, top, bottom = (52 if end_labels else 44), 14, 30
    pw, ph = W - left - right, H - top - bottom
    t0, t1 = idx[0].timestamp(), idx[-1].timestamp()
    span = (t1 - t0) or 1
    values = [v for s in series.values() for v in s.tolist() if v == v]
    lo, hi = min(values + [0.0]), max(values + [0.0])
    ticks = _nice_ticks(lo, hi)
    ymin, ymax = ticks[0], ticks[-1]
    yr = (ymax - ymin) or 1

    def X(ts) -> float:
        return left + (ts.timestamp() - t0) / span * pw

    def Y(v) -> float:
        return top + (ymax - v) / yr * ph

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Return after costs over time, by contestant" '
             f'class="chart" id="chart-{uid}">']
    for t in ticks:  # gridlines + y labels
        y = Y(t)
        zero = abs(t) < 1e-9
        parts.append(f'<line class="{"baseline" if zero else "grid"}" x1="{left}" x2="{left + pw}" '
                     f'y1="{y:.1f}" y2="{y:.1f}"/>')
        label = "0%" if zero else f"{t:+g}%"
        parts.append(f'<text class="axis" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{label}</text>')
    for ts, label in _time_ticks(idx[0], idx[-1], max_ticks=max_xticks):  # x labels, spaced by time
        x = X(ts)
        anchor = "start" if x < left + 30 else "end" if x > left + pw - 30 else "middle"
        parts.append(f'<text class="axis" x="{x:.1f}" y="{H - 8}" text-anchor="{anchor}">{label}</text>')
    ends = []
    for n in names:
        s = series[n]
        pts = [(X(ts), Y(v)) for ts, v in zip(idx, s.tolist()) if v == v]
        if not pts:
            continue
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        slot = color_slot[n] + 1
        parts.append(f'<path class="line" d="{d}" style="stroke:var(--series-{slot})"/>')
        ex, ey = pts[-1]
        parts.append(f'<circle class="dot" cx="{ex:.1f}" cy="{ey:.1f}" r="4" style="fill:var(--series-{slot})"/>')
        ends.append((ey, n, s.dropna().iloc[-1]))
    ends.sort()
    if end_labels and len(ends) <= 4 and all(b[0] - a[0] >= 15 for a, b in zip(ends, ends[1:])):
        for ey, n, v in ends:
            parts.append(f'<text class="endlabel" x="{left + pw + 10}" y="{ey + 4:.1f}">'
                         f'{html.escape(n[:14])} <tspan class="endval">{v:+.2f}%</tspan></text>')
    parts.append(f'<line id="xhair-{uid}" class="xhair" x1="0" x2="0" y1="{top}" y2="{top + ph}" visibility="hidden"/>')
    parts.append(f'<rect id="hit-{uid}" x="{left}" y="{top}" width="{pw}" height="{ph}" fill="transparent" tabindex="0" '
                 f'aria-label="Chart: use arrow keys to read values"/>')
    parts.append("</svg>")

    payload = {
        "uid": uid,
        "x": [round(X(ts), 1) for ts in idx],
        "labels": [f"{ts.tz_convert(_local_tz()):%a %d %b %H:%M}" for ts in idx],
        "series": [{"name": n, "slot": color_slot[n] + 1,
                    "v": [None if v != v else round(v, 3) for v in series[n].tolist()]} for n in names],
    }
    return (f'<div class="chartwrap {uid}">{"".join(parts)}<div id="tip-{uid}" class="tip" hidden></div></div>'
            f'<script type="application/json" class="chartdata">{json.dumps(payload)}</script>')


def _local_tz():
    return datetime.now().astimezone().tzinfo


def _time_ticks(first: pd.Timestamp, last: pd.Timestamp, max_ticks: int = 6) -> list[tuple[pd.Timestamp, str]]:
    """Day boundaries for multi-day runs, round hours for short ones."""
    tz = _local_tz()
    a, b = first.tz_convert(tz), last.tz_convert(tz)
    hours = (b - a).total_seconds() / 3600
    if hours >= 36:
        cands = list(pd.date_range(a.normalize() + pd.Timedelta(days=1), b, freq="D"))
        step = max(1, math.ceil(len(cands) / max_ticks))
        return [(t, f"{t:%a %d %b}" if max_ticks > 3 else f"{t:%a %d}") for t in cands[::step]]
    for h in (1, 2, 3, 4, 6, 12):
        if hours / h <= max_ticks:
            break
    cands = list(pd.date_range(a.ceil(f"{h}h"), b, freq=f"{h}h"))
    return [(t, f"{t:%H:%M}") for t in cands]


def _fmt(v, spec="{:,.2f}", none="-"):
    return none if v is None or (isinstance(v, float) and v != v) else spec.format(v)


WEB_ASSETS = Path(__file__).parent / "web"
MANIFEST = {
    "name": "Kart Arena",
    "short_name": "Kart Arena",
    "description": "Paper-trading arena: AI vs rules vs buy-and-hold.",
    "start_url": "./",
    "scope": "./",
    "display": "standalone",
    "background_color": "#0d0d0d",
    "theme_color": "#0d0d0d",
    "icons": [
        {"src": "icon-192.png", "sizes": "192x192", "type": "image/png"},
        {"src": "icon-512.png", "sizes": "512x512", "type": "image/png"},
        {"src": "icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"},
    ],
}
WEB_HEAD = """<meta name="robots" content="noindex, nofollow">
<meta http-equiv="refresh" content="300">
<link rel="manifest" href="manifest.webmanifest">
<link rel="icon" type="image/png" sizes="192x192" href="icon-192.png">
<link rel="apple-touch-icon" href="apple-touch-icon.png">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="Kart Arena">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" media="(prefers-color-scheme: light)" content="#f9f9f7">
<meta name="theme-color" media="(prefers-color-scheme: dark)" content="#0d0d0d">"""


def _market_line(arena) -> str:
    """'Open now: UK · Next: US opens Fri 14:30' as of this update."""
    now = datetime.now(timezone.utc)
    open_now = [m.name for m in arena.markets if m.is_open(now)]
    parts = [("Open now: " + ", ".join(open_now)) if open_now else "All markets closed"]
    closed = [m for m in arena.markets if not m.is_open(now)]
    if closed:
        nxt = min(closed, key=lambda m: m.next_open(now))
        when = nxt.next_open(now).astimezone(_local_tz())
        parts.append(f"next: {nxt.name} opens {when:%a %H:%M}")
    return " · ".join(parts)


def write_html(arena, out_path: str | Path | None = None, web: bool = False) -> str:
    """Write the report. With web=True it also becomes an installable phone app:
    manifest, icons, auto-refresh, and 'updated X min ago'."""
    rows = leaderboard(arena)
    base = arena.cfg.arena.base_currency
    keys = [m.key for m in arena.markets]
    idx, series = _series(arena)
    trades = list(reversed(arena.store.read_jsonl(arena.store.trades_path, last=30)))
    notes = [d for d in reversed(arena.store.read_jsonl(arena.store.decisions_path, last=600))
             if d.get("kind") == "ai" and (d.get("note") or d.get("error"))][:12]
    now_local = datetime.now(timezone.utc).astimezone(_local_tz())

    lb = []
    for i, r in enumerate(rows, 1):
        markets = "".join(f"<td class='num'>{_fmt(r['markets'].get(k), '{:+.2f}%')}</td>" for k in keys)
        spend = html.escape(spend_text(r).replace("about ", "").replace(" (estimate)", ""))
        lb.append(
            f"<tr><td>{i}</td><td>{html.escape(r['name'])}</td><td class='num strong'>{r['return_pct']:+.2f}%</td>"
            f"<td class='num'>{r['value_base']:,.2f}</td><td class='num'>{r['trades']}</td>"
            f"<td class='num'>{_fmt(r['win_rate'], '{:.0f}%')}</td><td class='num'>{r['costs_base']:,.2f}</td>"
            f"<td class='num'>{r['banked_base']:,.2f}</td>{markets}<td class='num'>{spend}</td></tr>"
        )
    status = "".join(f"<li>{html.escape(r['name'])} {html.escape(n)}</li>" for r in rows for n in r["notes"])

    tr_rows = "".join(
        f"<tr><td>{html.escape(_short_ts(t['ts']))}</td><td>{html.escape(t['contestant'])}</td><td>{t['market']}</td>"
        f"<td>{t['side']}</td><td>{html.escape(t['symbol'])}</td><td class='num'>{t['qty']}</td>"
        f"<td class='num'>{t['price']:,.2f}</td><td class='num'>{_fmt(t.get('pnl'), '{:+,.2f}')}</td>"
        f"<td>{html.escape(str(t.get('reason', ''))[:90])}</td></tr>" for t in trades
    )
    note_items = "".join(
        f"<li><span class='muted'>{html.escape(_short_ts(d['ts']))} · {html.escape(d['contestant'])} · "
        f"{d['market']}</span><br>{html.escape(d.get('note') or '')}"
        + "".join(f"<br><span class='muted'>proposed: {html.escape(i['action'])} {html.escape(i['symbol'])} "
                  f"{i['pct']:g}% · {html.escape(i.get('reason') or '')}</span>" for i in d.get("intents", []))
        + "".join(f"<br><span class='muted'>blocked: {html.escape(r['action'])} {html.escape(r['symbol'])} · "
                  f"{html.escape(r['why'])}</span>" for r in d.get("rejected", []))
        + (f"<br><span class='muted'>error: {html.escape(d['error'])}</span>" if d.get("error") else "")
        + "</li>"
        for d in notes
    )
    table_rows = ""
    if len(idx):
        daily = pd.DataFrame(series, index=idx)
        daily.index = daily.index.tz_convert(_local_tz())
        daily = daily.groupby(daily.index.date).last()
        table_rows = "".join(
            f"<tr><td>{d}</td>" + "".join(f"<td class='num'>{_fmt(v, '{:+.2f}%')}</td>" for v in row) + "</tr>"
            for d, row in zip(daily.index, daily.itertuples(index=False))
        )
        table_head = "".join(f"<th>{html.escape(n)}</th>" for n in daily.columns)
    else:
        table_head = ""

    css_vars_light = "".join(f"--series-{i + 1}:{c};" for i, c in enumerate(SERIES_LIGHT))
    css_vars_dark = "".join(f"--series-{i + 1}:{c};" for i, c in enumerate(SERIES_DARK))
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Kart Arena</title>
{WEB_HEAD if web else ""}
<style>
:root {{ color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,0.10); {css_vars_light} }}
@media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) {{ color-scheme: dark; --page:#0d0d0d;
  --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --ring:rgba(255,255,255,0.10); {css_vars_dark} }} }}
:root[data-theme="dark"] {{ color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10); {css_vars_dark} }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--page); color:var(--ink); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width: 980px; margin: 0 auto; padding: max(24px, env(safe-area-inset-top)) 16px 48px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }} h2 {{ font-size: 16px; margin: 0 0 12px; }}
.sub, .muted {{ color: var(--ink2); }} .muted {{ font-size: 13px; }}
.card {{ background: var(--surface); border: 1px solid var(--ring); border-radius: 12px; padding: 16px; margin-top: 16px; }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--grid); white-space: nowrap; }}
th {{ color: var(--ink2); font-weight: 600; }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.strong {{ font-weight: 650; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 8px; font-size: 13px; color: var(--ink2); }}
.key {{ display: inline-flex; align-items: center; gap: 6px; }}
.swatch {{ width: 14px; height: 2px; border-radius: 2px; display: inline-block; }}
.chartwrap {{ position: relative; }}
.chartwrap.narrow {{ display: none; }}
@media (max-width: 640px) {{ .chartwrap.wide {{ display: none; }} .chartwrap.narrow {{ display: block; }} }}
.chart {{ width: 100%; height: auto; display: block; }}
.chart .grid {{ stroke: var(--grid); stroke-width: 1; }} .chart .baseline {{ stroke: var(--axis); stroke-width: 1; }}
.chart .axis {{ fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }}
.chart .line {{ fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
.chart .dot {{ stroke: var(--surface); stroke-width: 2; }}
.chart .endlabel {{ fill: var(--ink2); font-size: 12px; }} .chart .endval {{ fill: var(--ink); font-weight: 600; }}
.chart .xhair {{ stroke: var(--axis); stroke-width: 1; }}
.chart #hit:focus {{ outline: none; }} .chart #hit:focus-visible {{ outline: 2px solid var(--axis); }}
.tip {{ position: absolute; top: 8px; pointer-events: none; background: var(--surface); border: 1px solid var(--ring);
  border-radius: 8px; padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 14px rgba(0,0,0,.12); min-width: 150px; }}
.tip .row {{ display: flex; align-items: center; gap: 8px; justify-content: space-between; }}
.tip .val {{ font-weight: 650; color: var(--ink); font-variant-numeric: tabular-nums; }}
.tip .name {{ color: var(--ink2); display: inline-flex; align-items: center; gap: 6px; }}
ul.notes {{ list-style: none; padding: 0; margin: 0; }} ul.notes li {{ padding: 8px 0; border-bottom: 1px solid var(--grid); }}
details summary {{ cursor: pointer; color: var(--ink2); font-size: 13px; margin-top: 10px; }}
</style></head>
<body><main>
<h1>Kart Arena · {html.escape(arena.mode)}</h1>
<div class="sub">Paper money only. Returns are after fees, taxes and slippage.
Updated <span id="updated" data-ts="{now_local.isoformat()}" title="{now_local:%a %d %b %Y %H:%M}">{now_local:%a %d %b %Y %H:%M}</span>.</div>
<div class="muted">{html.escape(_market_line(arena))} (as of this update)</div>
<section class="card"><h2>Leaderboard</h2><div class="scroll"><table>
<thead><tr><th>#</th><th>Contestant</th><th class="num">Return</th><th class="num">Value ({base})</th>
<th class="num">Trades</th><th class="num">Win rate</th><th class="num">Costs ({base})</th><th class="num">Banked ({base})</th>
{"".join(f'<th class="num">{k}</th>' for k in keys)}<th class="num">AI spend</th></tr></thead>
<tbody>{"".join(lb) or '<tr><td colspan="12" class="muted">No results yet.</td></tr>'}</tbody></table></div>
{f'<ul class="muted">{status}</ul>' if status else ''}</section>
<section class="card"><h2>Return after costs over time</h2>{_chart(arena, idx, series)}
<details><summary>Show as a table (end of each day)</summary><div class="scroll"><table>
<thead><tr><th>Date</th>{table_head}</tr></thead><tbody>{table_rows}</tbody></table></div></details></section>
<section class="card"><h2>What the AI was thinking</h2>{f'<ul class="notes">{note_items}</ul>' if note_items else '<p class="muted">No AI notes yet.</p>'}</section>
<section class="card"><h2>Recent trades</h2><div class="scroll"><table>
<thead><tr><th>Time</th><th>Contestant</th><th>Market</th><th>Side</th><th>Stock</th><th class="num">Qty</th>
<th class="num">Price</th><th class="num">P&amp;L</th><th>Why</th></tr></thead>
<tbody>{tr_rows or '<tr><td colspan="9" class="muted">No trades yet.</td></tr>'}</tbody></table></div></section>
<p class="muted">Simulated results. Real trading adds things paper can't model well: partial fills, gaps, outages,
and your own nerves. Prices from Yahoo Finance may be delayed.</p>
</main>
<script>
(() => {{
  const u = document.getElementById("updated");
  if (!u) return;
  const t = new Date(u.dataset.ts);
  const tick = () => {{
    const m = Math.round((Date.now() - t) / 60000);
    u.textContent = m < 1 ? "just now" : m < 60 ? m + " min ago" : m < 1440 ? Math.round(m / 60) + " h ago"
      : Math.round(m / 1440) + " days ago";
  }};
  tick(); setInterval(tick, 60000);
}})();
document.querySelectorAll("script.chartdata").forEach(el => {{
  const data = JSON.parse(el.textContent), u = data.uid;
  const svg = document.getElementById("chart-" + u), hit = document.getElementById("hit-" + u);
  const xhair = document.getElementById("xhair-" + u), tip = document.getElementById("tip-" + u);
  let idx = data.x.length - 1;
  function nearest(px) {{
    let best = 0, dist = Infinity;
    data.x.forEach((x, i) => {{ const d = Math.abs(x - px); if (d < dist) {{ dist = d; best = i; }} }});
    return best;
  }}
  function show(i) {{
    idx = Math.max(0, Math.min(data.x.length - 1, i));
    const x = data.x[idx];
    xhair.setAttribute("x1", x); xhair.setAttribute("x2", x); xhair.setAttribute("visibility", "visible");
    tip.replaceChildren();
    const head = document.createElement("div"); head.className = "muted"; head.textContent = data.labels[idx];
    tip.appendChild(head);
    data.series.forEach(s => {{
      const row = document.createElement("div"); row.className = "row";
      const name = document.createElement("span"); name.className = "name";
      const key = document.createElement("span"); key.className = "swatch";
      key.style.background = "var(--series-" + s.slot + ")";
      name.appendChild(key); name.appendChild(document.createTextNode(s.name));
      const val = document.createElement("span"); val.className = "val";
      const v = s.v[idx]; val.textContent = v === null ? "-" : (v >= 0 ? "+" : "") + v.toFixed(2) + "%";
      row.appendChild(val); row.appendChild(name); tip.appendChild(row);
    }});
    tip.hidden = false;
    const box = svg.getBoundingClientRect(), scale = box.width / svg.viewBox.baseVal.width;
    const left = x * scale, w = tip.offsetWidth;
    tip.style.left = (left + 12 + w > box.width ? left - w - 12 : left + 12) + "px";
  }}
  function hide() {{ xhair.setAttribute("visibility", "hidden"); tip.hidden = true; }}
  hit.addEventListener("pointermove", e => {{
    const box = svg.getBoundingClientRect();
    show(nearest((e.clientX - box.left) * svg.viewBox.baseVal.width / box.width));
  }});
  hit.addEventListener("pointerleave", hide);
  hit.addEventListener("focus", () => show(idx));
  hit.addEventListener("blur", hide);
  hit.addEventListener("keydown", e => {{
    if (e.key === "ArrowLeft") {{ show(idx - 1); e.preventDefault(); }}
    if (e.key === "ArrowRight") {{ show(idx + 1); e.preventDefault(); }}
  }});
}});
</script>
</body></html>"""
    path = Path(out_path) if out_path else arena.store.report_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    if web:
        (path.parent / "manifest.webmanifest").write_text(json.dumps(MANIFEST, indent=2), encoding="utf-8")
        for icon in ("icon-192.png", "icon-512.png", "apple-touch-icon.png"):
            shutil.copyfile(WEB_ASSETS / icon, path.parent / icon)
    return str(path)


def _short_ts(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).astimezone(_local_tz()).strftime("%a %d %b %H:%M")
    except ValueError:
        return iso
