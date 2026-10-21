"""Where a run keeps its state and logs: runs/<mode>/..."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

EQUITY_FIELDS = [
    "ts", "contestant", "market", "currency", "equity", "banked",
    "total_local", "start_local", "return_pct", "total_base",
]


class Store:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.trades_path = self.dir / "trades.jsonl"
        self.decisions_path = self.dir / "decisions.jsonl"
        self.equity_path = self.dir / "equity.csv"
        self.report_path = self.dir / "report.html"

    def reset(self) -> None:
        for p in (self.state_path, self.trades_path, self.decisions_path, self.equity_path, self.report_path):
            p.unlink(missing_ok=True)

    def load_state(self) -> dict | None:
        if not self.state_path.exists():
            return None
        with self.state_path.open(encoding="utf-8") as fh:
            return json.load(fh)

    def save_state(self, state: dict) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=1, default=str)
        os.replace(tmp, self.state_path)

    def append_jsonl(self, path: Path, obj: dict) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, default=str) + "\n")

    def append_equity(self, row: dict) -> None:
        new = not self.equity_path.exists()
        with self.equity_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=EQUITY_FIELDS)
            if new:
                writer.writeheader()
            writer.writerow({k: row.get(k) for k in EQUITY_FIELDS})

    def read_jsonl(self, path: Path, last: int | None = None) -> list[dict]:
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as fh:
            lines = fh.readlines()
        if last:
            lines = lines[-last:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
