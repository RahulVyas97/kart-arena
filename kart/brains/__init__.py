from __future__ import annotations

from ..config import ContestantCfg
from .ai import AITrader, ClaudeTrader
from .base import Brain, Context, Decision
from .hold import BuyAndHold
from .rules import TrendRules

__all__ = ["Brain", "Context", "Decision", "build_brain", "AITrader", "ClaudeTrader", "TrendRules", "BuyAndHold"]


def build_brain(cfg: ContestantCfg, ai_client=None, usage: dict | None = None, workdir: str | None = None) -> Brain:
    if cfg.brain == "ai":
        return AITrader(cfg.name, cfg.params, client=ai_client, usage=usage, workdir=workdir)
    if cfg.brain == "rules":
        return TrendRules(cfg.name, cfg.params)
    if cfg.brain == "hold":
        return BuyAndHold(cfg.name, cfg.params)
    raise ValueError(f"Unknown brain '{cfg.brain}'")
