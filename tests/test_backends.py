"""The keyless AI backends, tested offline: a fake `claude` CLI and a fake Ollama server."""

from __future__ import annotations

import json
import subprocess
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from kart.brains import AITrader
from kart.brains.ai import ORDER_SCHEMA
from kart.brains.backends import ClaudeCodeBackend, OllamaBackend, make_backend
from kart.config import RiskCfg, load_config
from kart.markets import Market
from kart.portfolio import Account
from kart.brains.base import Context

ROOT = Path(__file__).resolve().parents[1]
ANSWER = {"market_view": "calm", "orders": [{"symbol": "MSFT", "action": "buy", "percent": 10, "reason": "trend"}]}


def us_ctx():
    cfg = load_config(ROOT / "config.toml")
    market = Market(next(m for m in cfg.markets if m.key == "US"))
    acct = Account("AI", "US", "USD", 10_000, 10_000)
    return Context(
        now=datetime(2026, 10, 16, 15, 0, tzinfo=timezone.utc), market=market, account=acct, bars={},
        prices={s: 100.0 for s in market.symbols}, minutes_to_close=200, interval_minutes=15,
        risk=RiskCfg(), round_trip_cost_pct=0.1,
    )


# ------------------------------------------------------------------ Claude Code (your plan)
class FakeCLI:
    """Pretends to be `claude`. Records every call."""

    def __init__(self, logged_in=True, envelope=None):
        self.logged_in = logged_in
        self.envelope = envelope or {"type": "result", "is_error": False, "result": json.dumps(ANSWER),
                                     "structured_output": ANSWER,
                                     "usage": {"input_tokens": 3, "cache_creation_input_tokens": 1200,
                                               "output_tokens": 80}}
        self.calls = []

    def __call__(self, cmd, **kw):
        record = {"cmd": cmd, **kw}
        if "--system-prompt-file" in cmd:
            record["system_text"] = Path(cmd[cmd.index("--system-prompt-file") + 1]).read_text(encoding="utf-8")
        self.calls.append(record)
        if cmd[1:3] == ["auth", "status"]:
            out = json.dumps({"loggedIn": self.logged_in, "authMethod": "oauth_token"})
        else:
            out = "some warning line\n" + json.dumps(self.envelope)
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")


def test_claude_code_backend_uses_plan_not_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-be-used")
    cli = FakeCLI()
    brain = AITrader("Claude", {"backend": "claude-code", "claude_path": "/fake/claude"}, runner=cli, usage={})
    assert brain.available, brain.disabled_reason
    d = brain.decide(us_ctx())
    assert d.note == "calm" and d.intents[0].symbol == "MSFT" and not d.error

    call = cli.calls[-1]
    cmd = call["cmd"]
    assert cmd[:2] == ["/fake/claude", "-p"]
    assert cmd[-2:] == ["--tools", ""]                         # no tools, and nothing after the variadic flag
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == ORDER_SCHEMA
    assert "%" not in cmd[cmd.index("--json-schema") + 1]       # safe through Windows .cmd shims
    assert {"--safe-mode", "--no-session-persistence", "--output-format"} <= set(cmd)
    assert "PAPER-TRADING" in call["system_text"]               # system prompt went via a file
    assert "Watchlist" in call["input"]                         # snapshot went via stdin
    assert "ANTHROPIC_API_KEY" not in call["env"]               # forces the subscription login
    assert brain.usage["Claude"]["backend"] == "claude-code" and brain.usage["Claude"]["usd"] == 0


def test_claude_code_backend_errors_and_setup_messages():
    assert "isn't signed in" in ClaudeCodeBackend("m", "low", cli="/fake/claude", runner=FakeCLI(logged_in=False)).reason
    missing = ClaudeCodeBackend("m", "low", cli=None, runner=FakeCLI())
    if missing.cli is None:  # only meaningful where no real `claude` is on PATH
        assert "isn't installed" in missing.reason

    err = FakeCLI(envelope={"type": "result", "is_error": True, "result": "usage limit reached"})
    brain = AITrader("Claude", {"backend": "claude-code", "claude_path": "/fake/claude"}, runner=err, usage={})
    d = brain.decide(us_ctx())
    assert d.intents == [] and "usage limit reached" in d.error

    text_only = FakeCLI(envelope={"type": "result", "is_error": False, "result": "Here: " + json.dumps(ANSWER)})
    brain = AITrader("Claude", {"backend": "claude-code", "claude_path": "/fake/claude"}, runner=text_only, usage={})
    assert brain.decide(us_ctx()).intents[0].action == "buy"   # falls back to parsing the text result


def test_auto_prefers_key_then_plan(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert make_backend({"backend": "auto", "claude_path": "/fake/claude"}, runner=FakeCLI()).name == "claude-code"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert make_backend({"backend": "auto"}).name == "api"


# ------------------------------------------------------------------ Ollama (free local model)
class _OllamaHandler(BaseHTTPRequestHandler):
    models = ["qwen2.5:7b"]
    seen: list = []

    def log_message(self, *args):  # keep test output quiet
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"models": [{"name": m, "model": m} for m in self.models]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append(req)
        self._send({"message": {"role": "assistant", "content": json.dumps(ANSWER)},
                    "prompt_eval_count": 900, "eval_count": 60, "done": True})


@pytest.fixture
def ollama_server():
    server = HTTPServer(("127.0.0.1", 0), _OllamaHandler)
    _OllamaHandler.seen = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_ollama_backend(ollama_server):
    brain = AITrader("Local AI", {"backend": "ollama", "model": "qwen2.5:7b", "host": ollama_server}, usage={})
    assert brain.available, brain.disabled_reason
    d = brain.decide(us_ctx())
    assert d.intents[0].symbol == "MSFT" and d.note == "calm"
    req = _OllamaHandler.seen[-1]
    assert req["format"] == ORDER_SCHEMA and req["stream"] is False and req["model"] == "qwen2.5:7b"
    assert req["messages"][0]["role"] == "system" and "Watchlist" in req["messages"][1]["content"]
    assert brain.usage["Local AI"] == {"calls": 1, "input_tokens": 900, "output_tokens": 60, "usd": 0.0,
                                       "backend": "ollama"}


def test_ollama_setup_messages(ollama_server):
    assert "ollama pull llama3.1:8b" in OllamaBackend("llama3.1:8b", host=ollama_server).reason
    assert OllamaBackend("qwen2.5", host=ollama_server).reason  # 'qwen2.5' means qwen2.5:latest, not pulled
    assert "isn't running" in OllamaBackend("qwen2.5:7b", host="http://127.0.0.1:9").reason
