"""Where the AI contestant's answers come from.

  api          Anthropic API key in .env (pay as you go)
  claude-code  your Claude subscription (Pro/Max/Team), through the Claude Code app. No API key.
  ollama       a free model running on your own PC through Ollama. No account at all.

Every backend takes the same system prompt, user prompt and JSON schema and
returns the parsed JSON answer plus token usage. Anything that goes wrong is
raised as BackendError, and the contestant simply holds that round.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request


class BackendError(Exception):
    pass


class Backend:
    name = "base"
    label = ""
    reason = ""  # non-empty = can't run here, and says how to fix it

    def ask(self, system: str, user: str, schema: dict) -> tuple[dict, dict]:  # pragma: no cover
        raise NotImplementedError


def _parse_json_text(text: str) -> dict:
    """Parse JSON, tolerating stray text around the object."""
    text = (text or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise BackendError("answer wasn't valid JSON") from None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            raise BackendError("answer wasn't valid JSON") from None
    if not isinstance(data, dict):
        raise BackendError("answer wasn't a JSON object")
    return data


# ---------------------------------------------------------------------------
class ApiBackend(Backend):
    name = "api"
    label = "API key"

    def __init__(self, model: str, effort: str, max_tokens: int, usd_in: float, usd_out: float, client=None):
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.usd_in, self.usd_out = usd_in, usd_out
        self.client = client
        if self.client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                self.reason = "no ANTHROPIC_API_KEY in .env"
            else:
                import anthropic

                self.client = anthropic.Anthropic(max_retries=2, timeout=90)

    def ask(self, system: str, user: str, schema: dict) -> tuple[dict, dict]:
        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        if self.effort:
            request["output_config"]["effort"] = self.effort
        try:
            resp = self.client.messages.create(**request)
        except Exception as exc:  # noqa: BLE001  (network, auth, rate limit...)
            raise BackendError(f"API call failed: {type(exc).__name__}: {str(exc)[:200]}") from exc
        u = getattr(resp, "usage", None)
        tin = int(getattr(u, "input_tokens", 0) or 0)
        tout = int(getattr(u, "output_tokens", 0) or 0)
        usage = {"in": tin, "out": tout, "usd": tin / 1e6 * self.usd_in + tout / 1e6 * self.usd_out}
        stop = getattr(resp, "stop_reason", None)
        if stop in ("refusal", "max_tokens"):
            raise BackendError(f"no usable answer (stop_reason={stop})", usage)
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        try:
            return _parse_json_text(text), usage
        except BackendError as exc:
            raise BackendError(str(exc), usage) from None


# ---------------------------------------------------------------------------
class ClaudeCodeBackend(Backend):
    """Runs `claude -p` (Claude Code's headless mode) signed in with your Claude plan.
    Usage counts against your plan's normal limits; there's no per-call bill."""

    name = "claude-code"
    label = "your Claude plan"

    def __init__(self, model: str, effort: str, cli: str | None = None, timeout: int = 180,
                 runner=subprocess.run, workdir: str | None = None):
        self.model, self.effort, self.timeout = model, effort, timeout
        self.runner = runner
        self.workdir = workdir
        self.cli = cli or shutil.which("claude")
        if not self.cli:
            self.reason = ("Claude Code isn't installed. Install it (see README), run `claude` once "
                           "and sign in with your Claude account")
            return
        if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
            return  # signed in with a token from `claude setup-token` (used when running online)
        status = self._auth_status()
        if status is False:
            self.reason = "Claude Code isn't signed in. Run `claude` once and sign in with your Claude account"

    def _env(self) -> dict:
        env = dict(os.environ)
        env.pop("ANTHROPIC_API_KEY", None)  # make Claude Code use the subscription login, not a key
        return env

    def _auth_status(self) -> bool | None:
        try:
            proc = self.runner([self.cli, "auth", "status"], capture_output=True, text=True, encoding="utf-8",
                               timeout=30, env=self._env())
            data = json.loads(proc.stdout)
            return bool(data.get("loggedIn"))
        except Exception:  # noqa: BLE001  (older version, odd output: find out on the first call instead)
            return None

    def ask(self, system: str, user: str, schema: dict) -> tuple[dict, dict]:
        fd, prompt_file = tempfile.mkstemp(prefix="kart-system-", suffix=".txt")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(system)
            cmd = [self.cli, "-p", "--output-format", "json", "--no-session-persistence", "--safe-mode",
                   "--model", self.model, "--system-prompt-file", prompt_file, "--json-schema", json.dumps(schema)]
            if self.effort:
                cmd += ["--effort", self.effort]
            cmd += ["--tools", ""]  # no tools: it only reads the snapshot and answers (keep this last)
            try:
                proc = self.runner(cmd, input=user, capture_output=True, text=True, encoding="utf-8",
                                   timeout=self.timeout, env=self._env(), cwd=self.workdir)
            except subprocess.TimeoutExpired:
                raise BackendError(f"Claude Code didn't answer within {self.timeout}s") from None
            except OSError as exc:
                raise BackendError(f"couldn't start Claude Code: {exc}") from exc
        finally:
            try:
                os.unlink(prompt_file)
            except OSError:
                pass

        out = None
        for line in reversed((proc.stdout or "").strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    out = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        if out is None:
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            raise BackendError(f"Claude Code gave no result (exit {proc.returncode}): {detail[-1][:200] if detail else ''}")

        u = out.get("usage") or {}
        tin = sum(int(u.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        usage = {"in": tin, "out": int(u.get("output_tokens") or 0), "usd": 0.0}
        if out.get("is_error"):
            raise BackendError(f"Claude Code: {str(out.get('result') or out.get('subtype'))[:200]}", usage)
        data = out.get("structured_output")
        if isinstance(data, dict):
            return data, usage
        try:
            return _parse_json_text(str(out.get("result", ""))), usage
        except BackendError as exc:
            raise BackendError(str(exc), usage) from None


# ---------------------------------------------------------------------------
class OllamaBackend(Backend):
    """A free local model via Ollama (https://ollama.com). Nothing leaves your PC."""

    name = "ollama"
    label = "local model"

    def __init__(self, model: str, host: str = "http://localhost:11434", timeout: int = 300, opener=None):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        # Talk to Ollama directly, never through a proxy (it normally lives on this PC).
        self.opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({})).open
        try:
            with self.opener(urllib.request.Request(f"{self.host}/api/tags"), timeout=3) as resp:
                tags = json.loads(resp.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            self.reason = f"Ollama isn't running. Install it from ollama.com, then run: ollama pull {model}"
            return
        names = {m.get("name", "") for m in tags.get("models", [])} | {m.get("model", "") for m in tags.get("models", [])}
        wanted = {model} if ":" in model else {model, f"{model}:latest"}
        if not names & wanted:
            self.reason = f"model '{model}' isn't downloaded yet. Run: ollama pull {model}"

    def ask(self, system: str, user: str, schema: dict) -> tuple[dict, dict]:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": schema,
            "options": {"temperature": 0},
        }).encode("utf-8")
        req = urllib.request.Request(f"{self.host}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self.opener(req, timeout=self.timeout) as resp:
                out = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise BackendError(f"Ollama error {exc.code}: {exc.read().decode('utf-8', 'replace')[:200]}") from exc
        except Exception as exc:  # noqa: BLE001
            raise BackendError(f"Ollama didn't answer: {type(exc).__name__}: {str(exc)[:200]}") from exc
        usage = {"in": int(out.get("prompt_eval_count") or 0), "out": int(out.get("eval_count") or 0), "usd": 0.0}
        content = (out.get("message") or {}).get("content", "")
        try:
            return _parse_json_text(content), usage
        except BackendError as exc:
            raise BackendError(str(exc), usage) from None


# ---------------------------------------------------------------------------
def make_backend(params: dict, client=None, runner=None, opener=None, workdir: str | None = None) -> Backend:
    kind = str(params.get("backend", "auto")).lower()
    model = params.get("model")
    effort = params.get("effort", "low")
    if kind == "auto":
        kind = "api" if (client is not None or os.environ.get("ANTHROPIC_API_KEY")) else "claude-code"
    if kind == "api":
        return ApiBackend(model or "claude-haiku-5-5", effort, int(params.get("max_tokens", 3000)),
                          float(params.get("usd_per_mtok_in", 0.10)), float(params.get("usd_per_mtok_out", 0.50)),
                          client=client)
    if kind in ("claude-code", "plan", "subscription"):
        return ClaudeCodeBackend(model or "claude-haiku-5-5", effort, cli=params.get("claude_path"),
                                 timeout=int(params.get("timeout_s", 180)), runner=runner or subprocess.run,
                                 workdir=workdir)
    if kind == "ollama":
        return OllamaBackend(model or "qwen2.5:7b", host=params.get("host", "http://localhost:11434"),
                             timeout=int(params.get("timeout_s", 300)), opener=opener)
    raise ValueError(f"Unknown AI backend '{kind}'. Use auto, api, claude-code or ollama.")
