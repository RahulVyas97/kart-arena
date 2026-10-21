# Kart Arena 🏎️

A paper-trading arena. AI traders, a rules bot and a buy-and-hold benchmark each get the same
fake money, the same prices, the same costs and the same risk limits, then race across
**India → UK → US** stock markets, switching as each market opens and closes.

The point: find out whether an AI trader actually beats a dumb rule *and* just holding, before
any real money is involved.

**What it is not:** it can't place real trades. There's no broker connection anywhere in the
code. It only uses public data: market prices and public news headlines.

---

## 1. Setup (one time)

You need **Python 3.11+**.

**Windows (PowerShell)**
```powershell
cd kart-arena
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**
```bash
cd kart-arena
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Pick how the AI thinks (no API key needed)

There are two AI racers in `config.toml`. Set up whichever you like; any that isn't set up just
sits out, and the rules bot and buy-and-hold race either way.

**"Claude": your Claude subscription, no API key.** Needs a Pro, Max, Team or Enterprise plan
(the free plan doesn't include Claude Code).
```powershell
irm https://claude.ai/install.ps1 | iex        # Windows PowerShell
```
```bash
curl -fsSL https://claude.ai/install.sh | bash  # macOS / Linux
```
Then open a new terminal, run `claude` once and sign in with your Claude account. That's it.
Kart Arena calls Claude Code in the background; it counts against your plan's normal usage limits,
shared with your everyday Claude use.

*(Prefer an API key instead? Copy `.env.example` to `.env` and paste the key. With
`backend = "auto"`, a key in `.env` wins.)*

**"Local AI": free model on your own PC, no account at all.** Install Ollama from
https://ollama.com, then:
```bash
ollama pull qwen2.5:7b
```
About a 5 GB download. Keep Ollama running in the background. A gaming GPU makes it quick;
on CPU only, each decision can take a while.

## 3. Try it in 10 seconds (no internet needed)

```bash
python -m kart sim
```
Five days on **made-up** prices. This only proves the machinery works; the numbers mean nothing.
Add `--with-ai` to include the AI racers.

## 4. Run it for real (paper money)

```bash
python -m kart markets    # what's open now, and the next opening
python -m kart run        # live: decides every 15 min while any market is open, sleeps otherwise
```
The first lines tell you which racers are in and why any are sitting out.
Leave it running. Press **Ctrl+C** to stop; it saves and resumes next time.
`python -m kart run --fresh` wipes the live arena and starts over.

## 5. See who's winning

```bash
python -m kart status     # leaderboard in the terminal
python -m kart report     # writes runs/live/report.html, open it in a browser
```
The report has the leaderboard, a return-over-time chart, every trade, and **what the AI was
thinking** at each decision (including orders the risk manager blocked).

## 6. Test on real history, fast

```bash
python -m kart replay --days 20              # rules bot + buy-and-hold on the last 20 days
python -m kart replay --days 20 --with-ai    # include the AI racers (asks first)
```

## 7. Run it online and use it as a phone app (free, no PC)

GitHub can run the arena for you every 15 minutes on weekdays and publish the leaderboard as a
page you add to your phone's home screen. Once it's set up, your PC can be off.

1. **Make a free GitHub account**, then a **new public repository** called `kart-arena`
   (leave "Add a README" unticked). Public is what makes this free: anyone who finds the repo can
   see the code and the paper trades. Your API key is never visible.
2. **Upload this folder.** In PowerShell, in the kart-arena folder (needs Git: git-scm.com):
   ```powershell
   git init
   git add .
   git commit -m "Kart Arena"
   git branch -M main
   git remote add origin https://github.com/YOUR-NAME/kart-arena.git
   git push -u origin main
   ```
   The first push opens a browser to sign in to GitHub. Your `.env`, `.venv` and local `runs`
   folder are left out automatically.
3. **Add your key:** on the repo page, Settings → Secrets and variables → Actions →
   New repository secret. Name `ANTHROPIC_API_KEY`, value your key.
   *Using your Claude plan instead?* Run `claude setup-token` on your PC and save the token as a
   secret named `CLAUDE_CODE_OAUTH_TOKEN`.
4. **Switch the page on:** Settings → Pages → Build and deployment → Source: **GitHub Actions**.
5. **First run:** Actions tab → Kart Arena → **Run workflow**. After a couple of minutes the run
   shows your page link: `https://YOUR-NAME.github.io/kart-arena/`
6. **On your phone**, open that link, then:
   - iPhone (Safari): Share → **Add to Home Screen**
   - Android (Chrome): ⋮ menu → **Add to Home screen** / **Install app**

From then on it races by itself while markets are open, and the page refreshes every 5 minutes.

- **Pause it:** Actions → Kart Arena → ⋯ → Disable workflow.
- **Moving country:** change `TZ: Europe/London` in `.github/workflows/arena.yml` (for example `Asia/Kolkata`).
- **Local AI** (Ollama) can't run on GitHub's servers, so it sits out online.
- Scheduled runs can start a few minutes late, and Yahoo sometimes turns away cloud servers.
  If the page stops updating, open the Actions tab to see why.

---

## How it works

| Piece | What it does |
|---|---|
| **Markets** | Follows the sun: India 09:15–15:30 IST, London 08:00–16:30, New York 09:30–16:00 ET. Holidays come from exchange calendars. |
| **Racers** | `Claude` (AI via your plan or a key), `Local AI` (free Ollama model), `Trend Rules` (moving-average crossover), `Buy & Hold` (benchmark). Each gets its own account per market. |
| **Risk manager** | Hard limits the AI can't override: 20% max per stock, 5 stocks max, 12 trades/day, no buys in the last 15 min, pause after a 3% losing day, full stop at 15% below peak. |
| **Costs** | Slippage, UK stamp duty (0.5% on buys), India STT (0.1% each way), commissions. Fees killed most AIs in public trading contests, so they're modelled from day one. |
| **Profit target** | Your idea, built in: when an account is up 10%, it sells everything, **banks the profit**, and restarts with the original capital. The AI is never told the target, because a target pushes an agent toward bigger risks. |

**It never truly stops, except when it must:** stocks don't trade at weekends, and on weeknights
there's a gap of roughly 7–8 hours after New York closes before India opens. Set `enabled = true`
for Australia and Japan in `config.toml` to shrink that gap.

## What the AI costs

| Backend | Cost |
|---|---|
| Your Claude plan (`claude-code`) | No bill per call; uses your plan's usage limits (~85 small calls on a full weekday) |
| Free local model (`ollama`) | Free; uses your PC's power |
| API key (`api`) | A few US cents a day on `claude-haiku-5-5` with `effort = "low"`; the report shows a running estimate |

## Testing with friends

Each friend runs their own copy on their own laptop with their own setup: their own Claude plan,
their own API key, or the free local model. **Never share API keys or your `.env` file**: anyone
holding a key can spend its credits.

## Make it yours (`config.toml`)

- **Watchlists:** edit `symbols` per market (Yahoo Finance tickers: `.NS` India, `.L` London).
- **Risk and target:** the `[risk]` section.
- **More racers:** copy an AI `[[contestants]]` block, change `name`, `backend` and `model`
  (for example a bigger Claude model, or `llama3.1:8b` on Ollama), and you've got AI vs AI vs rules.
- **Currency:** `base_currency` for the leaderboard (then `run --fresh`).

## Keeping it running

- Your PC must stay awake: Windows → Settings → System → Power → Sleep: Never (while plugged in).
- If you move to a new country, nothing changes: times display in your computer's time zone.
- Later, a small cloud server can run it 24/7 without your laptop.

## Files

```
runs/live/state.json      accounts (resumes from here)
runs/live/trades.jsonl    every fill
runs/live/decisions.jsonl every decision, AI notes, blocked orders
runs/live/equity.csv      account values over time
runs/live/report.html     the report
.github/workflows/arena.yml  the online runner (GitHub Actions)
```

## Honest notes

- Paper results flatter you. Real trading adds partial fills, price gaps, outages and nerves.
- Yahoo Finance data is free and unofficial, may be delayed, and is meant for personal use.
- Judge the AI over **weeks**, against Buy & Hold. If it can't beat holding with fake money,
  it won't with real money.

## Tests

```bash
python -m pytest -q
```
Runs offline with stand-ins (a fake Claude Code, a fake Ollama server, a fake API): market hours,
costs, risk limits, every AI backend, and a full simulated race.
