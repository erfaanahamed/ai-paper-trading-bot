# Hermes Agent Integration for AI Brain V7

This is the recommended way to use **Hermes Agent** (top model) with your AI Brain bot.

## What this does

- Hermes Agent becomes the **decision brain** (it decides BUY / SELL / WAIT)
- AI Brain V7 keeps **full control** of:
  - Risk management
  - Position sizing
  - News protection
  - Max daily loss / max trades
  - Paper execution

This is safer and better than letting Hermes trade with no limits.

---

## Architecture

```
Hermes Agent (top model)
        │
        │  submit_decision(BUY / SELL / WAIT)
        ▼
AI Brain V7  ──► Risk + News + Sizing ──► Paper Trade
```

---

## Step 1: Install Hermes Agent (on your computer or VPS)

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
source ~/.bashrc
hermes setup
```

During setup choose a strong model (recommended):
- Hermes 3 / Hermes 4 (Nous)
- or Claude Sonnet / Opus
- or DeepSeek / GLM via OpenRouter

---

## Step 2: How Hermes talks to your bot

I already added a bridge file:

`trading_engine/hermes_bridge.py`

Hermes can call this function:

```python
from trading_engine.hermes_bridge import submit_decision

result = submit_decision(
    side="BUY",                    # or "SELL" or "WAIT"
    market="Gold (XAUUSD)",
    confidence=0.78,
    reason="Strong momentum + RSI/MACD confluence, regime=TRENDING",
    model="Hermes-3"
)
print(result)
```

---

## Step 3: Create a Hermes Skill (recommended)

Inside Hermes, create a skill (example name: `ai_brain_trader`):

```text
You are a trading agent connected to AI Brain V7.

Your job:
1. Analyse Gold (XAUUSD) and other markets.
2. Decide BUY, SELL or WAIT.
3. Call the tool submit_trading_decision with your decision.
4. Always respect risk. Never force trades when confidence is low.
5. Prefer high-quality setups only (momentum + mean reversion based on regime).

Markets available:
- Gold (XAUUSD)
- EUR/USD
- GBP/USD
- BTC/USD
- Crude Oil
```

Then register the tool using the schema from:

```python
from trading_engine.hermes_bridge import hermes_tool_schema
print(hermes_tool_schema())
```

---

## Step 4: Run order (important)

1. Start your AI Brain V7 (paper mode)
2. Start Hermes Agent
3. Tell Hermes:  
   “Monitor Gold and submit decisions to AI Brain using the bridge”

AI Brain will only take the trade if risk rules allow it.

---

## Important warnings

- This is still **paper trading** first.
- Hermes can produce very good results (some public agents show 65-77% win rate) but many also lose.
- Never give Hermes live broker access until you have at least 100+ paper trades with positive expectancy.
- Always keep max daily loss and news protection ON.

---

## Files added

- `trading_engine/hermes_bridge.py`  ← main integration
- `HERMES_INTEGRATION.md`           ← this guide

---

## Next action for you

1. Install Hermes Agent on your machine (Step 1)
2. Come back and tell me:  
   “Hermes is installed”  
   Then I will give you the exact skill + tool connection code.
