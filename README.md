# Multi-Market AI-Assisted Paper Trading Bot (Prototype)

A mobile-friendly Streamlit dashboard for simulated-only trading analysis. It includes Gold, selected Forex pairs, and selected Crypto assets; 15m, 30m, and 1h bars; EMA 9/21/50/200, RSI(14), ATR(14), ADX(14), +DI/-DI, a volume-weighted average approximation, rolling support/resistance, and an adaptive rolling channel.

## Safety and scope

- Starting virtual balance: ₹100,000.
- Paper trading only. No broker, MT5, wallet, API secret, or real-money order integration.
- This is a prototype, not a proven profitable AI strategy or investment advice.
- Market data is provided by Yahoo Finance via `yfinance`; it may be delayed, incomplete, or unavailable. Symbol coverage can change.
- The bot only refreshes while the dashboard and hosting process are active. Free hosting is not guaranteed to run 24/7.
- SMC concepts are approximated with rolling swing/breakout logic; Power Channel is a rolling mean ± 2 standard deviations.
- P/L conversion and position sizing are simplified for research and are not broker-accurate contract sizing.

## Run locally (PC/Linux/Mac)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy from an Android phone using free cloud hosting

You will need a GitHub account and a Streamlit Community Cloud account. The exact interface may change.

1. Extract this ZIP on your phone, or upload the files to a new GitHub repository using the GitHub mobile website/app.
2. Put `app.py`, `requirements.txt`, and `README.md` at the repository root.
3. Open Streamlit Community Cloud, sign in with GitHub, and create a new app pointing to the repository and `app.py`.
4. Wait for dependencies to install and the app to launch.
5. Open the app URL on your Android phone. Keep the dashboard open for periodic refreshes.

### Persistence warning

The app writes `paper_state.json` beside the code. Some cloud hosts use ephemeral filesystems or may restart apps, so the virtual trade history/balance may reset or be lost. This starter does not yet use a persistent database or a separate always-on worker.

## Strategy summary

- Each timeframe receives an indicator score.
- A BUY/SELL signal is emitted only when 15m, 30m, and 1h all agree.
- Entry uses the latest 15m close; stop distance is based on 1.5 × ATR (with a small minimum distance); targets are 1R and 2R.
- New trades risk a configurable fraction of virtual balance; a 3% daily loss guardrail blocks new entries.
- The simulator checks candle highs/lows for stop/target events and conservatively assumes the stop is hit first if both stop and target are touched within one bar.

## Before relying on results

Run a meaningful backtest with historical data, include spread/fees/slippage and instrument contract specifications, verify time-zone/session behavior, and compare paper fills against real market conditions. Do not enable real-money trading based on a short or favorable demo period.
