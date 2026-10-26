AI Brain Self-Learning Paper Trading Bot — Updated Dashboard

Files:
- app.py: Streamlit paper-trading app with the existing all-market scanner, indicator diagnostics, self-learning baseline, paper positions, trade history, and simplified backtest.
- requirements.txt: Python dependencies.

New dashboard features in this update:
- Daily performance report: closed trades, wins, losses, win rate, and net estimated P&L by closing date.
- Select a date to inspect its completed trades, including market, side, entry/exit, targets, stop, and estimated P&L.
- Downloadable daily report CSV.
- Individual ON/OFF toggle for every market. OFF prevents new automatic paper entries for that symbol; existing positions are still managed for stop-loss/targets.
- All-market analysis refreshes about every 60 seconds while the Streamlit page remains open and connected. Data is cached for about 55 seconds to limit repeated requests.

Run locally:
1. Install Python 3.10+.
2. In the project folder run: pip install -r requirements.txt
3. Run: streamlit run app.py

Important limitations:
- Paper trading only. No broker, exchange, wallet, private key, or real-money order connection is included.
- The bot uses Yahoo Finance as a market-data source. Intraday data may be delayed, stale, incomplete, or unavailable; the app cannot guarantee true real-time broker quotes.
- The displayed P&L is an approximate simulator estimate in price-units, not reliable INR P&L. Broker contract sizes, pip/tick values, commissions, spread, swaps/funding and slippage are not fully modeled.
- The self-learning model is a simple baseline and requires enough completed feature-rich paper trades before training. Win rates and predicted probabilities are not guarantees.
- Automatic scanning stops when the app/browser session sleeps, disconnects, or the hosting service suspends the app. This is not a persistent background worker.
- Test carefully and inspect data freshness, position management, and out-of-sample results before considering any real-money deployment.
