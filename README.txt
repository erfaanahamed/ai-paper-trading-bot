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
- The starting virtual account balance can be selected from $10 to $1,000 in the sidebar. The displayed P&L is an approximate simulator estimate in price-units, not broker-calibrated USD P&L. Broker contract sizes, pip/tick values, commissions, spread, swaps/funding and slippage are not fully modeled.
- The self-learning model is a simple baseline and requires enough completed feature-rich paper trades before training. Win rates and predicted probabilities are not guarantees.
- Automatic scanning stops when the app/browser session sleeps, disconnects, or the hosting service suspends the app. This is not a persistent background worker.
- Test carefully and inspect data freshness, position management, and out-of-sample results before considering any real-money deployment.

Multi-strategy engine update:
- Added separate rule-based strategy votes for trend following, mean reversion, breakout, momentum, range trading, volatility expansion, price action, and a simplified liquidity-sweep proxy.
- Added market regime labels (trending, ranging, squeeze/quiet, mixed) and strategy vote counts to the all-market scanner.
- Automatic paper entries now use the ensemble: at least two aligned strategy votes and a non-tied majority are required, before the existing learning gate and risk/position checks.
- This is a limited strategy ensemble, not literally every strategy used worldwide. SMC is represented only by a simplified liquidity-sweep proxy. News trading is not activated because no live economic calendar/news feed is configured; do not assume news protection is working.
- Backtest functions still use the legacy directional signal logic, so results from that backtest do not yet validate the new ensemble. A separate ensemble backtest and walk-forward validation are needed before assessing performance.

Adaptive strategy-selection update:
- The selector now evaluates the eight implemented strategy families and detects a broad market regime, then chooses up to three candidate strategies separately for each market and timeframe.
- Completed auto-paper trades record the aligned strategy names and regime. When at least five recorded outcomes exist for a strategy in that exact market/timeframe, recent outcomes influence its ranking; otherwise the selector uses conservative regime-based priors and labels itself as warming up.
- At least two selected strategy votes must agree before a BUY/SELL signal can pass to the existing ML gate and paper risk checks. The scanner shows selected strategies and whether selection is still warming up or using performance history.
- The previous backtest is still not an adaptive-strategy walk-forward backtest. It must not be used as proof that this new selector is profitable. News/calendar integration is not included, so no active news trading or automatic news protection is claimed.
