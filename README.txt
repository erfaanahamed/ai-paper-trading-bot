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
- This is a limited strategy ensemble, not literally every strategy used worldwide. SMC is represented only by a simplified liquidity-sweep proxy. News Protection uses the published Forex Factory/Fair Economy weekly calendar feed and filters strictly to red-folder (High-impact) events. The feed can fail or be incomplete; when no valid feed/cache is available, new entries are blocked while protection is ON.
- Backtest functions still use the legacy directional signal logic, so results from that backtest do not yet validate the new ensemble. A separate ensemble backtest and walk-forward validation are needed before assessing performance.

Adaptive strategy-selection update:
- The selector now evaluates the eight implemented strategy families and detects a broad market regime, then chooses up to three candidate strategies separately for each market and timeframe.
- Completed auto-paper trades record the aligned strategy names and regime. When at least five recorded outcomes exist for a strategy in that exact market/timeframe, recent outcomes influence its ranking; otherwise the selector uses conservative regime-based priors and labels itself as warming up.
- At least two selected strategy votes must agree before a BUY/SELL signal can pass to the existing ML gate and paper risk checks. The scanner shows selected strategies and whether selection is still warming up or using performance history.
- The previous backtest is still not an adaptive-strategy walk-forward backtest. It must not be used as proof that this new selector is profitable. News/calendar integration is not included, so no active news trading or automatic news protection is claimed.


V2 ADVANCED RESEARCH + NEWS PROTECTION:
- Added a News Protection ON/OFF control to the Streamlit sidebar.
- News protection uses a user-uploaded economic-calendar CSV with columns:
  datetime_utc,currency,impact,event. datetime_utc must be UTC; only High/High Impact/3/Red events are considered.
- With News Protection ON and no valid calendar loaded, new automatic paper entries are blocked (fail-closed). Existing paper positions continue to be managed. This is intentional so a missing feed is not mistaken for protection.
- This package does NOT fetch live news/calendar data automatically. Upload and keep the calendar current. If the calendar is stale or incomplete, protection can miss events.
- Added a bounded Strategy Discovery Lab that compares five candidate strategy families on a chronological validation segment after approximate costs. It reports research metrics only and does not automatically promote a candidate. Promotion requires separate walk-forward and paper validation.
- Added `trading_engine/news_protection.py`, `trading_engine/strategy_discovery.py`, and tests.
- Still research/paper trading only. Yahoo Finance data may be delayed, and the main Streamlit app is not a guaranteed always-on background service. Cloud hosting must be tested separately.


V3 AUTOMATIC NEWS CALENDAR:
- News Protection now automatically fetches scheduled high-impact ("red-folder") economic events from Forex Factory' public economic calendar API.
- API documentation: https://showmytrades.com/docs/api/economic-calendar
- Forex Factory weekly feed refresh interval is 15 minutes, cached in the Streamlit session. No other calendar provider or CSV fallback is used.
- If the feed is unavailable and no recent cache/valid fallback exists, News Protection ON blocks new entries (fail-closed). It continues managing existing paper positions.
- API coverage is the provider's supported currencies (USD, EUR, GBP, JPY, AUD, NZD, CAD, CHF, CNY); event schedules can be revised and feeds can fail. This is scheduled economic-news protection, not a guarantee against all news or unscheduled headlines.


NEWS SOURCE UPDATE (V5): News Protection uses only Forex Factory/Fair Economy published weekly JSON feed at https://nfs.faireconomy.media/ff_calendar_thisweek.json. Only impact="High" (red-folder) events are accepted. No ShowMyTrades, third-party calendar, or uploaded CSV fallback is used. Feed refresh interval: 15 minutes. The published feed covers the current week; a successful cached result is used if refresh fails. If no valid feed/cache exists, News Protection blocks new entries. Forex Factory notices prohibit copying/redistribution of calendar data without written consent; this bot uses the public feed for local paper-trading protection and does not republish it.


V7 OPTIONAL SMC MODULE
- Optional Smart Money Concepts heuristic features: Order Blocks, BOS, CHOCH, FVG and liquidity sweeps.
- Toggle "Use SMC confluence in decisions" in the sidebar. Other strategy families remain available when SMC has no setup.
- SMC chart zone lines and a diagnostic panel are available in the selected-market view.
- Strategy Discovery Lab includes an SMC candidate for chronological validation; it does not auto-promote strategies.
- SMC definitions are algorithmic heuristics, not proof of institutional orders; paper-test and validate before relying on them.
