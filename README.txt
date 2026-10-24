AI Brain Self-Learning Paper Trading Bot — Market Session Upgrade

Files:
- app.py: Streamlit paper-trading app with the existing trade-outcome learning layer, optional ML model, all-market scanner, and estimated market-session intelligence.
- requirements.txt: Python dependencies.

Run:
1. Use Python 3.10+.
2. In the project folder run: pip install -r requirements.txt
3. Run: streamlit run app.py

Market-session changes:
- The all-market scanner displays Session status, Session detail, and Session source.
- Automatic paper entries are blocked when the configured schedule says CLOSED.
- New entries are also blocked when the latest candle timestamp is stale or cannot be verified.
- Crypto is treated as a 24/7 market schedule, but stale/missing provider data still blocks new entries.
- Forex and futures session times are approximate public schedule estimates in New York and Chicago time zones respectively. Holidays, special sessions, broker-specific CFD hours, and symbol-specific differences may not be fully represented.
- Yahoo Finance does not provide broker-confirmed tradability. This version does NOT yet connect to MT5. MT5 broker-specific session checks can only be enabled after an MT5 connector is configured.

Self-learning:
- The rule-adaptation layer adjusts required signal confirmations based on recent completed, feature-rich paper-trade outcomes (bounded to 5–7 confirmations).
- Logistic-regression ML uses entry-time indicator features and completed outcomes. It requires at least 30 eligible completed trades, both outcome classes, and at least 8 wins and 8 losses.
- Before enough eligible data exists, the bot falls back to rule-based signals.
- Older records without entry features do not count toward ML training. Model training is a simple baseline, not advanced deep learning.

Important operating limits:
- Paper trading only. No broker, exchange, wallet, or real-money order connection is included.
- Streamlit Cloud's page refresh loop is not an independent background worker. Scanning can stop when the app session disconnects or the cloud app sleeps. This package does not make the bot run continuously after the dashboard closes or the phone is off.
- True unattended execution needs a separately hosted worker (for example, a VPS), reliable persistent storage, health checks, and restart handling. VPS deployment remains a future step.
- Yahoo Finance intraday data can be delayed, incomplete, or stale. Session status from this package is an estimate, not a guarantee that a broker is accepting orders.
- Paper P&L and position sizing are simplified estimates; probabilities and backtests do not guarantee future results.
