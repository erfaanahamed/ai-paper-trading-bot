# Brain AI Trading Bot V2 — Android-friendly web app

## What changed in this version
- Scans all configured markets in one mobile-friendly table: Gold/XAUUSD proxy, US Oil/WTI proxy, EURUSD, GBPUSD, USDJPY, AUDUSD, BTCUSD, ETHUSD and SOLUSD.
- Shows H1/M30/M15 trend, AI BUY/SELL estimates, best-side estimate, historical closed paper-trade win rate, number of closed trades, and scanner status.
- Includes an all-symbol Trade History table with trade ID, open/close timestamps, side, timeframe, entry/SL/TP levels, TP1 status, exit reason, virtual P&L, open/closed status, market/status filters, and CSV export.
- Trade entries and exit records are combined by Trade ID for a single row per trade in the history view; older log rows remain viewable.
- Automatic paper trading is enabled for every listed symbol when the AI estimate is strictly above 60% and all configured filters pass.
- Daily loss limit: 3% of virtual account starting balance for the day.
- Default virtual risk budget per new trade: 0.5%; maximum 20 new paper trades per day across all symbols.
- On TP1, closes 60% and moves the remaining stop to break-even plus an ATR buffer; TP2 closes the remaining 40%.
- Uses 5m, 15m, 30m and 1h data. The selected entry timeframe applies to the all-symbol scanner.
- Paper only: no broker connection and no real-money orders.

## Metrics are not the same
- **AI Win Estimate %** is a model estimate based on historical labels; it is not a measured live win rate.
- **Paper Win Rate %** is calculated only from closed simulated trades for that market. It remains blank until there are closed paper trades.
- Neither metric guarantees future results. Backtest/forward-test and inspect transaction costs before considering any real trading.

## Run on Android
This is a Streamlit web application, not an APK. Deploy `app.py` and `requirements.txt` using Streamlit Community Cloud connected to a GitHub repository, then open the deployed app URL in Chrome. Use Chrome's menu > Add to Home screen for an app-like shortcut.

## Important operational limits
- The trade log is written to a local CSV. On ephemeral cloud hosting, history may be lost after a restart/redeploy unless you configure persistent storage or periodically download the CSV.
- The scanner refreshes while the page/session is active; it is not guaranteed to run when the browser or cloud app is sleeping.
- The included symbols are Yahoo Finance data symbols/proxies, not necessarily the exact CFD price/contract available at a broker. Gold uses `GC=F`; US Oil uses `CL=F`.
- Yahoo Finance may not provide sufficient intraday history for every symbol/timeframe. Those rows may show NO DATA or TRAINING.
- Paper state/logs may not persist permanently on ephemeral cloud hosting unless persistent storage is configured.
