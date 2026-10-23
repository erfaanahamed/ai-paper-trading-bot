# AI Paper Trading Brain — Backtest Diagnostics (A first)

## What's changed
- Added a Backtest diagnostics table that counts evaluated bars and explains common reasons trades are skipped or rejected.
- Keeps existing signal filters unchanged; this version does not loosen the strategy.
- Fixed the all-market dashboard unpacking bug: `score_signal()` returns three values, not four.
- Backtest remains paper-only and educational. A higher rule-fit score is not a win-probability estimate.

## Deploy
1. Download and extract this ZIP.
2. In your GitHub repository, replace `app.py` and `requirements.txt` with these files.
3. Commit the changes and wait for Streamlit Community Cloud to redeploy.
4. Open the app, confirm settings, select a market/timeframe with enough history, expand **Backtest (historical, simplified)**, then press **Run backtest**.
5. Read the **Backtest diagnostics** table even if no trades are found.

## Important
- This is not live trading and does not connect a broker or wallet.
- Yahoo Finance history/timeframes may be incomplete.
- Diagnostics help identify the bottleneck; they do not prove profitability.
