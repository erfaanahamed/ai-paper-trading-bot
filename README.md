# AI Paper Trading Brain — Backtest Diagnostics

## Version A
- Adds a **Backtest diagnostics** table after every backtest run, including when no trades pass.
- Counts common reasons historical bars are skipped/rejected, including insufficient data, higher-timeframe history, ADX, Bollinger Band, HTF bias, and rule-fit threshold.
- Keeps the existing trading rules unchanged.
- Fixes the market-dashboard `score_signal()` unpacking mismatch.

## Install
1. Extract this ZIP.
2. In your GitHub repository, replace `app.py` and `requirements.txt`.
3. Commit the changes and wait for Streamlit Community Cloud to redeploy.
4. Open the app, expand **Backtest (historical, simplified)**, and press **Run backtest**.
5. Scroll to the **Backtest diagnostics** table and send a screenshot.

## Important
This is paper-only and educational. The diagnostic counts are explanatory counts, not independent trades. They do not prove profitability. The bot does not connect to a broker or place live orders.
