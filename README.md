# AI Paper Trading Brain — Dashboard Fix 2

## Fixes
- Corrected the all-market dashboard's return-value unpacking bug: `score_signal()` returns three values, not four.
- Changed the selected-market title to normal-sized text.
- Python syntax compilation passed.

## Deploy
1. Replace `app.py` in the existing GitHub repository on the `main` branch with this version and commit.
2. Keep `requirements.txt` aligned with the included version.
3. Wait for Streamlit Community Cloud to redeploy, then reboot/refresh if needed.
4. Confirm the all-market dashboard shows usable prices. If prices remain unavailable, inspect the Data status column; that may be a separate provider/ticker/timeframe issue.

Paper trading only. No guarantee of profitability or win rate.
