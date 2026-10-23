AI Paper Trading Brain — Bug Fix

Fixes:
- Fixed the all-market dashboard unpacking error. score_signal() returns three values, not four.
- Changed the selected-market title to normal-sized text.
- Kept the Confirm settings button.
- Kept higher-timeframe options: 15m, 30m, 1h, and 4h.
- Kept this app paper-only; no broker/live orders are connected.

Deploy:
1. Replace app.py and requirements.txt in your GitHub repository with these files.
2. Commit the changes to the branch used by Streamlit Cloud.
3. Wait for the app to redeploy, then reload the page.
4. Check whether the dashboard reports usable market data.

Notes:
- A WAIT signal is normal when the strategy's filters are not all satisfied.
- Rule-fit score is not a win probability.
- Yahoo Finance may still return unavailable data for some symbols/timeframes.
- This code passed an in-memory Python syntax compilation check; hosted runtime/data access has not been tested here.
