# AI Paper Trading Brain — display fix

## Changes
- Fixed the all-market dashboard error: the caller expected four return values from `score_signal`, but that function returns three. This caused market rows to be marked as data errors even when market data downloaded successfully.
- Changed the selected-market title from a large heading to normal-sized bold text.

## Deploy
Replace `app.py` in your GitHub repository with this version. Ensure `requirements.txt` is also present, then let Streamlit Community Cloud redeploy.

## Notes
- This app is for paper trading only and does not place real-money orders.
- Yahoo Finance may still return unavailable data for some symbols/timeframes. Those rows should now show the provider/data status instead of failing because of the return-value mismatch.
- The file passed Python syntax compilation, but live market-data availability and the hosted app still need to be checked after deployment.
