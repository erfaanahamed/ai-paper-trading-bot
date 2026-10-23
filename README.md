# AI Paper Trading Brain — Full Files

This package contains the full `app.py`, its Python dependencies, and these deployment notes.

## Deploy on Streamlit Community Cloud
1. Download and extract this ZIP.
2. In your GitHub repository, replace `app.py` and `requirements.txt` with the files in this package.
3. Commit the changes to the `main` branch.
4. Wait for Streamlit Community Cloud to redeploy, then reload your app.

## Notes
- This app is for paper trading only. It does not connect to a broker, wallet, exchange, private key, or real-money order system.
- The backtest is simplified and approximate. If it reports no trades, that can mean the entry filters were not met in the loaded historical candles; it does not prove the strategy is profitable or broken.
- Market prices come from Yahoo Finance and may be delayed, incomplete, or unavailable. Gold uses `GC=F` as a futures proxy, not the exact quote from a broker's XAUUSD CFD.
- The rule-fit score is not a win probability. Test with out-of-sample data and realistic costs before relying on results.
