# AI Paper Trading Bot — Full Clean File

This package contains the full `app.py` file, requirements, and deployment notes.

Changes included:
- Fixed all-market signal function unpacking to use three return values.
- Removed SMC-specific scoring gates, chart markers, status table, and recent-candle columns.
- Removed Start/Pause/Stop controls; automatic paper scanning defaults to RUNNING while the page is open and connected.
- Reduced selected-market metrics to compact label/value text.
- Reduced the all-market dashboard heading size.
- Python syntax compilation passed.

Deploy by replacing `app.py` in the existing GitHub repository on `main`, committing it, and waiting for Streamlit Community Cloud to redeploy.

Important: paper trading only; no live broker orders. Market data may be delayed or unavailable. Strategy behavior must be backtested and paper-tested; no win rate or profitability is guaranteed.
