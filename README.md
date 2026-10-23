# AI Paper Trading Brain — No SMC, No Start/Pause/Stop Buttons

Changes:
- Reduced oversized selected-market metric text to compact labels and values.
- Removed SMC-specific signal conditions, chart markers, SMC status table, and SMC columns from recent candles.
- Signal scoring uses standard indicators and price-action filters only.
- Removed Start, Pause, and Stop buttons.
- Automatic paper scanning is enabled by default while the app remains open and connected.
- Kept the manual “Scan market & manage paper positions now” action.
- Python syntax compilation passed.

Deploy:
1. Replace `app.py` in the existing GitHub repository on the `main` branch with this file and commit.
2. Keep `requirements.txt` aligned with this version.
3. Wait for Streamlit Community Cloud to redeploy, then refresh the app.
4. Verify market data and paper-trading behavior.

Important:
- Paper-only; no broker or live orders are connected.
- Removing SMC changes signal logic. Test with historical backtesting before relying on results.
- Automatic scans require the app page to remain open and connected; cloud sleep/provider interruptions may stop them.
