# AI Paper Trading Bot — Higher-Timeframe Options Update

This package contains the full `app.py` file, dependencies, and deployment notes.

Change included: the Higher-timeframe filter dropdown now offers **15m, 30m, 1h, and 4h**.

## Deploy
1. Download and extract this ZIP.
2. Replace `app.py` in your GitHub repository (`main` branch) with the included `app.py`.
3. Verify `requirements.txt` contains the listed packages.
4. Commit the changes and wait for Streamlit Community Cloud to redeploy.
5. Reload the app, open Bot settings, choose the higher-timeframe filter, and tap **Confirm settings**.

Note: this remains a paper-trading app. Adding timeframes to the dropdown does not guarantee trades or improve performance by itself. Other filters can still reject entries, and data availability depends on Yahoo Finance.
