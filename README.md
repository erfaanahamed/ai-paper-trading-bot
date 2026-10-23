# Brain AI Trading Bot V2 — Android/mobile edition

## Changes in this edition
- Removed the Bot Controls sidebar, Start/Pause buttons, Confirm Settings, and Reset Account button.
- Automatic paper mode starts when the app opens.
- Daily loss stop is fixed at **3%** of the virtual account's start-of-day balance. When reached, new paper entries are blocked until the next day; existing positions continue to be managed.
- Fixed AI probability threshold: strictly greater than 60%.
- Fixed virtual risk budget: 0.5% per trade; maximum 20 new paper trades per day.
- Mobile-friendly centered layout for Android browser screens.
- Select market and entry timeframe at the top of the page. H1/M30/M15 are still used for trend/setup confirmation.

## Important: using an Android phone
This is a Python Streamlit web app, not a native Android APK. The `.bat` installer works on Windows only. To use it from your Android phone, deploy the project to a cloud Streamlit host using a GitHub repository, then open the provided web address in Chrome.

### Deployment steps from Android
1. Sign in to GitHub in your phone browser and create a new repository.
2. Upload `app.py` and `requirements.txt` from this ZIP to the repository root.
3. Open Streamlit Community Cloud in your browser, connect GitHub, and choose that repository, `app.py` as the main file, then deploy.
4. Open the deployed app URL on your Android phone and use **Add to Home screen** in Chrome if you want an app-like shortcut.

Cloud-hosted filesystems may reset or be ephemeral, and free hosting may sleep. Do not assume paper-trade history/state will persist permanently. The app only updates while the web session is active; it is not a guaranteed always-on service. For durable logs, add a persistent database before relying on long-running paper tests.

## Features
- Timeframes: 5m, 15m, 30m, 1h.
- EMA 9/21/50/200, Bollinger Bands, ATR, ADX with +DI/-DI, RSI, VWAP, MFI, volume/relative volume.
- Adaptive support/resistance and ATR-buffered power channel.
- Random Forest machine-learning model trained on historical candles.
- Automatic paper BUY/SELL entries when estimated probability is >60% and hard filters pass.
- TP1 books 60%; remaining 40% stop moves to break-even + 0.05 ATR; TP2 closes remaining 40%.
- Virtual account, trade log, charts, AI estimates, and historical paper-trade win rate.

## Limitations and safety
- Paper trading only; no broker or exchange connection and no real-money orders.
- A displayed AI probability is not a verified future win rate or guarantee. Backtest and forward-test before making decisions.
- Yahoo Finance is a research feed, not a broker-grade execution feed. Intraday history can be limited.
- Gold/oil use futures proxies (`GC=F`, `CL=F`), which can differ from broker CFD prices. Forex volume is not centralized.
- Position sizing is simplified and not contract-, lot-, tick-value-, or broker-aware.
