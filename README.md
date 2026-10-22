# AI Paper Trading Brain — Version C research build

## Included
- Rule-based paper-trading dashboard for Gold futures proxy, US Oil futures proxy, selected FX pairs, and crypto tickers using Yahoo Finance data.
- EMA 9/21/50/200, RSI, MFI, Bollinger Bands (20, 2.5), Wilder-smoothed ATR/ADX/+DI/-DI, approximate session VWAP, relative volume, adaptive support/resistance and Power Channel.
- SMC-style OHLC proxies: BOS, CHOCH, liquidity sweeps, 3-candle FVG, approximate order-block zones, and premium/discount dealing-range zones.
- Rule-fit scoring (not a probability), higher-timeframe filter, configurable risk settings, Start/Pause/Stop controls, button-driven paper scan, CSV trade log, and candlestick chart with indicator/SMC markers.
- Simplified historical backtest using the selected signal timeframe and selected HTF, with configurable estimated costs and basic performance metrics.

## Run locally
1. Install Python 3.10+.
2. In this folder run: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`

## Important limitations
- PAPER TRADING ONLY. No broker, exchange, wallet, API key, or real-money orders are connected.
- Start arms the scanner; scans are button-driven, not continuous/background. Pause/Stop stop new scans but do not liquidate open simulated positions.
- Yahoo Finance feeds may be delayed, incomplete, unavailable, or unsuitable for execution. Gold and oil are futures proxies, not necessarily your broker's XAUUSD/WTI CFD prices.
- FX volume may be absent or tick-based. MFI and relative volume are therefore not reliable on every symbol; MFI is marked unavailable when all volume is zero.
- SMC concepts are algorithmic approximations from OHLC data, not proof of actual institutional orders. Order blocks are heuristic and need visual review.
- The backtester is simplified and not event-driven. It approximates costs in price-units and does not model broker contract sizes, pip values, leverage, financing, exchange fees, spread dynamics, or realistic fills. Do not compare its virtual INR balance to real P&L.
- Local CSV trade history may not persist across cloud restarts/redeploys. Streamlit session state and this prototype are not a durable trading database.
- No win rate is promised. Validate with clean out-of-sample and walk-forward tests before making decisions.
