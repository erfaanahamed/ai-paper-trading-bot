# AI Paper Trading Brain — Version C research build

## Included
- Rule-based paper-trading dashboard for Gold futures proxy, US Oil futures proxy, selected FX pairs, and crypto tickers using Yahoo Finance data.
- EMA 9/21/50/200, RSI, MFI, Bollinger Bands (20, 2.5), Wilder-smoothed ATR/ADX/+DI/-DI, approximate session VWAP, relative volume, adaptive support/resistance and Power Channel.
- SMC-style OHLC proxies: BOS, CHOCH, liquidity sweeps, 3-candle FVG, approximate order-block zones, and premium/discount dealing-range zones.
- Rule-fit scoring (not a probability), higher-timeframe filter, configurable risk settings, Start/Pause/Stop controls, 60-second refresh-based paper scanning while the app remains open, CSV trade log, and candlestick chart with indicator/SMC markers.
- Simplified historical backtest using the selected signal timeframe and selected HTF, with configurable estimated costs and basic performance metrics.

## Run locally
1. Install Python 3.10+.
2. In this folder run: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`

## Important limitations
- PAPER TRADING ONLY. No broker, exchange, wallet, API key, or real-money orders are connected.
- Start arms the scanner; while the app page remains open and connected, the selected market is scanned about every 60 seconds. The manual scan button remains available. Cloud sleep, browser disconnection, and data-provider delays can interrupt scans. Pause/Stop stop new scans but do not liquidate open simulated positions.
- Yahoo Finance feeds may be delayed, incomplete, unavailable, or unsuitable for execution. Gold and oil are futures proxies, not necessarily your broker's XAUUSD/WTI CFD prices.
- FX volume may be absent or tick-based. MFI and relative volume are therefore not reliable on every symbol; MFI is marked unavailable when all volume is zero.
- SMC concepts are algorithmic approximations from OHLC data, not proof of actual institutional orders. Order blocks are heuristic and need visual review.
- The backtester is simplified and not event-driven. It approximates costs in price-units and does not model broker contract sizes, pip values, leverage, financing, exchange fees, spread dynamics, or realistic fills. Do not compare its virtual INR balance to real P&L.
- Automatically closed paper positions are appended to the CSV trade log. Local CSV trade history may not persist across cloud restarts/redeploys. Streamlit session state and this prototype are not a durable trading database.
- No win rate is promised. Validate with clean out-of-sample and walk-forward tests before making decisions.


## Validation note
The Python source syntax was checked after editing. This does not prove live market-data availability or profitable trading performance. Run the app and verify data retrieval and simulated trades in your own Streamlit session before relying on it.


## All-market dashboard
The dashboard attempts to scan all configured market proxies and shows each market's default-timeframe signal, HTF bias, rule-fit score, RSI, ADX, and data status. Results are cached for about 55 seconds to reduce requests. It is a signal overview only; the automatic paper-trading manager remains attached to the market selected in the sidebar. Symbol mappings and data availability vary by provider, and some markets may report unavailable data.
