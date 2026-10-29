# AI Brain Hybrid Self-Learning Research Bot

## Run on a computer
```bash
pip install -r requirements.txt
python -m pytest trading_engine/tests -q
python -m trading_engine.main_worker
```
For the original dashboard: `streamlit run app.py`
For the research viewer: `streamlit run trading_engine/app.py`

## Adaptive learning
The dashboard learns from completed paper trades only. It records trade outcomes and summarizes repeated loss patterns by market, regime, side, timeframe and strategy. These are investigation flags, not proof of causation and not a reason to disable a strategy from a few trades.

A new trade-outcome model requires at least 300 completed feature-rich trades and at least 50 wins and 50 losses. A candidate is evaluated on a chronological holdout (last 30%); it must reach AUC >= 0.53 and cannot replace a validated model if its holdout AUC is more than 0.02 worse. Model writes are atomic and a last-known-good artifact is saved before promotion. If the candidate fails, the current validated model is retained.

Strategy discovery reports candidate research results but does not automatically activate candidates from one run. Existing weighted voting and regime-based strategy selection remain in place.

## Research path
Raw OHLCV -> causal indicators -> rule ensemble -> features -> historical labels -> calibrated classifier with purged walk-forward evaluation -> probability and estimated net-expectancy gate -> BUY/SELL eligibility or WAIT/SKIP.

## Limitations
AUC and win probability are not profitability guarantees. Keep evaluating net returns after costs, profit factor, drawdown, calibration and performance across unseen regimes. Data vendors may be delayed or incomplete, symbol proxies may differ from broker instruments, and simulated fills simplify real execution. No broker/wallet integration or live trading is included.
