# AI Brain Hybrid Self-Learning Research Bot

## Run on a computer
```bash
pip install -r requirements.txt
python -m pytest trading_engine/tests -q
python -m trading_engine.main_worker
```
For the original dashboard: `streamlit run app.py`
For the research viewer: `streamlit run trading_engine/app.py`

## Hybrid scan path
Raw OHLCV -> causal indicator builder -> rule ensemble -> feature builder -> historical triple-barrier labels -> calibrated classifier with purged walk-forward evaluation -> probability and estimated net-expectancy gate -> BUY/SELL eligibility or WAIT/SKIP.

A model is eligible only if the current trainer has sufficient examples and its walk-forward AUC is at least 0.53. This is a research guardrail, not evidence of profitability by itself. Costs are approximate, Yahoo Finance data may be delayed or incomplete, and fills/position orchestration remain simplified.

## Important limitations
- The current scan trains a model when no qualifying saved model exists; it does not yet run a scheduled retraining lifecycle or automatically update from every completed paper trade.
- Full open-position management and end-to-end paper execution are not yet integrated into this scan loop.
- Indicator-based signals may be sparse. If no model qualifies, the safe decision is WAIT/SKIP.
- Yahoo futures/FX/crypto symbols are proxies and can differ from broker instruments.
- No broker/wallet integration or live trading. No profit or win-rate guarantee.
