AI Brain Self-Learning Paper Trading Bot

Files:
- app.py: Streamlit paper-trading app with a trade-outcome learning layer and optional supervised ML model.
- requirements.txt: Python dependencies.

Run locally:
1. Install Python 3.10+.
2. In the project folder run: pip install -r requirements.txt
3. Run: streamlit run app.py

Learning behavior:
- Rule-adaptation layer adjusts required signal confirmations from recent completed, feature-rich paper-trade outcomes (conservative bounds: 5–7 confirmations).
- Logistic-regression model uses entry-time indicator features and completed paper-trade outcomes. It is not activated until there are at least 30 eligible completed trades, both outcome classes, and at least 8 wins and 8 losses.
- When there is not enough data or scikit-learn is unavailable, the app falls back to its rule-based strategy and reports that status.
- Learning uses completed automatic paper trades with saved entry features. Old trade records without entry features do not count toward model training.
- The model is retrained from saved trade history on app reruns; this is a simple baseline, not an advanced deep-learning system.

Important:
- Paper trading only. No broker, exchange, wallet, or real-money order connection is included.
- Yahoo Finance data can be delayed/incomplete and the existing backtest is simplified.
- Estimated win probabilities are not guarantees. Validate with out-of-sample testing before considering any live deployment.
