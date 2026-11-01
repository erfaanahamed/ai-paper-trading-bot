AI Brain V7 - Multi-Strategy Engine + Multi-Strategy AI Agent

Integration summary:
- Keeps the existing deterministic Multi-Strategy Engine as the sole final signal authority.
- Adds trading_engine/multi_strategy_agent.py and a root-level fallback module.
- Adds a dashboard panel showing BUY/SELL/WAIT vote counts, active-vote agreement,
  current engine decision, market regime, closed trades reviewed, and advisory text.
- Agent is advisory only: no order execution, no automatic strategy-weight changes,
  and no ability to bypass existing confidence/risk/session/order checks.
- Confidence/agreement is a heuristic score, not a calibrated probability of winning.

This is a code integration, not proof of profitability. Verify on Streamlit and run
historical/paper tests before considering any live deployment.
