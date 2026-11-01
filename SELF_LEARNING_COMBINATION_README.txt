

SELF-LEARNING ENGINE + SELF-LEARNING AI AGENT
================================================
The existing adaptive_learning.py remains the learning engine: it evaluates completed
paper trades, trains a candidate model only when sample requirements are met, uses a
chronological holdout, and rejects candidates that fail its configured validation gate.

The new self_learning_agent.py is a separate advisory layer. It summarizes learning
diagnostics and suggests research actions. It has no order-execution authority and does
not automatically rewrite strategy rules. The dashboard's existing Self-learning engine
section remains the user-facing monitor.

This is an integration scaffold, not proof of improved trading performance. Continue
paper trading and validate on unseen data before any live deployment.
