AI BRAIN V7 — STANDALONE IMPORT FIX

This package uses the full standalone root app.py from the original AI Brain V7
project. It does NOT use the wrapper app.py that raises ModuleNotFoundError when
trading_engine/ is absent. The SMC import now tries root-level smc.py first.

IMPORTANT: A downloaded ZIP does not change GitHub automatically.

ANDROID DEPLOY STEPS
1. Download and extract this ZIP.
2. Open the SAME GitHub repository and branch configured in Streamlit.
3. Replace the repository's root app.py with this ZIP's app.py. Also upload/keep
   the root-level smc.py and requirements.txt from this ZIP.
4. Commit the changes. Make sure the commit is on the deployed branch.
5. In Streamlit Manage app, verify Main file path is app.py, then Reboot app.

The root app.py is the full AI Brain V7 paper-trading application. Optional
strategy-discovery code reports its own error if its package module is absent;
it should not prevent the main app from starting. No live broker or real-money
trading is enabled. Market hours from public data proxies may differ from broker
sessions.
