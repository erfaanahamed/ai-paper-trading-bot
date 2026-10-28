AI BRAIN V7 — DEPLOYMENT FIX V2

FIX INCLUDED
- app.py now checks the normal repository-root package path and one-level nested path.
- If trading_engine is truly missing from the deployed GitHub repository, it displays a clear action message.
- 30-minute pre-close and post-open protection settings remain ON by default.
- Saturday/Sunday no-new-entry protection remains ON by default, including crypto.

IMPORTANT DIAGNOSIS
The screenshot's line-12 ModuleNotFoundError means Streamlit ran app.py but could not find trading_engine/__init__.py at the expected location. A ZIP download alone does not update GitHub. The files must be committed to the same repository/branch used by Streamlit.

DEPLOY STEPS (ANDROID)
1. Download and extract this ZIP.
2. In GitHub, open the exact repository used by Streamlit.
3. Upload the CONTENTS of this ZIP to the repository root; do not upload the ZIP as a single file. Preserve the trading_engine/ folder and its __init__.py file.
4. Commit the changes to the branch configured in Streamlit.
5. In Streamlit Manage app, confirm Main file path is app.py, then reboot.
6. If the error remains, verify the commit appears in GitHub and send a screenshot of the repository root file list.

EXPECTED ROOT
app.py
requirements.txt
trading_engine/__init__.py
trading_engine/app.py
trading_engine/runner.py
trading_engine/state.py
trading_engine_data/

RESEARCH/PAPER TRADING ONLY. No broker connection or real-money orders are included. Market session estimates must be checked against the intended broker's actual sessions.
