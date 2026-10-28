AI BRAIN V7 — FULL PROJECT ZIP (IMPORT-STRUCTURE REPAIR)

WHAT WAS CHECKED
- Python syntax compilation: passed.
- Automated unit tests: 37 passed using PYTHONPATH=. pytest -q.
- ZIP contains app.py and trading_engine/ at the same top-level directory.
- Removed Python cache files and pytest cache files from the release package.

IMPORTANT: this package is for paper-trading research only. It does not connect to a broker or place real-money orders. Passing tests does not guarantee profitability or prove the live Streamlit deployment is working.

DEPLOY TO GITHUB / STREAMLIT
1. Download and extract this ZIP.
2. Upload/commit the CONTENTS of the extracted folder into the ROOT of the GitHub repository.
   Do not upload the ZIP itself. Do not put all files inside an extra nested folder.
3. Verify the repository root contains:
     app.py
     requirements.txt
     trading_engine/__init__.py
     trading_engine/app.py
     trading_engine/runner.py
     trading_engine/state.py
     trading_engine_data/
4. Streamlit app settings -> Main file path should be app.py at repository root.
5. Commit changes, then use Manage app -> Reboot app (or wait for redeploy).
6. If ModuleNotFoundError remains, inspect Manage app -> Logs. The error can also mean the deployed GitHub repository was not updated with the trading_engine/ folder.

PROTECTION SETTINGS
- Market session protection is ON by default, with a 30-minute buffer.
- Saturday/Sunday no-new-entry protection is ON by default, including crypto.
- These settings block new entries; they are not intended to stop managing existing paper positions.
- Session times are approximations for the configured Yahoo Finance market proxies. Check them against the intended broker's actual sessions before relying on them.
