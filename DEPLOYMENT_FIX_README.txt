AI Brain V7 — SMC import compatibility fix

IMPORTANT: This ZIP is a project package, not an automatic update to GitHub/Streamlit.
To deploy it, upload/commit the CONTENTS of this ZIP to the repository root (not the ZIP itself and not an extra enclosing folder).

Required root structure:
  app.py
  requirements.txt
  smc.py                    (compatibility fallback)
  trading_engine/
    __init__.py
    smc.py
    ... other Python modules ...
  trading_engine_data/

The app first imports trading_engine.smc and falls back to root smc.py if the package-level SMC file is absent. This is a compatibility safeguard; all project files should still be uploaded.

After committing files, open Streamlit Manage app -> Reboot app (or wait for redeploy). If it still fails, inspect the full logs because a redacted ModuleNotFoundError screenshot does not identify the missing module name.

This remains paper-trading research software. Syntax/import checks and unit tests do not guarantee error-free operation in every cloud environment or profitability.


DEPLOYMENT CHECKLIST
1. Upload the contents of this ZIP preserving folder paths.
2. Keep root app.py and root requirements.txt at repository root.
3. Keep trading_engine/*.py inside trading_engine/.
4. Commit the requirements.txt change so Streamlit reinstalls Plotly.
5. In Streamlit settings, Main file path must be app.py at repository root.
6. If errors persist, inspect Manage app > Logs for the full traceback.

No ZIP can guarantee a cloud deployment is error-free without testing the exact deployed repository and environment.
