HERMES AI AGENT — COMPLETE ROOT DEPLOYMENT ZIP

This package has app.py, requirements.txt, runtime.txt, .streamlit/ and
trading_engine/ directly at the ZIP root. Upload the CONTENTS into the root
of the GitHub repository (not the ZIP's enclosing folder).

IMPORTANT FINDING FROM THE PROVIDED STREAMLIT LOG:
- Streamlit used Python 3.12.16 successfully. Python 3.14 was NOT used in this
  captured deployment log.
- All listed dependencies installed successfully.
- The log ends with: Uvicorn server started on :::8501.
- There is no Python traceback or app exception in the uploaded log.
Therefore the captured log does not identify the exact reason the browser
remains on a spinner. This package pins dependencies to the versions shown
in that log to make deployment reproducible, but a successful browser render
cannot be guaranteed from this log alone.

DEPLOYMENT STEPS (GitHub on Android):
1. Back up the existing repository.
2. Extract this ZIP.
3. Upload/replace its CONTENTS at the repository ROOT on branch main.
4. Confirm app.py, requirements.txt, runtime.txt, .streamlit/ and
   trading_engine/ are all visible at the same root level.
5. In Streamlit Cloud, confirm Main file path is app.py.
6. Save changes and reboot the app.
7. If it still spins, download the NEW log after the reboot and send it.
   The next log must include the app's runtime output after Uvicorn starts.

VALIDATION PERFORMED ON THIS PACKAGE:
- Python compileall: passed.
- trading_engine test suite: 36 passed.
- Harmonic Pattern agent is not active; Chart Analysis agent remains.
- SMC/liquidity-sweep logic is not active in the current build.

This is paper-trading software only. It does not connect to a live broker
or execute real-money trades. Test with paper data only.
