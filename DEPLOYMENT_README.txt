HERMES AI AGENT — CLEAN ROOT DEPLOYMENT PACKAGE

IMPORTANT: The ZIP is arranged with app.py, requirements.txt, runtime.txt,
.streamlit/, and trading_engine/ directly at the archive root.
Do NOT upload the outer ZIP folder as a nested directory.

GitHub update (Android):
1. Back up/download the current repository first.
2. Extract this ZIP.
3. Upload/replace the CONTENTS of the extracted folder into the repository ROOT.
4. Confirm app.py, requirements.txt, runtime.txt, .streamlit/ and trading_engine/
   are visible at the same repository root level.
5. Commit to branch main (the Streamlit log showed branch main).
6. Streamlit Manage app > Settings: Main file path must be app.py.
7. Save, then Reboot app. Wait for dependency installation to finish.

The supplied deployment log showed dependencies installed and the Uvicorn
server started; it did NOT contain a Python traceback. It also showed Python
3.14.10, although this package requests Python 3.12. If Streamlit continues
using 3.14, verify runtime.txt is at repository ROOT and check the app's
Python-version setting in Streamlit Cloud before rebooting.

Paper-trading only. Do not connect real-money execution.
