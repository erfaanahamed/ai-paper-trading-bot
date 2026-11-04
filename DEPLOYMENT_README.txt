HERMES AI AGENT — STREAMLIT DEPLOYMENT PACKAGE

IMPORTANT: Upload the CONTENTS of this ZIP to the ROOT of your GitHub repository (branch main).
Do not upload the ZIP itself and do not put these files inside an extra folder.

Required root layout:
  app.py
  requirements.txt
  runtime.txt
  .streamlit/config.toml
  trading_engine/__init__.py
  trading_engine/*.py
  trading_engine/tests/*.py

Streamlit Community Cloud settings:
  Repository: your GitHub repository
  Branch: main
  Main file path: app.py
  Python: 3.12 (runtime.txt also requests 3.12)

After committing all files, open Streamlit Manage app -> Reboot app.
If the page remains blank, open Manage app -> Logs, download the NEW app runtime log
(not only the dependency-install log), and share it. The supplied Nov 4 log shows
dependencies installed and Uvicorn started, but contains no Python traceback; it does
not identify the cause of the blank browser page.

This package is for PAPER TRADING. It does not guarantee profitable trades or live
execution. Keep paper mode enabled while validating data feeds and risk controls.
