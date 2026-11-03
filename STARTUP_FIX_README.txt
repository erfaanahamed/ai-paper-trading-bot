HERMES AI AGENT — STREAMLIT STARTUP FIX

ROOT CAUSE FOUND
The dashboard called st.markdown() before st.set_page_config(). Streamlit requires
st.set_page_config() to be the first Streamlit command in the script. This can stop
the app before the dashboard renders, leaving the deployed page blank/loading.

FIX APPLIED
- Moved st.set_page_config() immediately after imports and before every other st.* call.
- Added runtime.txt to request Python 3.12 for a conservative dependency runtime.
- Added .streamlit/config.toml with standard headless server settings.

DEPLOYMENT
1. Back up the current GitHub repository.
2. Extract this ZIP.
3. Upload ALL extracted files and folders to the repository root, preserving
   the .streamlit folder and trading_engine folder.
4. Commit to the exact branch configured in Streamlit.
5. In Streamlit Manage app, verify Main file path is app.py, then Reboot.

VALIDATION LIMIT
This fix addresses a definite Streamlit startup-order defect. Unit tests and syntax
checks do not prove the hosted app works; confirm startup in the deployed logs.
Paper-only operation remains enabled. Do not use real-money execution.
