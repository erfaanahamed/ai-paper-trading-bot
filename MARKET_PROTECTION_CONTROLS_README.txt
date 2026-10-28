AI Brain V7 - Explicit Market Protection Controls

This build exposes three separate sidebar ON/OFF controls:
1. 1 HOUR BEFORE MARKET CLOSE — ON/OFF (default ON)
2. 1 HOUR AFTER MARKET OPENS — ON/OFF (default ON)
3. SATURDAY & SUNDAY — NO TRADING — ON/OFF (default ON)

The protection buffer defaults to 60 minutes and is adjustable from 0 to 180 minutes.
These protections block new paper-trade entries; existing paper positions continue to be managed.
Market hours are estimates and must be checked against the actual broker/session schedule.
Crypto has no daily close/open window; the weekend toggle can still block crypto entries.

Install: replace the files in the GitHub repository with the contents of this ZIP, commit them
to the branch deployed by Streamlit, then reboot the Streamlit app.
