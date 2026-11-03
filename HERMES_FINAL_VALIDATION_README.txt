HERMES AI AGENT — CHART-ANALYSIS-ONLY BUILD
Build date: 2026-11-03

Changes:
1. Dashboard and Telegram test/open-trade labels use Hermes AI Agent.
2. Harmonic Pattern AI Agent has been removed from active code, imports, voting, scan tables, event text, dashboard UI and tests.
3. SMC and the liquidity-sweep proxy have been removed from active strategy logic.
4. Chart Analysis AI Agent is the only dedicated chart-analysis agent.
5. The system remains paper-only; no broker/exchange live-order routing is included.
6. Existing risk controls, multi-strategy voting, self-learning components, market/news filters and Telegram test-message UI remain.

Validation after this change:
- Python compileall: PASS.
- Automated tests: 36 passed.
- Streamlit Cloud deployment: NOT VERIFIED.
- Telegram delivery: NOT VERIFIED; requires user secrets and a successful in-app test.

Important limitations:
- Dashboard backtest uses the multi-strategy directional signal path and next-bar entry, but it does not reproduce all live dashboard gates or all execution details.
- The research backtest currently does not apply the 1H higher-timeframe bias filter even though the dashboard uses that filter for live paper signals.
- Costs are approximate; Yahoo Finance symbols are market-data proxies, not broker contract specifications.
- Directional vote strength and chart-agent confidence are NOT calibrated win probabilities or guaranteed win rates.

Deployment:
Upload the CONTENTS of this ZIP to the existing GitHub repository root, replacing the old files. Ensure app.py, requirements.txt and trading_engine/ are at the repository root. Then allow Streamlit Community Cloud to redeploy.

Telegram:
In Streamlit app settings -> Secrets, add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID. Never paste the bot token into ChatGPT, GitHub, screenshots or public messages. After saving and redeploying, open Trade execution notifications and press Send Telegram test message.
