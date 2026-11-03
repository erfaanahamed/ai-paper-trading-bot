HERMES AI AGENT — CORRECTED VALIDATION BUILD
Build date: 2026-11-03

Changes made in this package:
1. Rebranded active dashboard and Telegram test/open-trade notification labels to Hermes AI Agent.
2. Removed SMC imports, signal voting, settings UI, chart panel, strategy-discovery candidate, implementation modules and SMC-specific tests.
3. Removed the daily-loss halt setting from the dashboard and disabled its entry/backtest halt. Other protections such as per-trade risk sizing, daily trade cap, duplicate-position controls, global open-position cap, session/news filters remain separate.
4. Switched the dashboard's Run backtest action to trading_engine.backtest, which generates historical signals through the same multi-strategy signal-selection path used by the engine. Entry is next-bar open to reduce same-bar look-ahead.
5. Kept Telegram credentials in Streamlit Secrets/environment only. The dashboard includes a Send Telegram test message button and will show Telegram's response/error.
6. Kept the system paper-only. No broker/exchange live-order routing was added.

Validation performed in this build environment:
- Python compileall: PASS
- Automated tests: 37 passed
- Telegram delivery: NOT VERIFIED; requires the user's TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in Streamlit Secrets and a successful click on the in-app test button.
- Full Streamlit cloud deployment: NOT VERIFIED from this local package.

Important backtest limitations:
- Uses the multi-strategy directional signal path but does not fully reproduce every live dashboard gate (broker-specific contract sizing, full session/news gate timing, partial TP1 execution and all execution details).
- Costs are approximate; Yahoo Finance symbols are market-data proxies, not broker contract specifications.
- Backtest results are not a guaranteed or predictive win rate.
- 70% directional vote strength is NOT a 70% probability of winning.

Deployment:
Upload the CONTENTS of this ZIP to the existing GitHub repository root, replacing the old files. Ensure app.py, requirements.txt and trading_engine/ are at the repository root. Then allow Streamlit Community Cloud to redeploy.

Telegram setup after deployment:
In Streamlit app settings -> Secrets, add:
TELEGRAM_BOT_TOKEN = "your_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"
Never paste the bot token into ChatGPT, GitHub, screenshots or public messages. After saving Secrets and redeploying, open the dashboard's Trade execution notifications panel and press Send Telegram test message. A successful API response confirms the bot can send to that chat.
