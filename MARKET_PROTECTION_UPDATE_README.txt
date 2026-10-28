AI BRAIN V7 — MARKET SESSION + WEEKEND PROTECTION UPDATE

Changes in this build:
- Added a Market Session Protection ON/OFF toggle.
- Added a configurable pre-close / post-open buffer; default is 30 minutes.
- Added a Saturday & Sunday: NO TRADING toggle, default ON, covering all symbols including crypto.
- Blocks new paper entries only; existing simulated positions continue to be managed.
- Protection status and reason are added to scan events when an entry is blocked.
- Kept standalone root app.py and root smc.py import compatibility.

GitHub layout requirement:
Upload the CONTENTS of this ZIP to the repository root (not the ZIP itself and not inside an extra nested folder). The repository root must show app.py, requirements.txt, smc.py, and trading_engine/ side by side.

Important limitations:
Market hours are approximate session proxies, not broker-specific schedules or holiday calendars. Crypto has no daily close; weekend blocking applies only when the weekend toggle is ON. This is a paper-trading research app and does not place live orders.
