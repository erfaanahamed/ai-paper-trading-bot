"""Market-session entry protection for the paper-trading scanner.

Times are local exchange/session times and use IANA zones so DST is handled.
This blocks NEW entries only; it must not stop management of open positions.
"""
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

# The symbols in config.py are Yahoo Finance proxies, not broker instruments.
# Sessions below are conservative approximations for the underlying markets.
SESSIONS = {
    "Gold (XAUUSD)": {"type": "futures", "tz": "America/Chicago", "open": time(17, 0), "close": time(16, 0)},
    "XAGUSD Silver (futures proxy)": {"type": "futures", "tz": "America/Chicago", "open": time(17, 0), "close": time(16, 0)},
    "Crude Oil (WTI proxy)": {"type": "futures", "tz": "America/Chicago", "open": time(17, 0), "close": time(16, 0)},
    **{name: {"type": "fx", "tz": "America/New_York", "open": time(17, 0), "close": time(17, 0)}
       for name in ("EUR/USD", "GBP/USD", "AUD/USD", "USD/JPY")},
    **{name: {"type": "continuous", "tz": "UTC"} for name in ("BTC/USD", "SOL/USD", "ETH/USD")},
}


def entry_session_status(market, now=None, buffer_minutes=30, no_weekend_trading=True):
    """Return (allowed, reason). `now` may be an aware datetime for testing."""
    cfg = SESSIONS.get(market, {"type": "continuous", "tz": "UTC"})
    now = now or datetime.now(ZoneInfo("UTC"))
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    local = now.astimezone(ZoneInfo(cfg["tz"]))
    if no_weekend_trading and local.weekday() >= 5:
        return False, "Saturday/Sunday no-trade protection is ON"

    kind = cfg["type"]
    if kind == "continuous":
        return True, "24/7 market; no daily session close"

    mins = max(0, int(buffer_minutes))
    minute_of_day = local.hour * 60 + local.minute
    buffer = timedelta(minutes=mins)

    if kind == "futures":
        # CME metals/WTI proxy: daily maintenance break 16:00–17:00 Chicago,
        # plus the weekly closure from Friday 16:00 until Sunday 17:00.
        close_min = 16 * 60
        open_min = 17 * 60
        if local.weekday() == 4 and minute_of_day >= close_min - mins:
            return False, "Within 30-minute pre-close buffer / weekly close"
        if local.weekday() == 5:
            return False, "Futures market closed on Saturday"
        if local.weekday() == 6 and minute_of_day < open_min + mins:
            return False, "Within 30-minute post-open buffer / weekly reopen"
        if close_min - mins <= minute_of_day < open_min + mins:
            return False, "Within daily futures close/reopen protection window"
        return True, "Inside estimated futures session"

    # FX: weekly session is approximately Sunday 17:00 to Friday 17:00 New York.
    open_min = 17 * 60
    close_min = 17 * 60
    if local.weekday() == 4 and minute_of_day >= close_min - mins:
        return False, "Within 30-minute pre-close buffer / weekly FX close"
    if local.weekday() == 5:
        return False, "FX market closed on Saturday"
    if local.weekday() == 6 and minute_of_day < open_min + mins:
        return False, "Within 30-minute post-open buffer / weekly FX reopen"
    return True, "Inside estimated FX session"
