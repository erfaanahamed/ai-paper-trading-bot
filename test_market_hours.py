from datetime import datetime
from zoneinfo import ZoneInfo
from trading_engine.market_hours import entry_session_status


def test_weekend_toggle_blocks_crypto_when_enabled():
    saturday = datetime(2026, 10, 31, 12, 0, tzinfo=ZoneInfo("UTC"))
    allowed, reason = entry_session_status("BTC/USD", now=saturday, no_weekend_trading=True)
    assert not allowed
    assert "Saturday/Sunday" in reason


def test_crypto_can_trade_weekend_when_toggle_off():
    saturday = datetime(2026, 10, 31, 12, 0, tzinfo=ZoneInfo("UTC"))
    allowed, _ = entry_session_status("BTC/USD", now=saturday, no_weekend_trading=False)
    assert allowed


def test_futures_daily_close_buffer_blocks_new_entries():
    # 15:45 Chicago on a Tuesday, inside the 30-minute pre-close window.
    now = datetime(2026, 10, 27, 15, 45, tzinfo=ZoneInfo("America/Chicago"))
    allowed, reason = entry_session_status("Gold (XAUUSD)", now=now, no_weekend_trading=False)
    assert not allowed
    assert "close/reopen" in reason


def test_fx_weekly_open_buffer_blocks_new_entries():
    # Sunday 17:15 New York is within 30 minutes of the weekly open.
    now = datetime(2026, 10, 25, 17, 15, tzinfo=ZoneInfo("America/New_York"))
    allowed, reason = entry_session_status("EUR/USD", now=now, no_weekend_trading=False)
    assert not allowed
    assert "post-open" in reason
