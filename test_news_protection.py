import pandas as pd
from trading_engine.news_protection import event_in_window

def test_event_inside_window_is_detected():
    now = pd.Timestamp("2026-01-01T12:00:00Z")
    df = pd.DataFrame([{
        "datetime_utc": pd.Timestamp("2026-01-01T12:10:00Z"),
        "currency": "USD", "impact": "High", "event": "CPI"
    }])
    assert event_in_window(df, {"USD"}, now=now, before_minutes=15, after_minutes=15)["event"] == "CPI"

def test_irrelevant_currency_is_ignored():
    now = pd.Timestamp("2026-01-01T12:00:00Z")
    df = pd.DataFrame([{
        "datetime_utc": pd.Timestamp("2026-01-01T12:05:00Z"),
        "currency": "JPY", "impact": "High", "event": "BoJ"
    }])
    assert event_in_window(df, {"USD"}, now=now) is None
