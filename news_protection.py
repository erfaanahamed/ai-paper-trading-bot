"""Economic calendar parsing and event-window logic used by dashboard protection."""
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

COLUMNS = ["datetime_utc", "currency", "impact", "event"]

def read_calendar(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS), "missing"
    try:
        df = pd.read_csv(path)
        if any(c not in df.columns for c in COLUMNS):
            return pd.DataFrame(columns=COLUMNS), "invalid_columns"
        df = df[COLUMNS].copy()
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
        df["currency"] = df["currency"].astype(str).str.upper().str.strip()
        df["impact"] = df["impact"].astype(str).str.title().str.strip()
        df = df.dropna(subset=["datetime_utc"])
        df = df[df["impact"].isin(["High", "High Impact", "3", "Red"])]
        return df.sort_values("datetime_utc"), ("ok" if not df.empty else "empty")
    except Exception:
        return pd.DataFrame(columns=COLUMNS), "error"

def event_in_window(df, currencies, now=None, before_minutes=30, after_minutes=30):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    if df is None or df.empty:
        return None
    matches = df[
        df["currency"].isin({str(c).upper() for c in currencies})
        & (df["datetime_utc"] >= now - pd.Timedelta(minutes=after_minutes))
        & (df["datetime_utc"] <= now + pd.Timedelta(minutes=before_minutes))
    ]
    return None if matches.empty else matches.iloc[0].to_dict()
