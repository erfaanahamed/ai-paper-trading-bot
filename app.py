import math
from datetime import datetime, timezone, date, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import json
from pathlib import Path
import requests
from trading_engine.agents import chart_analysis_agent, harmonic_pattern_agent
try:
    from trading_engine.adaptive_learning import update_model as update_adaptive_model, learning_diagnostics
    from trading_engine.strategy_discovery import discover_strategies
except Exception:
    update_adaptive_model = None
    learning_diagnostics = None
    discover_strategies = None
# Standalone-first SMC import: the main app.py contains the full application and
# must not require importing trading_engine.app. Prefer the root-level smc.py so
# deployments still start even if GitHub's upload omitted the package directory.
try:
    from smc import add_smc_features, smc_signal
except ModuleNotFoundError:
    try:
        from trading_engine.smc import add_smc_features, smc_signal
    except ModuleNotFoundError as _smc_import_error:
        raise ModuleNotFoundError(
            "SMC module is missing. Upload the root-level smc.py file with app.py."
        ) from _smc_import_error


# -----------------------------
# News protection (calendar CSV)
# -----------------------------
NEWS_DATA_DIR = Path("trading_engine_data")
NEWS_CALENDAR_PATH = NEWS_DATA_DIR / "economic_calendar.csv"
NEWS_CALENDAR_COLUMNS = ["datetime_utc", "currency", "impact", "event"]

def _fetch_automatic_news_calendar():
    """Fetch only Forex Factory's published weekly calendar and keep red-folder events."""
    cache_key = "auto_news_calendar_cache"
    cache_time_key = "auto_news_calendar_cache_time"
    status_key = "auto_news_calendar_status"
    now = datetime.now(timezone.utc)
    cached_at = st.session_state.get(cache_time_key)
    cached = st.session_state.get(cache_key)
    # Refresh every 15 minutes; reuse the last successful feed between refreshes.
    if cached is not None and cached_at and (now - cached_at).total_seconds() < 900:
        return cached.copy(), st.session_state.get(status_key, "Forex Factory calendar cached.")

    # Forex Factory/Fair Economy published weekly calendar feed. This is not a third-party calendar.
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "AIBrainPaperTrader/5.0"},
            timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("Forex Factory feed returned an unexpected format")
        rows = []
        for item in payload:
            # Strictly accept High impact only (Forex Factory red folder). No medium/low/holiday events.
            if str(item.get("impact", "")).strip().lower() != "high":
                continue
            event_time = item.get("date")
            if not event_time:
                continue
            parsed_time = pd.to_datetime(event_time, utc=True, errors="coerce")
            if pd.isna(parsed_time):
                continue
            currency = str(item.get("country", "")).strip().upper()
            if currency not in ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF", "CNY"]:
                continue
            rows.append({
                "datetime_utc": parsed_time,
                "currency": currency,
                "impact": "High",
                "event": str(item.get("title", "Economic event")),
                "source": "Forex Factory weekly calendar feed",
            })
        frame = pd.DataFrame(rows, columns=NEWS_CALENDAR_COLUMNS + ["source"])
        if frame.empty:
            raise ValueError("Forex Factory feed returned no parseable red-folder events")
        frame["datetime_utc"] = pd.to_datetime(frame["datetime_utc"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["datetime_utc"]).sort_values("datetime_utc").reset_index(drop=True)
        st.session_state[cache_key] = frame.copy()
        st.session_state[cache_time_key] = now
        status = f"Forex Factory connected · red-folder events only · refreshed {now.strftime('%H:%M:%S UTC')}"
        st.session_state[status_key] = status
        return frame, status
    except Exception as exc:
        # Fail closed if there is no successful cached Forex Factory response.
        if cached is not None and not cached.empty:
            msg = f"Forex Factory refresh failed; using last successful Forex Factory cache. Error: {str(exc)[:100]}"
            st.session_state[status_key] = msg
            return cached.copy(), msg
        return pd.DataFrame(columns=NEWS_CALENDAR_COLUMNS), f"Forex Factory automatic feed unavailable: {str(exc)[:140]}"

def load_news_calendar():
    """Use only Forex Factory's automatic weekly feed; never substitute another provider or CSV."""
    return _fetch_automatic_news_calendar()

def relevant_news_currencies(market_name):
    """Currencies most relevant to the selected instrument; this is an approximation."""
    name = str(market_name).upper()
    if "EUR/USD" in name: return {"EUR", "USD"}
    if "GBP/USD" in name: return {"GBP", "USD"}
    if "AUD/USD" in name: return {"AUD", "USD"}
    if "USD/JPY" in name: return {"USD", "JPY"}
    if any(x in name for x in ["GOLD", "XAU", "SILVER", "XAG", "CRUDE", "OIL", "BTC", "ETH", "SOL"]):
        return {"USD"}
    return {"USD"}

def news_protection_status(market_name, now_utc=None):
    """Return (blocked, reason, next_event); fail closed when protection is ON but calendar is missing."""
    enabled = bool(st.session_state.get("news_protection_enabled", True))
    if not enabled:
        return False, "News protection OFF", None
    calendar, status = load_news_calendar()
    if calendar.empty:
        return True, "News protection ON but no valid calendar is loaded; new entries blocked for safety.", None
    now_utc = now_utc or datetime.now(timezone.utc)
    pre_mins = int(st.session_state.get("news_pre_minutes", 30))
    post_mins = int(st.session_state.get("news_post_minutes", 30))
    relevant = relevant_news_currencies(market_name)
    upcoming = calendar[
        calendar["currency"].isin(relevant)
        & (calendar["datetime_utc"] >= pd.Timestamp(now_utc) - pd.Timedelta(minutes=post_mins))
        & (calendar["datetime_utc"] <= pd.Timestamp(now_utc) + pd.Timedelta(minutes=pre_mins))
    ]
    if not upcoming.empty:
        event = upcoming.iloc[0]
        when = event["datetime_utc"].strftime("%Y-%m-%d %H:%M UTC")
        label = f"{event['currency']} {event['event']} ({when})"
        return True, f"High-impact news window: {label}", label
    future = calendar[(calendar["currency"].isin(relevant)) & (calendar["datetime_utc"] > pd.Timestamp(now_utc))]
    next_event = None if future.empty else f"{future.iloc[0]['currency']} {future.iloc[0]['event']} — {future.iloc[0]['datetime_utc'].strftime('%Y-%m-%d %H:%M UTC')}"
    return False, "No matching high-impact event inside the configured protection window.", next_event

# Optional ML dependencies: the rule-based bot still runs if scikit-learn is unavailable.
try:
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

st.set_page_config(page_title="AI Paper Trading Brain", page_icon="🧠", layout="wide")

# -----------------------------
# Configuration / data helpers
# -----------------------------
STARTING_BALANCE = 100.0  # USD-denominated virtual starting balance
MARKETS = {
    "Gold (XAUUSD)": {"symbol": "GC=F", "signal_tf": "5m", "allowed_tfs": ["5m", "15m", "30m"], "htf": "1h", "kind": "gold"},
    "XAGUSD Silver (futures proxy)": {"symbol": "SI=F", "signal_tf": "15m", "allowed_tfs": ["5m", "15m", "30m"], "htf": "1h", "kind": "silver"},
    "Crude Oil (WTI proxy)": {"symbol": "CL=F", "signal_tf": "15m", "allowed_tfs": ["5m", "15m", "30m"], "htf": "1h", "kind": "oil"},
    "EUR/USD": {"symbol": "EURUSD=X", "signal_tf": "15m", "allowed_tfs": ["5m", "15m", "30m"], "htf": "1h", "kind": "forex"},
    "GBP/USD": {"symbol": "GBPUSD=X", "signal_tf": "15m", "allowed_tfs": ["5m", "15m", "30m"], "htf": "1h", "kind": "forex"},
    "AUD/USD": {"symbol": "AUDUSD=X", "signal_tf": "15m", "allowed_tfs": ["5m", "15m", "30m"], "htf": "1h", "kind": "forex"},
    "USD/JPY": {"symbol": "JPY=X", "signal_tf": "15m", "allowed_tfs": ["5m", "15m", "30m"], "htf": "1h", "kind": "forex"},
    "BTC/USD": {"symbol": "BTC-USD", "signal_tf": "15m", "allowed_tfs": ["5m", "15m", "30m"], "htf": "1h", "kind": "crypto"},
    "SOL/USD": {"symbol": "SOL-USD", "signal_tf": "15m", "allowed_tfs": ["5m", "15m", "30m"], "htf": "1h", "kind": "crypto"},
    "ETH/USD": {"symbol": "ETH-USD", "signal_tf": "15m", "allowed_tfs": ["5m", "15m", "30m"], "htf": "1h", "kind": "crypto"},
}
INTERVAL_MAP = {"5m": ("5m", "5d"), "15m": ("15m", "60d"), "30m": ("30m", "60d"), "1h": ("1h", "60d")}

# Agreed global safety settings. Confidence is directional vote alignment, not a calibrated win probability.
MIN_CONFIDENCE_SCORE = 70
MAX_GLOBAL_OPEN_POSITIONS = 5


TRADE_LOG_PATH = Path("paper_trade_history.csv")
def persist_trade_log(trades):
    try:
        pd.DataFrame(trades).to_csv(TRADE_LOG_PATH, index=False)
    except Exception:
        pass

def load_trade_log():
    try:
        if TRADE_LOG_PATH.exists():
            return pd.read_csv(TRADE_LOG_PATH).to_dict(orient="records")
    except Exception:
        pass
    return []

def market_schedule_status(market_name, now_utc=None):
    """Approximate exchange/session status; broker holidays and symbol-specific hours may differ."""
    now_utc = now_utc or datetime.now(timezone.utc)
    kind = MARKETS.get(market_name, {}).get("kind", "forex")
    if kind == "crypto":
        return "OPEN 24/7", "Always open (crypto market)"
    ny = now_utc.astimezone(ZoneInfo("America/New_York"))
    # Common FX and CME-style metals/energy weekly schedule, approximate in ET.
    weekday, mins = ny.weekday(), ny.hour * 60 + ny.minute
    if weekday == 5 or (weekday == 6 and mins < 18 * 60) or (weekday == 4 and mins >= 17 * 60):
        return "CLOSED", "Weekend closure (approx. reopens Sunday 18:00 ET)"
    if 17 * 60 <= mins < 18 * 60:
        return "DAILY BREAK", "Daily maintenance break (approx. 17:00–18:00 ET)"
    return "OPEN", "Weekday session (approximate hours; verify broker schedule)"

def market_entry_protection_status(market_name, now_utc=None):
    """Return (blocked, reason) for NEW entries only. Session hours are approximate."""
    now_utc = now_utc or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    cfg = MARKETS.get(market_name, {})
    kind = cfg.get("kind", "forex")
    # Strict, fixed protection window requested by the user: 60 minutes.
    # Do not read a stale Streamlit session-state slider value; it could silently leave this at 30.
    buffer_minutes = 60
    preclose_on = bool(st.session_state.get("preclose_protection_enabled", True))
    postopen_on = bool(st.session_state.get("postopen_protection_enabled", True))

    tz_name = "UTC" if kind == "crypto" else ("America/Chicago" if kind in ("gold", "silver", "oil") else "America/New_York")
    local = now_utc.astimezone(ZoneInfo(tz_name))
    minute = local.hour * 60 + local.minute

    # Weekend switch applies to every symbol, including 24/7 crypto.
    weekend_on = bool(st.session_state.get("weekend_no_trade_enabled", True))
    if weekend_on and local.weekday() >= 5:
        return True, f"Saturday/Sunday no-trading is ON ({tz_name})"

    if not (preclose_on or postopen_on):
        return False, "Market close/open protections are OFF"

    # Crypto has no daily exchange close/open window; weekend toggle still applies.
    if kind == "crypto":
        return False, "Crypto trades continuously; daily close/open protection not applicable"

    if kind in ("gold", "silver", "oil"):
        # Approximate CME-style maintenance break, Chicago time: 16:00 close / 17:00 reopen.
        close_min, open_min = 16 * 60, 17 * 60
        # The maintenance break itself is always blocked.
        if close_min <= minute < open_min:
            return True, "Market maintenance break; no new entries"
        if preclose_on and close_min - buffer_minutes <= minute < close_min:
            return True, f"Within {buffer_minutes} minutes before market close"
        if postopen_on and open_min <= minute < open_min + buffer_minutes:
            return True, f"Within {buffer_minutes} minutes after market reopening"
        # Approximate weekly close/reopen windows.
        if local.weekday() == 4 and minute >= close_min:
            return True, "Weekly market closure"
        if local.weekday() == 6 and minute < open_min:
            return True, "Weekly market closure before reopening"
        return False, "Outside configured market close/open protection windows"

    # Approximate FX rollover at 17:00 New York time.
    rollover = 17 * 60
    if local.weekday() == 4 and minute >= rollover:
        return True, "Weekly FX market closure"
    if local.weekday() == 6 and minute < rollover:
        return True, "FX market closed before Sunday reopening"
    if preclose_on and rollover - buffer_minutes <= minute < rollover:
        return True, f"Within {buffer_minutes} minutes before FX rollover/close"
    if postopen_on and rollover <= minute < rollover + buffer_minutes:
        return True, f"Within {buffer_minutes} minutes after FX rollover/reopening"
    return False, "Outside configured market close/open protection windows"


def current_global_sessions(now_utc=None):
    now_utc = now_utc or datetime.now(timezone.utc)
    sessions = [
        ("Sydney", "Australia/Sydney", 8, 17),
        ("Tokyo", "Asia/Tokyo", 9, 18),
        ("London", "Europe/London", 8, 17),
        ("New York", "America/New_York", 8, 17),
    ]
    rows = []
    for name, tz_name, start_hour, end_hour in sessions:
        local = now_utc.astimezone(ZoneInfo(tz_name))
        is_open = local.weekday() < 5 and start_hour <= local.hour < end_hour
        rows.append({"Session": name, "Status": "OPEN" if is_open else "CLOSED", "Local time": local.strftime("%H:%M"), "Session hours": f"{start_hour:02d}:00–{end_hour:02d}:00 local"})
    return pd.DataFrame(rows)

def download_data(symbol, interval):
    yf_interval, period = INTERVAL_MAP[interval]
    try:
        df = yf.download(symbol, period=period, interval=yf_interval, auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={c: str(c).title() for c in df.columns})
        required = ["Open", "High", "Low", "Close"]
        if not all(c in df.columns for c in required):
            return pd.DataFrame()
        if "Volume" not in df.columns:
            df["Volume"] = 0.0
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df
    except Exception:
        return pd.DataFrame()

def add_indicators(df):
    if df.empty:
        return df
    x = df.copy()
    c, h, l, o = x["Close"], x["High"], x["Low"], x["Open"]
    # RSI using Wilder-style exponentially smoothed average gains/losses.
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss.ne(0), 100.0).where(avg_gain.ne(0) | avg_loss.ne(0), 50.0)
    x["RSI"] = rsi

    typical = (h + l + c) / 3
    raw_flow = typical * x["Volume"].fillna(0).clip(lower=0)
    flow_delta = typical.diff()
    pos_flow = raw_flow.where(flow_delta > 0, 0).rolling(14, min_periods=14).sum()
    neg_flow = raw_flow.where(flow_delta < 0, 0).rolling(14, min_periods=14).sum()
    mfr = pos_flow / neg_flow.replace(0, np.nan)
    mfi = 100 - (100 / (1 + mfr))
    mfi = mfi.where(neg_flow.ne(0), 100.0).where(pos_flow.ne(0) | neg_flow.ne(0), 50.0)
    # When the feed has no usable volume, MFI is not meaningful; mark it unavailable.
    if x["Volume"].fillna(0).sum() <= 0:
        mfi[:] = np.nan
    x["MFI"] = mfi

    x["BB_Mid"] = c.rolling(20, min_periods=20).mean()
    bb_std = c.rolling(20, min_periods=20).std(ddof=0)
    x["BB_Upper"] = x["BB_Mid"] + 2.5 * bb_std
    x["BB_Lower"] = x["BB_Mid"] - 2.5 * bb_std

    prev_close = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_close).abs(), (l-prev_close).abs()], axis=1).max(axis=1)
    x["ATR"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    up = h.diff()
    down = -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr_safe = x["ATR"].replace(0, np.nan)
    plus_sm = plus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    minus_sm = minus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    x["PLUS_DI"] = 100 * plus_sm / atr_safe
    x["MINUS_DI"] = 100 * minus_sm / atr_safe
    di_sum = (x["PLUS_DI"] + x["MINUS_DI"]).replace(0, np.nan)
    dx = 100 * (x["PLUS_DI"] - x["MINUS_DI"]).abs() / di_sum
    x["ADX"] = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    x["EMA9"] = c.ewm(span=9, adjust=False).mean()
    x["EMA21"] = c.ewm(span=21, adjust=False).mean()
    x["EMA50"] = c.ewm(span=50, adjust=False).mean()
    x["EMA200"] = c.ewm(span=200, adjust=False).mean()

    # Approximate intraday VWAP; Yahoo FX/crypto volume can be absent or non-centralized.
    idx = pd.to_datetime(x.index)
    session = pd.Series(idx.date, index=x.index)
    vol = x["Volume"].fillna(0).clip(lower=0)
    denom = vol.groupby(session).cumsum().replace(0, np.nan)
    x["VWAP"] = (typical * vol).groupby(session).cumsum() / denom
    x["VWAP"] = x["VWAP"].fillna(x["BB_Mid"])

    x["VolAvg20"] = x["Volume"].rolling(20).mean()
    x["RelVol"] = x["Volume"] / x["VolAvg20"].replace(0, np.nan)
    # Adaptive support/resistance + power channel.
    # The lookback scales with timeframe: shorter for 5m, longer for higher TFs.
    # Caller adds timeframe-specific settings via add_adaptive_levels().
    x["Support"] = x["Low"].rolling(20).min().shift(1)
    x["Resistance"] = x["High"].rolling(20).max().shift(1)
    x["PowerMid"] = (x["Support"] + x["Resistance"]) / 2
    x["PowerWidth"] = (x["Resistance"] - x["Support"])
    x["PowerUpper"] = x["Resistance"]
    x["PowerLower"] = x["Support"]
    # Simple candle reversal patterns.
    body = (c-o).abs()
    candle_range = (h-l).replace(0, np.nan)
    lower_wick = np.minimum(o, c) - l
    upper_wick = h - np.maximum(o, c)
    x["BullReversal"] = ((c > o) & ((lower_wick > body * 1.2) | ((c > o.shift(1)) & (o < c.shift(1)) & (c > o.shift(1)))).fillna(False))
    x["BearReversal"] = ((c < o) & ((upper_wick > body * 1.2) | ((c < o.shift(1)) & (o > c.shift(1)) & (c < o.shift(1)))).fillna(False))
    return x

def adaptive_lookback(interval):
    """Choose a support/resistance and power-channel lookback by chart timeframe."""
    return {"5m": 48, "15m": 40, "30m": 36, "1h": 30}.get(interval, 40)

def add_adaptive_levels(df, interval):
    """Calculate rolling swing levels and an adaptive power channel for the selected timeframe."""
    if df.empty:
        return df
    x = df.copy()
    n = adaptive_lookback(interval)
    # Use prior candles only to avoid using the current candle's own high/low as its level.
    x["Support"] = x["Low"].rolling(n, min_periods=max(8, n//2)).min().shift(1)
    x["Resistance"] = x["High"].rolling(n, min_periods=max(8, n//2)).max().shift(1)
    x["PowerMid"] = (x["Support"] + x["Resistance"]) / 2
    x["PowerWidth"] = x["Resistance"] - x["Support"]
    # ATR buffer makes channel boundaries respond to volatility.
    atr = x["ATR"] if "ATR" in x.columns else pd.Series(np.nan, index=x.index)
    x["PowerUpper"] = x["Resistance"] + 0.25 * atr
    x["PowerLower"] = x["Support"] - 0.25 * atr
    x["NearSupport"] = x["Close"] <= (x["Support"] + 0.35 * atr)
    x["NearResistance"] = x["Close"] >= (x["Resistance"] - 0.35 * atr)
    return x

def htf_bias(htf):
    if htf.empty or len(htf) < 205:
        return "Unknown", None
    h = add_indicators(htf)
    last = h.iloc[-1]
    if last["Close"] > last["EMA50"] and last["Close"] > last["EMA200"]:
        return "Bullish", last
    if last["Close"] < last["EMA50"] and last["Close"] < last["EMA200"]:
        return "Bearish", last
    return "Mixed", last

# -----------------------------
# Learning engine (trade-outcome adaptation + ML)
# -----------------------------
ML_FEATURES = [
    "ML_RSI", "ML_ADX", "ML_DI_Gap", "ML_EMA_Gap_Pct",
    "ML_ATR_Pct", "ML_VWAP_Gap_Pct", "ML_RelVol", "ML_MFI",
    "ML_Score", "ML_HTF_Bull", "ML_HTF_Bear", "ML_IsBuy",
]
MIN_ML_TRADES = 300
LEARNING_MODEL_VERSION = "adaptive-paper-v1"


def extract_ml_features(last, bias, score, side):
    """Create stable numeric features from the signal candle; no future data is used."""
    close = float(last.get("Close", np.nan))
    safe_close = max(abs(close), 1e-9) if np.isfinite(close) else np.nan
    def num(key, default=np.nan):
        value = last.get(key, default)
        try:
            value = float(value)
            return value if np.isfinite(value) else default
        except Exception:
            return default
    ema9, ema21 = num("EMA9"), num("EMA21")
    atr = num("ATR")
    vwap = num("VWAP")
    plus_di, minus_di = num("PLUS_DI"), num("MINUS_DI")
    return {
        "ML_RSI": num("RSI"), "ML_ADX": num("ADX"),
        "ML_DI_Gap": plus_di - minus_di if np.isfinite(plus_di) and np.isfinite(minus_di) else np.nan,
        "ML_EMA_Gap_Pct": (ema9 - ema21) / safe_close * 100 if np.isfinite(ema9) and np.isfinite(ema21) else np.nan,
        "ML_ATR_Pct": atr / safe_close * 100 if np.isfinite(atr) and np.isfinite(safe_close) else np.nan,
        "ML_VWAP_Gap_Pct": (close - vwap) / safe_close * 100 if np.isfinite(close) and np.isfinite(vwap) and np.isfinite(safe_close) else np.nan,
        "ML_RelVol": num("RelVol"), "ML_MFI": num("MFI"),
        "ML_Score": float(score), "ML_HTF_Bull": int(bias == "Bullish"),
        "ML_HTF_Bear": int(bias == "Bearish"), "ML_IsBuy": int(side == "BUY"),
    }


def _closed_learning_rows(trades):
    """Only use completed records with recorded entry features and a numeric realized P&L."""
    if not trades:
        return pd.DataFrame()
    frame = pd.DataFrame(trades)
    if not all(c in frame.columns for c in ML_FEATURES):
        return pd.DataFrame()
    pnl_col = next((c for c in ["PnL (estimated price-units)", "PnL (price units)"] if c in frame.columns), None)
    if not pnl_col:
        return pd.DataFrame()
    frame["_pnl"] = pd.to_numeric(frame[pnl_col], errors="coerce")
    frame = frame[frame[ML_FEATURES].notna().any(axis=1) & frame["_pnl"].notna()].copy()
    if frame.empty:
        return frame
    # Do not train on trades still open; only rows with a close/exit outcome qualify.
    if "Status" in frame.columns:
        frame = frame[frame["Status"].astype(str).str.contains("CLOSED", case=False, na=False)]
    if "Exit reason" in frame.columns:
        frame = frame[frame["Exit reason"].notna()]
    frame["_label"] = (frame["_pnl"] > 0).astype(int)
    return frame


def learning_profile(trades):
    """Build a conservative learning profile from completed paper trades only.

    Model promotion requires at least 300 completed feature-rich trades, at least
    50 wins and 50 losses, and chronological holdout AUC >= 0.53. A previously
    validated model remains active if a new candidate fails validation.
    """
    rows = _closed_learning_rows(trades)
    profile = {"closed_samples": int(len(rows)), "recent_win_rate": None,
               "min_confirmations": 6, "model": None, "model_ready": False,
               "model_note": "Collecting completed paper trades with entry features.",
               "learning_diagnostics": None, "model_auc": None, "model_brier": None,
               "model_status": "COLLECTING"}
    if len(rows):
        recent = rows.tail(20)
        win_rate = float(recent["_label"].mean())
        profile["recent_win_rate"] = win_rate
        if len(recent) >= 10 and win_rate < 0.40:
            profile["min_confirmations"] = 7
        elif len(recent) >= 10 and win_rate > 0.65:
            profile["min_confirmations"] = 5

    if learning_diagnostics is not None:
        try:
            profile["learning_diagnostics"] = learning_diagnostics(trades)
        except Exception:
            profile["learning_diagnostics"] = None

    if not SKLEARN_AVAILABLE:
        profile["model_note"] = "scikit-learn is not installed; rule analysis remains active and ML is disabled."
        return profile

    if update_adaptive_model is not None:
        try:
            model_path = Path("trading_engine_data/models/paper_trade_meta_model.pkl")
            result = update_adaptive_model(trades, ML_FEATURES, model_path)
            profile["model"] = result.get("model")
            profile["model_ready"] = result.get("model") is not None
            profile["model_status"] = str(result.get("status", "COLLECTING")).upper()
            profile["model_auc"] = result.get("auc")
            profile["model_brier"] = result.get("brier")
            if result.get("model_ready"):
                profile["model_ready"] = True
            profile["model_note"] = result.get("reason", "Adaptive model status unavailable.")
            if profile["model_ready"]:
                profile["model_note"] += " A previously validated champion is retained if a candidate fails."
            return profile
        except Exception as exc:
            profile["model_note"] = f"Adaptive model update failed safely; rule-based signal retained: {str(exc)[:120]}"
            return profile

    profile["model_note"] = "Adaptive learning module unavailable; rule-based signal remains active."
    return profile


def apply_learning_gate(side, last, bias, score, profile):
    """Filter a rule signal only after enough labeled trades exist; otherwise preserve baseline rules."""
    if side not in ("BUY", "SELL") or not profile.get("model_ready"):
        return side, None, "ML not applied; using rule-based signal." if side in ("BUY", "SELL") else "No rule-based signal."
    try:
        features = extract_ml_features(last, bias, score, side)
        x = pd.DataFrame([{k: features.get(k, np.nan) for k in ML_FEATURES}], columns=ML_FEATURES)
        win_probability = float(profile["model"].predict_proba(x)[0][1])
        recent_wr = profile.get("recent_win_rate")
        cutoff = 0.55
        if recent_wr is not None and recent_wr < 0.40:
            cutoff = 0.62
        elif recent_wr is not None and recent_wr > 0.65:
            cutoff = 0.52
        if win_probability < cutoff:
            return "WAIT", win_probability, f"ML filter blocked setup ({win_probability:.0%} estimated win probability; required {cutoff:.0%})."
        return side, win_probability, f"ML filter passed ({win_probability:.0%} estimated win probability; threshold {cutoff:.0%})."
    except Exception as exc:
        return side, None, f"ML filter error; baseline signal retained ({str(exc)[:80]})."


def score_signal(last, bias, threshold=None, min_confirmations=6):
    """Brain-style directional decision from trend, momentum, DI, VWAP and volatility.
    Returns an internal diagnostic strength for compatibility; it is not an entry threshold.
    """
    def ok(key):
        value = last.get(key, np.nan)
        return pd.notna(value) and bool(value)

    close = last.get("Close", np.nan)
    ema9, ema21 = last.get("EMA9", np.nan), last.get("EMA21", np.nan)
    ema50, ema200 = last.get("EMA50", np.nan), last.get("EMA200", np.nan)
    rsi, adx = last.get("RSI", np.nan), last.get("ADX", np.nan)
    plus_di, minus_di = last.get("PLUS_DI", np.nan), last.get("MINUS_DI", np.nan)
    vwap = last.get("VWAP", np.nan)
    bb_mid = last.get("BB_Mid", np.nan)
    mfi = last.get("MFI", np.nan)
    relvol = last.get("RelVol", np.nan)

    # Directional confirmation: no signal confirmation and no requirement to touch a Bollinger Band.
    buy_checks = {
        "HTF bullish": bias == "Bullish",
        "EMA9 above EMA21": pd.notna(ema9) and pd.notna(ema21) and ema9 > ema21,
        "Price above EMA50": pd.notna(ema50) and close > ema50,
        "DI bullish": pd.notna(plus_di) and pd.notna(minus_di) and plus_di > minus_di,
        "RSI supports upside": pd.notna(rsi) and 50 <= rsi <= 72,
        "Price above VWAP": pd.notna(vwap) and close > vwap,
        "Price above Bollinger midline": pd.notna(bb_mid) and close > bb_mid,
        "ADX usable": pd.notna(adx) and adx >= 12,
        "MFI not extremely weak": pd.isna(mfi) or mfi >= 35,
        "Volume confirmation": pd.isna(relvol) or relvol >= 0.7,
    }
    sell_checks = {
        "HTF bearish": bias == "Bearish",
        "EMA9 below EMA21": pd.notna(ema9) and pd.notna(ema21) and ema9 < ema21,
        "Price below EMA50": pd.notna(ema50) and close < ema50,
        "DI bearish": pd.notna(plus_di) and pd.notna(minus_di) and minus_di > plus_di,
        "RSI supports downside": pd.notna(rsi) and 28 <= rsi <= 50,
        "Price below VWAP": pd.notna(vwap) and close < vwap,
        "Price below Bollinger midline": pd.notna(bb_mid) and close < bb_mid,
        "ADX usable": pd.notna(adx) and adx >= 12,
        "MFI not extremely strong": pd.isna(mfi) or mfi <= 65,
        "Volume confirmation": pd.isna(relvol) or relvol >= 0.7,
    }
    buy_count = sum(bool(v) for v in buy_checks.values())
    sell_count = sum(bool(v) for v in sell_checks.values())
    buy_strength = round(100 * buy_count / len(buy_checks))
    sell_strength = round(100 * sell_count / len(sell_checks))

    # Require a clear majority of aligned signals, with trend + DI + HTF mandatory.
    buy_mandatory = buy_checks["HTF bullish"] and buy_checks["EMA9 above EMA21"] and buy_checks["DI bullish"]
    sell_mandatory = sell_checks["HTF bearish"] and sell_checks["EMA9 below EMA21"] and sell_checks["DI bearish"]
    if buy_mandatory and buy_count >= int(min_confirmations):
        return "BUY", int(buy_strength), buy_checks
    if sell_mandatory and sell_count >= int(min_confirmations):
        return "SELL", int(sell_strength), sell_checks
    return "WAIT", int(max(buy_strength, sell_strength)), buy_checks if buy_strength >= sell_strength else sell_checks


@st.cache_data(ttl=55, show_spinner=False)
def detect_market_regime(last):
    """Classify broad conditions using only the latest completed indicator row."""
    adx = last.get("ADX", np.nan)
    close, mid = last.get("Close", np.nan), last.get("BB_Mid", np.nan)
    upper, lower = last.get("BB_Upper", np.nan), last.get("BB_Lower", np.nan)
    atr = last.get("ATR", np.nan)
    width = ((upper - lower) / abs(mid)) if pd.notna(upper) and pd.notna(lower) and pd.notna(mid) and mid else np.nan
    atr_pct = (atr / abs(close) * 100) if pd.notna(atr) and pd.notna(close) and close else np.nan
    if pd.notna(adx) and adx >= 25:
        regime = "TRENDING"
    elif pd.notna(width) and width < 0.012:
        regime = "SQUEEZE / QUIET"
    elif pd.notna(adx) and adx < 18:
        regime = "RANGING"
    else:
        regime = "MIXED"
    return regime, width, atr_pct


def evaluate_multi_strategy(last, bias="Unknown"):
    """Rule-based strategy ensemble; separate votes, not a claim of universal profitability.
    Strategies: trend, mean reversion, breakout, momentum, range, volatility expansion,
    candle/price action, and liquidity-sweep proxy. News strategy requires an external feed.
    """
    def n(key):
        v = last.get(key, np.nan)
        return float(v) if pd.notna(v) else np.nan
    c, o, h, l = n("Close"), n("Open"), n("High"), n("Low")
    prev_c = n("PrevClose")
    ema9, ema21, ema50, ema200 = n("EMA9"), n("EMA21"), n("EMA50"), n("EMA200")
    rsi, adx, pdi, mdi = n("RSI"), n("ADX"), n("PLUS_DI"), n("MINUS_DI")
    bb_mid, bb_up, bb_low = n("BB_Mid"), n("BB_Upper"), n("BB_Lower")
    sup, res, atr, relvol, vwap = n("Support"), n("Resistance"), n("ATR"), n("RelVol"), n("VWAP")
    mfi = n("MFI")
    votes = {"Trend following": "WAIT", "Mean reversion": "WAIT", "Breakout": "WAIT",
             "Momentum": "WAIT", "Range trading": "WAIT", "Volatility expansion": "WAIT",
             "Price action": "WAIT", "Liquidity sweep (proxy)": "WAIT"}
    # Trend following
    if all(np.isfinite(x) for x in [ema9, ema21, ema50, pdi, mdi, adx]):
        if ema9 > ema21 > ema50 and pdi > mdi and adx >= 18 and (bias != "Bearish"):
            votes["Trend following"] = "BUY"
        elif ema9 < ema21 < ema50 and mdi > pdi and adx >= 18 and (bias != "Bullish"):
            votes["Trend following"] = "SELL"
    # Mean reversion: oversold/overbought extreme near outer band, filtered by moderate ADX.
    if all(np.isfinite(x) for x in [c, bb_low, bb_up, rsi, adx]):
        if c <= bb_low and rsi <= 30 and adx < 28 and (not np.isfinite(mfi) or mfi < 30): votes["Mean reversion"] = "BUY"
        elif c >= bb_up and rsi >= 70 and adx < 28 and (not np.isfinite(mfi) or mfi > 70): votes["Mean reversion"] = "SELL"
    # Breakout of previous rolling support/resistance (levels are shifted to avoid look-ahead).
    if all(np.isfinite(x) for x in [c, sup, res, atr]):
        if c > res and (not np.isfinite(relvol) or relvol >= 1.1): votes["Breakout"] = "BUY"
        elif c < sup and (not np.isfinite(relvol) or relvol >= 1.1): votes["Breakout"] = "SELL"
    # Momentum
    if all(np.isfinite(x) for x in [c, prev_c, rsi, pdi, mdi]):
        if c > prev_c and 55 <= rsi <= 78 and pdi > mdi: votes["Momentum"] = "BUY"
        elif c < prev_c and 22 <= rsi <= 45 and mdi > pdi: votes["Momentum"] = "SELL"
    # Range trading: low ADX and price near range edge, without a confirmed breakout.
    if all(np.isfinite(x) for x in [c, sup, res, adx]) and res > sup and adx < 20:
        span = res - sup
        if c <= sup + 0.18 * span: votes["Range trading"] = "BUY"
        elif c >= res - 0.18 * span: votes["Range trading"] = "SELL"
    # Volatility expansion: wide candle relative to ATR, aligned with candle direction and volume.
    if all(np.isfinite(x) for x in [c, o, h, l, atr]) and atr > 0 and (h-l) >= 1.25*atr and c != o:
        if c > o and (not np.isfinite(relvol) or relvol >= 1.0): votes["Volatility expansion"] = "BUY"
        elif c < o and (not np.isfinite(relvol) or relvol >= 1.0): votes["Volatility expansion"] = "SELL"
    # Price action: strong body closing near candle extreme, with a simple trend alignment.
    if all(np.isfinite(x) for x in [c, o, h, l, ema21]) and h > l:
        body = abs(c-o); close_position = (c-l)/(h-l)
        if c > o and body/(h-l) >= 0.60 and close_position >= 0.8 and c > ema21: votes["Price action"] = "BUY"
        elif c < o and body/(h-l) >= 0.60 and close_position <= 0.2 and c < ema21: votes["Price action"] = "SELL"
    # Liquidity-sweep proxy: prior rolling level swept intrabar then close returns inside.
    prev_sup = n("PrevSupport"); prev_res = n("PrevResistance")
    if all(np.isfinite(x) for x in [c, h, l, prev_sup, prev_res]):
        if l < prev_sup and c > prev_sup: votes["Liquidity sweep (proxy)"] = "BUY"
        elif h > prev_res and c < prev_res: votes["Liquidity sweep (proxy)"] = "SELL"
    # Optional SMC vote: explicit heuristic features are added to candle frames upstream.
    # It is one strategy family among several and is never mandatory for all entries.
    votes["SMC confluence (OB/BOS/CHOCH/FVG/sweep)"] = smc_signal(last) if st.session_state.get("smc_enabled", True) else "WAIT"
    buys = sum(v == "BUY" for v in votes.values())
    sells = sum(v == "SELL" for v in votes.values())
    regime, width, atr_pct = detect_market_regime(last)
    # Require two independent strategy votes and reject ties/conflicting majorities.
    if buys >= 2 and buys > sells: side = "BUY"
    elif sells >= 2 and sells > buys: side = "SELL"
    else: side = "WAIT"
    strength = round(100 * max(buys, sells) / len(votes))
    return side, strength, votes, regime, buys, sells



def adaptive_strategy_selection(votes, regime, market_name, timeframe, trades):
    """Weighted strategy voting with chronological train/validation checks.

    Closed trade outcomes are attributed only to the strategies recorded as supporting
    the entry. For each strategy, older outcomes form the training window and the most
    recent 30% form a holdout validation window. Learned weights are used only when
    sample-size and holdout expectancy checks pass; otherwise conservative regime priors
    keep the system in warm-up. This is a lightweight walk-forward-style validation,
    not a guarantee of future performance.
    """
    regime_priors = {
        "TRENDING": ["Trend following", "Momentum", "Breakout", "Price action", "Volatility expansion"],
        "RANGING": ["Mean reversion", "Range trading", "Liquidity sweep (proxy)"],
        "SQUEEZE / QUIET": ["Breakout", "Volatility expansion", "Range trading"],
        "MIXED": ["Trend following", "Breakout", "Mean reversion", "Momentum", "Price action", "Range trading"],
    }
    priors = regime_priors.get(regime, regime_priors["MIXED"])
    df = pd.DataFrame(trades or [])
    perf = {}
    strategy_outcomes = {}
    if not df.empty and "Strategy names" in df.columns and "PnL (estimated price-units)" in df.columns:
        df = df.copy()
        if "Status" in df.columns:
            df = df[df["Status"].astype(str).str.upper().eq("CLOSED")]
        if "Market" in df.columns:
            df = df[df["Market"].astype(str).eq(str(market_name))]
        if "TF" in df.columns:
            df = df[df["TF"].astype(str).eq(str(timeframe))]
        df["_pnl"] = pd.to_numeric(df["PnL (estimated price-units)"], errors="coerce")
        time_col = next((c for c in ["Time", "Closed", "Closed at"] if c in df.columns), None)
        if time_col:
            df["_time"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
            df = df.sort_values("_time", kind="stable")
        for _, row in df.dropna(subset=["_pnl"]).iterrows():
            names = [x.strip() for x in str(row.get("Strategy names", "")).split("; ") if x.strip() and x.strip() != "nan"]
            for strategy in names:
                strategy_outcomes.setdefault(strategy, []).append(float(row["_pnl"]))
                perf.setdefault(strategy, []).append(float(row["_pnl"]))

    candidates = list(dict.fromkeys(list(votes.keys()) + priors))
    weights = {}
    validation_notes = []
    for strategy in candidates:
        outcomes = strategy_outcomes.get(strategy, [])
        # Conservative warm-up weights: regime-fit strategies receive a modest prior.
        base = 1.15 if strategy in priors else 0.85
        if len(outcomes) < 20:
            weights[strategy] = base
            continue
        split = int(len(outcomes) * 0.70)
        train = np.asarray(outcomes[:split], dtype=float)
        holdout = np.asarray(outcomes[split:], dtype=float)
        if len(train) < 12 or len(holdout) < 5 or not np.isfinite(train).all() or not np.isfinite(holdout).all():
            weights[strategy] = base
            continue
        train_wr = (float(np.sum(train > 0)) + 2.5) / (len(train) + 5.0)
        train_edge = float(np.mean(train))
        holdout_wr = float(np.mean(holdout > 0))
        holdout_edge = float(np.mean(holdout))
        # Only deploy a learned weight when the untouched chronological holdout
        # supports positive expectancy; otherwise fall back to the regime prior.
        if holdout_edge > 0 and holdout_wr >= 0.50 and train_edge > 0:
            evidence = max(0.0, train_wr - 0.5) + 0.25 * np.tanh(train_edge)
            weights[strategy] = float(np.clip(base + 4.0 * evidence, 0.5, 2.5))
            validation_notes.append(strategy)
        else:
            weights[strategy] = max(0.5, base * 0.75)

    # Normalize weights across all strategies so BUY/SELL scores are comparable.
    total_weight = sum(weights.values()) or 1.0
    buy_weight = sum(weights.get(name, 1.0) for name, signal in votes.items() if signal == "BUY")
    sell_weight = sum(weights.get(name, 1.0) for name, signal in votes.items() if signal == "SELL")
    buy_score = 100.0 * buy_weight / total_weight
    sell_score = 100.0 * sell_weight / total_weight
    buy_count = sum(signal == "BUY" for signal in votes.values())
    sell_count = sum(signal == "SELL" for signal in votes.values())
    # Entry threshold is based on the share of active directional weight, not
    # the total weight of strategies that did not emit a signal.
    active_directional_weight = buy_weight + sell_weight
    buy_support = 100.0 * buy_weight / active_directional_weight if active_directional_weight else 0.0
    sell_support = 100.0 * sell_weight / active_directional_weight if active_directional_weight else 0.0
    vote_threshold = 70.0
    if buy_count >= 2 and buy_support >= vote_threshold and buy_weight > sell_weight:
        side = "BUY"
    elif sell_count >= 2 and sell_support >= vote_threshold and sell_weight > buy_weight:
        side = "SELL"
    else:
        side = "WAIT"

    selected = sorted(candidates, key=lambda name: weights.get(name, 1.0), reverse=True)[:3]
    active_votes = dict(votes)
    # Keep every strategy signal visible; selected lists the top three weights.
    # Display vote scores remain percentages, not raw strategy counts.
    note = ("Walk-forward weighted voting; 70% active-vote threshold; "
            f"BUY support {buy_support:.1f}%, SELL support {sell_support:.1f}%" if validation_notes else
            "Walk-forward warm-up with regime priors; 70% active-vote threshold; "
            f"BUY support {buy_support:.1f}%, SELL support {sell_support:.1f}%")
    return side, selected, active_votes, round(buy_score, 1), round(sell_score, 1), note, {"pnl": perf, "weights": weights, "validated": validation_notes}



def scan_all_market_signals():
    """Fetch and evaluate each configured market for the overview dashboard.
    This reports signals only; it does not open paper positions for every market.
    """
    rows = []
    profile = learning_profile(st.session_state.get("trades", load_trade_log())) if "trades" in st.session_state else learning_profile(load_trade_log())
    for name, config in MARKETS.items():
        tf = config["signal_tf"]
        try:
            raw_market = download_data(config["symbol"], tf)
            htf_market = download_data(config["symbol"], config["htf"])
            if raw_market.empty or len(raw_market) < 60:
                rows.append({
                    "Market": name, "Symbol": config["symbol"], "TF": tf,
                    "Last close": None, "HTF bias": "Unknown",
                    "Signal": "DATA UNAVAILABLE", "Signal strength (diagnostic)": None,
                    "RSI": None, "ADX": None, "Data status": "Insufficient candles or provider returned no data"
                })
                continue
            market_data = add_smc_features(add_adaptive_levels(add_indicators(raw_market), tf))
            market_data["PrevClose"] = market_data["Close"].shift(1)
            market_data["PrevSupport"] = market_data["Low"].rolling(20).min().shift(2)
            market_data["PrevResistance"] = market_data["High"].rolling(20).max().shift(2)
            htf_processed = add_adaptive_levels(add_indicators(htf_market), config["htf"]) if not htf_market.empty else htf_market
            bias_value, _ = htf_bias(htf_processed) if not htf_processed.empty else ("Unknown", None)
            last_row = market_data.iloc[-1]
            chart_agent = chart_analysis_agent(market_data, bias_value)
            harmonic_agent = harmonic_pattern_agent(market_data)
            _, raw_score, strategy_votes, regime, raw_buys, raw_sells = evaluate_multi_strategy(last_row, bias_value)
            # Agents contribute to the final vote only when their own confidence is at least 70%.
            strategy_votes["Chart Analysis AI Agent"] = (
                chart_agent["side"] if chart_agent["side"] in ("BUY", "SELL") and chart_agent["confidence"] >= 70 else "WAIT")
            strategy_votes["Harmonic Pattern AI Agent"] = (
                harmonic_agent["side"] if harmonic_agent["side"] in ("BUY", "SELL") and harmonic_agent["confidence"] >= 70 else "WAIT")
            best_side, selected_strategies, active_votes, buy_votes, sell_votes, adaptive_note, _ = adaptive_strategy_selection(
                strategy_votes, regime, name, tf, st.session_state.get("trades", load_trade_log()))
            best_score = round(max(buy_votes, sell_votes))
            best_side, ml_prob, ml_note = apply_learning_gate(best_side, last_row, bias_value, best_score, profile)
            signal_value = best_side if best_side in ("BUY", "SELL") else "WAIT"
            rows.append({
                "Market": name, "Symbol": config["symbol"], "TF": tf,
                "Last close": round(float(last_row["Close"]), 5),
                "HTF bias": bias_value, "Signal": (best_side if best_side in ("BUY", "SELL") else "WAIT"),
                "Signal strength (diagnostic)": int(best_score),
                "Market regime": regime, "Strategy BUY votes": buy_votes, "Strategy SELL votes": sell_votes,
                "Selected strategies": ", ".join(selected_strategies), "Selection mode": adaptive_note,
                "Chart Analysis Agent": chart_agent["side"],
                "Chart Agent confidence (%)": chart_agent["confidence"],
                "Harmonic Agent": harmonic_agent["side"],
                "Harmonic pattern": harmonic_agent["pattern"],
                "Harmonic confidence (%)": harmonic_agent["confidence"],
                "ML win probability": round(ml_prob * 100, 1) if ml_prob is not None else None,
                "RSI": round(float(last_row["RSI"]), 1) if pd.notna(last_row["RSI"]) else None,
                "ADX": round(float(last_row["ADX"]), 1) if pd.notna(last_row["ADX"]) else None,
                "Data status": "OK"
            })
        except Exception as exc:
            rows.append({
                "Market": name, "Symbol": config["symbol"], "TF": tf,
                "Last close": None, "HTF bias": "Unknown",
                "Signal": "DATA ERROR", "Signal strength (diagnostic)": None,
                "RSI": None, "ADX": None, "Data status": str(exc)[:120]
            })
    return pd.DataFrame(rows)


def make_trade_levels(last, side, risk_pct, rr2):
    """Create strictly ordered, directionally valid SL/TP levels."""
    entry = float(last["Close"])
    atr = float(last["ATR"])
    if not np.isfinite(entry) or not np.isfinite(atr) or atr <= 0:
        return None

    rr2 = max(float(rr2), 1.5)
    if side == "BUY":
        stop = min(float(last["Low"]), entry - 1.2 * atr)
        risk = entry - stop
        if not np.isfinite(risk) or risk <= 0:
            return None
        # BUY targets must both be ABOVE entry; TP2 must be beyond TP1.
        tp1 = entry + 0.9 * risk
        fallback_tp2 = entry + rr2 * risk
        candidates = [fallback_tp2]
        for key in ("BB_Upper", "Resistance"):
            value = last.get(key, np.nan)
            if pd.notna(value) and float(value) > tp1:
                candidates.append(float(value))
        farther = [v for v in candidates if v > tp1]
        tp2 = min(farther) if farther else max(fallback_tp2, tp1 + 0.1 * risk)
        if not (stop < entry < tp1 < tp2):
            return None
    elif side == "SELL":
        stop = max(float(last["High"]), entry + 1.2 * atr)
        risk = stop - entry
        if not np.isfinite(risk) or risk <= 0:
            return None
        # SELL targets must both be BELOW entry; TP2 must be beyond TP1.
        tp1 = entry - 0.9 * risk
        fallback_tp2 = entry - rr2 * risk
        candidates = [fallback_tp2]
        for key in ("BB_Lower", "Support"):
            value = last.get(key, np.nan)
            if pd.notna(value) and float(value) < tp1:
                candidates.append(float(value))
        farther = [v for v in candidates if v < tp1]
        tp2 = max(farther) if farther else min(fallback_tp2, tp1 - 0.1 * risk)
        if not (tp2 < tp1 < entry < stop):
            return None
    else:
        return None
    values = (entry, stop, tp1, tp2)
    if not all(np.isfinite(v) for v in values):
        return None
    if side == "BUY" and not (stop < entry < tp1 < tp2):
        return None
    if side == "SELL" and not (tp2 < tp1 < entry < stop):
        return None
    return {"entry": entry, "stop": stop, "tp1": tp1, "tp2": tp2, "risk_per_unit": abs(entry-stop)}


DEFAULT_ESTIMATED_COST_BPS = 2.0  # illustrative round-trip cost estimate; tune for the venue

def cost_adjusted_breakeven(entry, side, cost_bps=DEFAULT_ESTIMATED_COST_BPS):
    """Return a conservative per-unit breakeven stop including estimated costs.

    BUY stops sit above entry; SELL stops sit below entry. This is a simplified
    price-based estimate, not a broker-specific commission/spread model.
    """
    entry = float(entry)
    buffer = abs(entry) * max(0.0, float(cost_bps)) / 10000.0
    return entry + buffer if str(side).upper() == "BUY" else entry - buffer


def simulate_auto_paper_scan(data, market_name, symbol, timeframe, bias, threshold,
                             risk_pct, rr2, balance, trades_today, max_trades_per_day,
                             daily_start_balance, max_daily_loss_pct, open_positions, allow_new_entries=True):
    """
    Process the latest CLOSED candle once per market/timeframe, update exits for existing
    simulated positions, and open at most one new paper position per scan.
    Price-distance sizing is an approximation, not broker contract-aware sizing.
    """
    events = []
    if data.empty or len(data) < 60:
        return balance, trades_today, daily_start_balance, open_positions, events, "Insufficient candle history."

    d = data.copy()
    # Use the last candle as latest available; live feeds may still include an unfinished candle.
    last = d.iloc[-1]
    bar_time = str(d.index[-1])
    market_key = f"{market_name}|{timeframe}"
    processed = st.session_state.get("auto_last_processed", {})
    already_processed = processed.get(market_key) == bar_time

    # Manage open positions on every scan, including repeated scans of the same candle.
    # This lets the simulator react if the provider updates the current candle's OHLC.
    bar = last
    still_open = []
    closed_records = []
    for pos in open_positions:
        if pos["Market"] != market_name:
            still_open.append(pos)
            continue
        side = pos["Side"]
        stop_hit = (bar["Low"] <= pos["Stop"]) if side == "BUY" else (bar["High"] >= pos["Stop"])
        tp1_hit = (bar["High"] >= pos["TP1"]) if side == "BUY" else (bar["Low"] <= pos["TP1"])
        tp2_hit = (bar["High"] >= pos["TP2"]) if side == "BUY" else (bar["Low"] <= pos["TP2"])
        # Conservative bar logic: if stop and a target are touched in the same candle, stop first.
        if stop_hit:
            qty = pos["Qty"]
            if pos["TP1_Hit"]:
                pnl = pos["RealizedPnL"] + ((pos["Stop"]-pos["Entry"]) if side=="BUY" else (pos["Entry"]-pos["Stop"])) * qty * 0.4
            else:
                pnl = ((pos["Stop"]-pos["Entry"]) if side=="BUY" else (pos["Entry"]-pos["Stop"])) * qty
            balance += pnl
            closed_at = float(pos["Stop"])
            closed_records.append({
                "Time": datetime.now().isoformat(timespec="seconds"), "Opened": pos.get("Opened"),
                "Market": pos.get("Market"), "Symbol": pos.get("Symbol"), "TF": pos.get("Timeframe"),
                "Side": side, "Score": pos.get("Score"), "Entry": pos.get("Entry"),
                "Stop": pos.get("InitialStop", pos.get("Stop")), "TP1": pos.get("TP1"), "TP2": pos.get("TP2"),
                "Exit": closed_at, "Exit reason": "SL / break-even stop", "Qty": qty,
                "PnL (estimated price-units)": pnl, "Status": "CLOSED",
                "Strategy names": pos.get("Strategy names", ""), "Selected strategy set": pos.get("Selected strategy set", ""),
                "Market regime": pos.get("Market regime", "Unknown"),
                **{k: pos.get(k, np.nan) for k in ML_FEATURES}
            })
            events.append(f"Closed {side} {market_name} at stop-loss; estimated paper P&L {pnl:.2f} price-units.")
            continue
        if tp1_hit and not pos["TP1_Hit"]:
            qty = pos["Qty"]
            partial = ((pos["TP1"]-pos["Entry"]) if side=="BUY" else (pos["Entry"]-pos["TP1"])) * qty * 0.6
            pos["RealizedPnL"] += partial
            pos["TP1_Hit"] = True
            cost_bps = float(pos.get("EstimatedCostBps", DEFAULT_ESTIMATED_COST_BPS))
            pos["Stop"] = cost_adjusted_breakeven(pos["Entry"], side, cost_bps)
            pos["TrailExtreme"] = float(bar["High"] if side == "BUY" else bar["Low"])
            events.append(
                f"TP1 hit on {market_name}; simulated 60% close and remaining 40% stop moved "
                f"to cost-adjusted breakeven {pos['Stop']:.5f} ({cost_bps:.1f} bps estimate)."
            )
        # After TP1, trail the remaining 40% using 1.5 x ATR. The stop may
        # tighten into profit but must never move backward or cross the
        # cost-adjusted breakeven floor/ceiling.
        if pos["TP1_Hit"] and not tp2_hit:
            atr_value = bar.get("ATR", np.nan)
            if pd.notna(atr_value) and float(atr_value) > 0:
                extreme_key = "TrailExtreme"
                current_extreme = float(bar["High"] if side == "BUY" else bar["Low"])
                previous_extreme = float(pos.get(extreme_key, current_extreme))
                extreme = max(previous_extreme, current_extreme) if side == "BUY" else min(previous_extreme, current_extreme)
                pos[extreme_key] = extreme
                distance = 1.5 * float(atr_value)
                candidate = extreme - distance if side == "BUY" else extreme + distance
                old_stop = float(pos["Stop"])
                breakeven_floor = cost_adjusted_breakeven(pos["Entry"], side, float(pos.get("EstimatedCostBps", DEFAULT_ESTIMATED_COST_BPS)))
                new_stop = max(old_stop, breakeven_floor, candidate) if side == "BUY" else min(old_stop, breakeven_floor, candidate)
                if (side == "BUY" and new_stop > old_stop) or (side == "SELL" and new_stop < old_stop):
                    pos["Stop"] = new_stop
                    events.append(f"ATR trailing stop updated for {market_name} to {new_stop:.5f} (1.5 x ATR).")
        if tp2_hit:
            qty = pos["Qty"]
            pnl = pos["RealizedPnL"] + ((pos["TP2"]-pos["Entry"]) if side=="BUY" else (pos["Entry"]-pos["TP2"])) * qty * 0.4
            if not pos["TP1_Hit"]:
                # If TP2 is reached without TP1 being recorded in this bar, still model TP1 then TP2.
                partial = ((pos["TP1"]-pos["Entry"]) if side=="BUY" else (pos["Entry"]-pos["TP1"])) * qty * 0.6
                pnl = partial + ((pos["TP2"]-pos["Entry"]) if side=="BUY" else (pos["Entry"]-pos["TP2"])) * qty * 0.4
            balance += pnl
            closed_records.append({
                "Time": datetime.now().isoformat(timespec="seconds"), "Opened": pos.get("Opened"),
                "Market": pos.get("Market"), "Symbol": pos.get("Symbol"), "TF": pos.get("Timeframe"),
                "Side": side, "Score": pos.get("Score"), "Entry": pos.get("Entry"),
                "Stop": pos.get("InitialStop", pos.get("Stop")), "TP1": pos.get("TP1"), "TP2": pos.get("TP2"),
                "Exit": float(pos["TP2"]), "Exit reason": "TP2", "Qty": qty,
                "PnL (estimated price-units)": pnl, "Status": "CLOSED",
                "Strategy names": pos.get("Strategy names", ""), "Selected strategy set": pos.get("Selected strategy set", ""),
                "Market regime": pos.get("Market regime", "Unknown"),
                **{k: pos.get(k, np.nan) for k in ML_FEATURES}
            })
            events.append(f"TP2 hit on {market_name}; position closed, estimated paper P&L {pnl:.2f} price-units.")
            continue
        still_open.append(pos)
    open_positions = still_open
    if closed_records:
        if "trades" not in st.session_state:
            st.session_state.trades = load_trade_log()
        st.session_state.trades.extend(closed_records)
        persist_trade_log(st.session_state.trades)

    # Reset daily risk baseline when the date changes.
    today = datetime.now().date().isoformat()
    daily_state = st.session_state.get("auto_daily_state", {})
    if daily_state.get("date") != today:
        daily_start_balance = balance
        trades_today = 0
        daily_state = {"date": today}
        st.session_state.auto_daily_state = daily_state

    # Pause mode still manages open positions but blocks all new entries.
    if not allow_new_entries:
        processed[market_key] = bar_time
        st.session_state.auto_last_processed = processed
        return balance, trades_today, daily_start_balance, open_positions, events, "Paused: existing positions managed; new entries disabled."

    daily_loss_pct = (daily_start_balance - balance) / max(daily_start_balance, 1) * 100
    if daily_loss_pct >= max_daily_loss_pct:
        events.append("Daily loss limit reached; automatic paper entries paused.")
        processed[market_key] = bar_time
        st.session_state.auto_last_processed = processed
        return balance, trades_today, daily_start_balance, open_positions, events, "Daily loss limit reached."

    if trades_today >= max_trades_per_day:
        events.append("Daily maximum trade count reached; no new entry opened.")
        processed[market_key] = bar_time
        st.session_state.auto_last_processed = processed
        return balance, trades_today, daily_start_balance, open_positions, events, "Daily trade cap reached."

    if not st.session_state.get("symbol_trade_enabled", {}).get(market_name, True):
        events.append(f"{market_name}: trading is OFF; existing positions were still managed, but no new entry was allowed.")
        processed[market_key] = bar_time
        st.session_state.auto_last_processed = processed
        return balance, trades_today, daily_start_balance, open_positions, events, "Symbol trading is OFF."

    session_blocked, session_reason = market_entry_protection_status(market_name)
    if session_blocked:
        events.append(f"{market_name}: MARKET SESSION BLOCK — {session_reason}. Existing positions remain managed; no new entry opened.")
        processed[market_key] = bar_time
        st.session_state.auto_last_processed = processed
        return balance, trades_today, daily_start_balance, open_positions, events, "Market session protection blocked new entry."

    news_blocked, news_reason, news_next = news_protection_status(market_name)
    if news_blocked:
        events.append(f"{market_name}: NEWS PROTECTION BLOCK — {news_reason}")
        processed[market_key] = bar_time
        st.session_state.auto_last_processed = processed
        return balance, trades_today, daily_start_balance, open_positions, events, "News protection blocked new entry."

    if already_processed:
        processed[market_key] = bar_time
        st.session_state.auto_last_processed = processed
        return balance, trades_today, daily_start_balance, open_positions, events, "Existing positions checked; this candle was already evaluated for entry."

    # Global duplicate guard: only one open position per market across all timeframes/agents.
    if any(p.get("Market") == market_name and p.get("Status", "OPEN") == "OPEN" for p in open_positions):
        events.append("An open simulated position already exists for this market across timeframes; duplicate entry blocked.")
        processed[market_key] = bar_time
        st.session_state.auto_last_processed = processed
        return balance, trades_today, daily_start_balance, open_positions, events, "Position already open for this market."

    # Global cap is across every configured market and timeframe, not per symbol.
    global_open_count = sum(1 for p in open_positions if p.get("Status", "OPEN") == "OPEN")
    if global_open_count >= MAX_GLOBAL_OPEN_POSITIONS:
        events.append(f"Global open-position cap reached ({MAX_GLOBAL_OPEN_POSITIONS}); no new position opened.")
        processed[market_key] = bar_time
        st.session_state.auto_last_processed = processed
        return balance, trades_today, daily_start_balance, open_positions, events, "Global open-position cap reached."

    current_trades = st.session_state.get("trades", load_trade_log())
    profile = learning_profile(current_trades)
    _, raw_score, checks, regime, raw_buy_votes, raw_sell_votes = evaluate_multi_strategy(last, bias)
    chart_agent = chart_analysis_agent(d, bias)
    harmonic_agent = harmonic_pattern_agent(d)
    # Only sufficiently confident agent outputs are added as directional votes.
    checks["Chart Analysis AI Agent"] = (
        chart_agent["side"] if chart_agent["side"] in ("BUY", "SELL") and chart_agent["confidence"] >= 70 else "WAIT")
    checks["Harmonic Pattern AI Agent"] = (
        harmonic_agent["side"] if harmonic_agent["side"] in ("BUY", "SELL") and harmonic_agent["confidence"] >= 70 else "WAIT")
    side, selected_strategies, active_votes, buy_votes, sell_votes, adaptive_note, strategy_perf = adaptive_strategy_selection(
        checks, regime, market_name, timeframe, current_trades)
    score = round(max(buy_votes, sell_votes))
    events.append(
        f"Chart Analysis Agent: {chart_agent['side']} ({chart_agent['confidence']}%); "
        f"Harmonic Agent: {harmonic_agent['side']} {harmonic_agent['pattern']} "
        f"({harmonic_agent['confidence']}%). Agent scores are heuristic, not calibrated win probabilities."
    )
    if side in ("BUY", "SELL") and score < MIN_CONFIDENCE_SCORE:
        side, ml_probability = "WAIT", None
        ml_note = (f"Confidence filter blocked setup ({score}% directional vote alignment; "
                   f"minimum {MIN_CONFIDENCE_SCORE}%). This score is not a calibrated win probability.")
    else:
        side, ml_probability, ml_note = apply_learning_gate(side, last, bias, score, profile)
    if side in ("BUY", "SELL"):
        events.append(f"Strategy selector: {adaptive_note}; selected: {', '.join(selected_strategies)}. {ml_note}")
        levels = make_trade_levels(last, side, risk_pct, rr2)
        if levels:
            entry = float(levels["entry"])
            stop = float(levels["stop"])
            # Use the same validated entry and targets from the level calculator.
            risk_unit = abs(entry - stop)
            if risk_unit > 0:
                risk_cash = balance * risk_pct / 100
                qty = risk_cash / risk_unit
                # Use the validated levels returned by make_trade_levels.
                tp1 = float(levels["tp1"])
                tp2 = float(levels["tp2"])
                # Final safety guard: never open a trade with targets on the wrong side.
                valid_levels = (
                    stop < entry < tp1 < tp2 if side == "BUY"
                    else tp2 < tp1 < entry < stop
                )
                if not valid_levels:
                    events.append(f"Rejected {market_name}: invalid stop/target ordering.")
                    processed[market_key] = bar_time
                    st.session_state.auto_last_processed = processed
                    return balance, trades_today, daily_start_balance, open_positions, events, "Invalid trade levels rejected."
                pos = {
                    "Opened": datetime.now().isoformat(timespec="seconds"), "Market": market_name,
                    "Symbol": symbol, "Timeframe": timeframe, "Side": side, "Score": score, "ConfidenceScore": score,
                    "Entry": entry, "Stop": stop, "InitialStop": stop, "TP1": tp1, "TP2": tp2,
                    "Strategy names": "; ".join([k for k,v in active_votes.items() if v == side]),
                    "Selected strategy set": "; ".join(selected_strategies), "Market regime": regime,
                    "Qty": qty, "RiskBudget": risk_cash, "TP1_Hit": False, "RealizedPnL": 0.0,
                    "EstimatedCostBps": DEFAULT_ESTIMATED_COST_BPS,
                    **extract_ml_features(last, bias, score, side), "Status": "OPEN"
                }
                open_positions.append(pos)
                trades_today += 1
                events.append(f"Automatically opened paper {side} on {market_name} at {entry:.5f}; selected strategies: {', '.join(k for k,v in active_votes.items() if v == side)}.")
            else:
                events.append("Signal rejected: stop distance was zero.")
    else:
        events.append(f"No entry: setup did not pass the filters (Signal strength {score}/100). {ml_note}")

    processed[market_key] = bar_time
    st.session_state.auto_last_processed = processed
    return balance, trades_today, daily_start_balance, open_positions, events, "Scan complete."

def diagnose_backtest_filters(df, htf, timeframe, htf_timeframe, threshold, max_daily_loss_pct):
    """Count why historical bars did not produce entries; does not change strategy rules."""
    counts = {
        "Bars in loaded data": int(len(df)),
        "Bars evaluated for signals": 0,
        "Skipped: need at least 250 candles": 0,
        "Skipped: higher-timeframe data unavailable": 0,
        "Skipped: not enough higher-timeframe history": 0,
        "Skipped: daily loss limit reached": 0,
        "Skipped: cooldown after prior trade": 0,
        "Skipped: indicators not ready": 0,
        "Rejected: HTF bias not aligned": 0,
        "Rejected: ADX hard filter (must be <25)": 0,
        "Rejected: Bollinger close hard filter": 0,
        "Rejected: Signal confirmation not met": 0,
        "Signals passing all filters": 0,
    }
    if len(df) < 250:
        counts["Skipped: need at least 250 candles"] = 1
        return pd.DataFrame([{"Diagnostic": k, "Count": v} for k, v in counts.items()])
    if htf is None or htf.empty:
        counts["Skipped: higher-timeframe data unavailable"] = 1
        return pd.DataFrame([{"Diagnostic": k, "Count": v} for k, v in counts.items()])
    d = add_adaptive_levels(add_indicators(df), timeframe)
    ht = add_adaptive_levels(add_indicators(htf), htf_timeframe)
    balance_proxy = float(STARTING_BALANCE)
    day_start_proxy = balance_proxy
    current_day = None
    cooldown_until = -1
    # Use same signal eligibility window as the simplified backtest. Since this is a diagnostic
    # pass, trade P&L is not recomputed; cooldown is approximated using accepted signal bars.
    for i in range(220, len(d)-1):
        row = d.iloc[i]
        ts = d.index[i]
        day = pd.Timestamp(ts).date()
        if current_day != day:
            current_day, day_start_proxy = day, balance_proxy
        if (day_start_proxy - balance_proxy) / max(day_start_proxy, 1) >= max_daily_loss_pct/100:
            counts["Skipped: daily loss limit reached"] += 1
            continue
        if i < cooldown_until:
            counts["Skipped: cooldown after prior trade"] += 1
            continue
        hslice = ht.loc[ht.index <= ts]
        if len(hslice) < 205:
            counts["Skipped: not enough higher-timeframe history"] += 1
            continue
        bias, _ = htf_bias(hslice.tail(500))
        required = ["ATR", "BB_Mid", "BB_Lower", "BB_Upper", "ADX", "RSI", "MFI"]
        if not all(pd.notna(row.get(k, np.nan)) for k in required):
            counts["Skipped: indicators not ready"] += 1
            continue
        counts["Bars evaluated for signals"] += 1
        long_hard_ok = (bias == "Bullish" and pd.notna(row["ADX"]) and row["ADX"] < 25 and row["Close"] < row["BB_Lower"])
        short_hard_ok = (bias == "Bearish" and pd.notna(row["ADX"]) and row["ADX"] < 25 and row["Close"] > row["BB_Upper"])
        if bias not in ("Bullish", "Bearish"):
            counts["Rejected: HTF bias not aligned"] += 1
        elif bias == "Bullish" and not long_hard_ok:
            if not (pd.notna(row["ADX"]) and row["ADX"] < 25):
                counts["Rejected: ADX hard filter (must be <25)"] += 1
            if not (row["Close"] < row["BB_Lower"]):
                counts["Rejected: Bollinger close hard filter"] += 1
        elif bias == "Bearish" and not short_hard_ok:
            if not (pd.notna(row["ADX"]) and row["ADX"] < 25):
                counts["Rejected: ADX hard filter (must be <25)"] += 1
            if not (row["Close"] > row["BB_Upper"]):
                counts["Rejected: Bollinger close hard filter"] += 1
        side, score, _ = score_signal(row, bias, threshold)
        if side in ("BUY", "SELL"):
            counts["Signals passing all filters"] += 1
            cooldown_until = i + 97
        elif score < threshold:
            counts["Rejected: Signal confirmation not met"] += 1
    return pd.DataFrame([{"Diagnostic": k, "Count": v} for k, v in counts.items()])


def run_backtest(df, htf, timeframe, htf_timeframe, threshold, risk_pct, rr2, starting_balance, max_daily_loss_pct, cost_bps=2.0):
    # Simple candle-close signal backtest; next-bar open entry avoids same-close lookahead.
    # Bar-level stop/target ordering is conservative: if stop and target both touched, stop wins.
    if len(df) < 250:
        return pd.DataFrame(), starting_balance, "Need at least 250 bars for a useful backtest."
    d = add_adaptive_levels(add_indicators(df), timeframe)
    ht = add_adaptive_levels(add_indicators(htf), htf_timeframe) if not htf.empty else pd.DataFrame()
    if ht.empty:
        return pd.DataFrame(), starting_balance, "Higher-timeframe data unavailable."
    trades = []
    balance = starting_balance
    day_start_balance = balance
    current_day = None
    cooldown_until = -1
    for i in range(220, len(d)-1):
        row = d.iloc[i]
        ts = d.index[i]
        day = pd.Timestamp(ts).date()
        if current_day != day:
            current_day, day_start_balance = day, balance
        if (day_start_balance - balance) / max(day_start_balance, 1) >= max_daily_loss_pct/100:
            continue
        if i < cooldown_until:
            continue
        hslice = ht.loc[ht.index <= ts]
        if len(hslice) < 205:
            continue
        bias, _ = htf_bias(hslice.tail(500))
        if not all(pd.notna(row.get(k, np.nan)) for k in ["ATR","BB_Mid","BB_Lower","BB_Upper","ADX","RSI","MFI"]):
            continue
        side, score, _ = score_signal(row, bias, threshold)
        if side not in ("BUY","SELL"):
            continue
        levels = make_trade_levels(row, side, risk_pct, rr2)
        if not levels:
            continue
        nxt = d.iloc[i+1]
        entry = float(nxt["Open"])
        # Re-anchor levels to next-bar entry, preserving stop distance.
        if side == "BUY":
            stop = min(levels["stop"], entry - 0.5*levels["risk_per_unit"])
            risk_unit = entry-stop
            tp1 = entry + 0.9*risk_unit
            tp2 = entry + rr2*risk_unit
        else:
            stop = max(levels["stop"], entry + 0.5*levels["risk_per_unit"])
            risk_unit = stop-entry
            tp1 = entry - 0.9*risk_unit
            tp2 = entry - rr2*risk_unit
        risk_cash = balance * risk_pct/100
        qty = risk_cash / max(risk_unit, 1e-9)
        outcome = None
        exit_price = None
        exit_time = None
        partial_pnl = 0.0
        tp1_recorded = False
        trail_extreme = None
        for j in range(i+1, min(i+97, len(d))):
            bar = d.iloc[j]
            if side == "BUY":
                stop_hit = bar["Low"] <= stop
                tp1_hit = bar["High"] >= tp1
                tp2_hit = bar["High"] >= tp2
            else:
                stop_hit = bar["High"] >= stop
                tp1_hit = bar["Low"] <= tp1
                tp2_hit = bar["Low"] <= tp2
            if stop_hit:
                # Conservative handling: stop wins if both are touched in one bar.
                exit_price, exit_time, outcome = stop, d.index[j], "SL"
                remaining_fraction = 0.4 if tp1_recorded else 1.0
                pnl = partial_pnl + ((exit_price-entry) if side=="BUY" else (entry-exit_price))*qty*remaining_fraction
                break
            if tp1_hit and not tp1_recorded:
                partial_pnl = ((tp1-entry) if side=="BUY" else (entry-tp1))*qty*0.6
                stop = cost_adjusted_breakeven(entry, side, cost_bps)
                tp1_recorded = True
                trail_extreme = float(bar["High"] if side == "BUY" else bar["Low"])
            elif tp1_recorded:
                atr_value = bar.get("ATR", np.nan)
                if pd.notna(atr_value) and float(atr_value) > 0:
                    current_extreme = float(bar["High"] if side == "BUY" else bar["Low"])
                    trail_extreme = max(trail_extreme, current_extreme) if side == "BUY" else min(trail_extreme, current_extreme)
                    candidate = trail_extreme - 1.5 * float(atr_value) if side == "BUY" else trail_extreme + 1.5 * float(atr_value)
                    floor = cost_adjusted_breakeven(entry, side, cost_bps)
                    stop = max(stop, floor, candidate) if side == "BUY" else min(stop, floor, candidate)
            if tp2_hit:
                exit_price, exit_time, outcome = tp2, d.index[j], "TP2"
                pnl = partial_pnl + ((exit_price-entry) if side=="BUY" else (entry-exit_price))*qty*0.4
                break
        else:
            j = min(i+96, len(d)-1)
            exit_price, exit_time, outcome = float(d.iloc[j]["Close"]), d.index[j], "TIME"
            pnl = ((exit_price-entry) if side=="BUY" else (entry-exit_price))*qty
        # Simple round-trip cost estimate in basis points of entry price; adjust in settings for your venue.
        cost = abs(entry) * (cost_bps / 10000.0) * qty
        pnl -= cost
        balance += pnl
        trades.append({"Entry time": d.index[i+1], "Exit time": exit_time, "Side": side, "Score": score,
                       "Entry": entry, "Stop": stop, "TP1": tp1, "TP2": tp2, "Exit": exit_price,
                       "Outcome": outcome, "PnL (price units)": pnl, "Balance": balance})
        cooldown_until = j + 1
    t = pd.DataFrame(trades)
    return t, balance, None

# -----------------------------
# UI
# -----------------------------
st.title("🧠 AI Brain — Paper Trading Lab")
st.caption("Multi-strategy paper-trading research engine with trend, mean-reversion, breakout, momentum, range, volatility-expansion, price-action and liquidity-sweep proxy strategies. Paper trading only; no broker, wallet, or live-order connection.")
with st.sidebar:
    st.header("Bot settings")
    # Keep changes staged until the user explicitly confirms them.
    with st.form("bot_settings_form"):
        starting_balance_setting = st.slider("Starting virtual balance (USD)", 10, 1000, int(st.session_state.get("starting_balance_applied", STARTING_BALANCE)), 10, key="settings_starting_balance")
        market_name = st.selectbox("Market", list(MARKETS.keys()), key="settings_market")
        cfg = MARKETS[market_name]
        allowed_tfs = cfg["allowed_tfs"]
        default_tf = cfg["signal_tf"]
        current_tf = st.session_state.get("settings_signal_tf")
        tf_index = allowed_tfs.index(current_tf) if current_tf in allowed_tfs else allowed_tfs.index(default_tf)
        signal_tf = st.selectbox("Signal timeframe", allowed_tfs, index=tf_index, key="settings_signal_tf_widget")
        htf_options = ["15m", "30m", "1h"]
        htf_default_index = htf_options.index("1h")
        htf_tf = st.selectbox("Higher-timeframe filter", htf_options, index=htf_default_index, key="settings_htf_tf")
        score_threshold = st.slider("Minimum Signal strength", 50, 95, 70, 1, key="settings_score_threshold")
        risk_pct = st.slider("Risk per trade (%)", 0.25, 1.0, 0.5, 0.25, key="settings_risk_pct")
        max_daily_loss = st.slider("Daily loss stop (%)", 1.0, 3.0, 2.0, 0.5, key="settings_max_daily_loss")
        max_trades_per_day = st.slider("Maximum paper trades per day", 1, 30, 20, 1, key="settings_max_trades_per_day")
        rr2 = st.slider("TP2 risk/reward", 1.5, 2.5, 2.0, 0.1, key="settings_rr2")
        confirm_settings = st.form_submit_button("✅ Confirm settings", type="primary", use_container_width=True)
    if confirm_settings:
        previous_starting_balance = float(st.session_state.get("starting_balance_applied", STARTING_BALANCE))
        st.session_state.starting_balance_applied = float(starting_balance_setting)
        if previous_starting_balance != float(starting_balance_setting) or "balance" not in st.session_state:
            st.session_state.balance = float(starting_balance_setting)
            st.session_state.daily_start = float(starting_balance_setting)
            st.session_state.auto_daily_start_balance = float(starting_balance_setting)
            # Open positions are tied to the old account size; close/reset them on capital change.
            st.session_state.auto_open_positions = []
        st.session_state.settings_market_applied = market_name
        st.session_state.settings_signal_tf = signal_tf
        st.session_state.settings_htf_tf_applied = htf_tf
        st.session_state.settings_score_threshold_applied = None
        st.session_state.settings_risk_pct_applied = risk_pct
        st.session_state.settings_max_daily_loss_applied = max_daily_loss
        st.session_state.settings_max_trades_per_day_applied = max_trades_per_day
        st.session_state.settings_rr2_applied = rr2
        st.success("Settings confirmed for this session.")
    st.caption("The multi-strategy engine evaluates trend, mean reversion, breakout, momentum, range, volatility expansion, price action and optional SMC confluence. Auto entries require at least 70% directional vote alignment. The 1H chart is a trend filter only; entries remain on 5m/15m/30m. Paper trading only.")
    st.subheader("🧩 Optional Smart Money Concepts")
    st.checkbox("Use SMC confluence in decisions", value=st.session_state.get("smc_enabled", True), key="smc_enabled", help="SMC is optional; other strategy families remain eligible when there is no SMC setup.")
    st.caption("SMC detects heuristic Order Blocks, BOS, CHOCH, Fair Value Gaps and liquidity sweeps from candle OHLC data.")

    st.divider()
    st.subheader("📰 Forex Factory Red-Folder News Protection")
    st.toggle("News Protection ON/OFF", value=st.session_state.get("news_protection_enabled", True),
              key="news_protection_enabled",
              help="When ON, high-impact calendar windows block new paper entries. Existing positions are still managed.")
    st.slider("Block before news (minutes)", 0, 180, int(st.session_state.get("news_pre_minutes", 30)), 5, key="news_pre_minutes")
    st.slider("Block after news (minutes)", 0, 180, int(st.session_state.get("news_post_minutes", 30)), 5, key="news_post_minutes")
    st.caption("Source: Forex Factory's published weekly calendar feed only. The bot automatically refreshes every 15 minutes and accepts only High-impact red-folder events. Medium, low, holiday and all other calendar sources are excluded. If the feed is unavailable and no successful cache exists, new entries are blocked while protection is ON.")
    _cal_df, _cal_status = load_news_calendar()
    if st.session_state.get("news_protection_enabled", True):
        if not _cal_df.empty:
            st.success(f"Protection enabled · {_cal_df.shape[0]} Forex Factory red-folder events loaded. {_cal_status}")
            _show_cols = [c for c in ["datetime_utc", "currency", "impact", "event", "status", "confidence"] if c in _cal_df.columns]
            st.dataframe(_cal_df[_show_cols].head(20), use_container_width=True, hide_index=True)
        else:
            st.error(f"Protection is enabled but no valid calendar is available. New entries will be blocked. {_cal_status}")
    else:
        st.warning("Protection disabled. The bot will not block entries around scheduled news.")

    st.divider()
    st.subheader("⏰ Market Session Protection")
    st.toggle(
        "1 HOUR BEFORE MARKET CLOSE — ON/OFF",
        value=st.session_state.get("preclose_protection_enabled", True),
        key="preclose_protection_enabled",
        help="ON strictly blocks new entries during the 60 minutes before the configured market close/FX rollover.",
    )
    st.toggle(
        "1 HOUR AFTER MARKET OPENS — ON/OFF",
        value=st.session_state.get("postopen_protection_enabled", True),
        key="postopen_protection_enabled",
        help="ON strictly blocks new entries during the first 60 minutes after the configured market reopening/FX rollover.",
    )
    st.info("Protection window is fixed at 60 minutes before close and 60 minutes after reopening. The window cannot be reduced by a stale setting.")
    st.toggle(
        "SATURDAY & SUNDAY — NO TRADING — ON/OFF",
        value=st.session_state.get("weekend_no_trade_enabled", True),
        key="weekend_no_trade_enabled",
        help="ON blocks all new entries on Saturdays and Sundays for every symbol, including crypto. Existing positions continue to be managed.",
    )
    st.caption("Session hours are approximate and must be verified against your broker. For each symbol, new entries are blocked for 60 minutes before its configured close/rollover and 60 minutes after reopening. Crypto has no daily close/open; the Saturday/Sunday switch applies to crypto too. These switches block new entries only; they do not close open positions.")

if "starting_balance_applied" not in st.session_state:
    st.session_state.starting_balance_applied = float(STARTING_BALANCE)
if "balance" not in st.session_state:
    st.session_state.balance = float(st.session_state.starting_balance_applied)
if "trades" not in st.session_state:
    st.session_state.trades = load_trade_log()
if "daily_start" not in st.session_state:
    st.session_state.daily_start = float(st.session_state.starting_balance_applied)
if "daily_date" not in st.session_state:
    st.session_state.daily_date = datetime.now().date()
if "halted" not in st.session_state:
    st.session_state.halted = False
if "trade_day" not in st.session_state:
    st.session_state.trade_day = datetime.now().date().isoformat()
if "trades_today" not in st.session_state:
    st.session_state.trades_today = 0
if "auto_open_positions" not in st.session_state:
    st.session_state.auto_open_positions = []
if "auto_closed_trades" not in st.session_state:
    st.session_state.auto_closed_trades = []
if "auto_last_processed" not in st.session_state:
    st.session_state.auto_last_processed = {}
if "auto_daily_start_balance" not in st.session_state:
    st.session_state.auto_daily_start_balance = st.session_state.balance
if "auto_bot_status" not in st.session_state:
    st.session_state.auto_bot_status = "RUNNING"
if "symbol_trade_enabled" not in st.session_state:
    st.session_state.symbol_trade_enabled = {name: True for name in MARKETS}
else:
    for _market_name in MARKETS:
        st.session_state.symbol_trade_enabled.setdefault(_market_name, True)

# Streamlit reruns the script on this timer while the browser session is connected.
# Automatic paper scans are gated below so ordinary UI reruns do not trigger new entries.
auto_refresh_count = st_autorefresh(interval=30_000, limit=None, key="ai_brain_auto_refresh_30s")
if "auto_last_refresh_count" not in st.session_state:
    st.session_state.auto_last_refresh_count = auto_refresh_count

today_key = datetime.now().date().isoformat()
if st.session_state.trade_day != today_key:
    st.session_state.trade_day = today_key
    st.session_state.trades_today = 0
    st.session_state.daily_start = st.session_state.balance
    st.session_state.halted = False

col1, col2, col3, col4 = st.columns(4)
col1.metric("Virtual balance (USD)", f"${st.session_state.balance:,.2f}")
col2.metric("Paper trades today", f"{st.session_state.trades_today}/{max_trades_per_day}")
col3.metric("Daily loss stop", f"{max_daily_loss:.1f}%")
col4.metric("Risk per setup", f"{risk_pct:.2f}%")

if st.button("🔄 Refresh all market data", type="primary"):
    st.cache_data.clear()

st.subheader("🕒 Market open/close & global trading sessions")
now_utc = datetime.now(timezone.utc)
st.caption("Session status refreshes with the dashboard. Times are approximate session hours; broker holidays, daylight-saving changes, and product-specific hours can differ.")
market_rows = []
for _name, _cfg in MARKETS.items():
    _status, _note = market_schedule_status(_name, now_utc)
    market_rows.append({"Market": _name, "Market status": _status, "Trading session / hours": _note, "Auto-trade": "ON" if st.session_state.symbol_trade_enabled.get(_name, True) else "OFF"})
st.dataframe(pd.DataFrame(market_rows), use_container_width=True, hide_index=True)
st.markdown("**Global forex sessions**")
st.dataframe(current_global_sessions(now_utc), use_container_width=True, hide_index=True)

st.subheader("🤖 Analysis Agents")
st.caption("These are local rule-based analytical components, not connected external LLM services. Their confidence scores are heuristic agreement scores, not calibrated win probabilities.")
agent_col1, agent_col2 = st.columns(2)
with agent_col1:
    st.markdown("**Chart Analysis AI Agent**")
    st.write("Evaluates EMA trend, 1H bias, RSI and directional strength.")
    st.success("Integrated into the strategy vote when confidence is at least 70%.")
with agent_col2:
    st.markdown("**Harmonic Pattern AI Agent**")
    st.write("Checks confirmed swing pivots for heuristic Gartley, Bat, Butterfly and Crab structures.")
    st.info("Only contributes a vote when a candidate pattern is detected with at least 70% heuristic confidence.")
st.caption("Hermes AI Agent is not connected in this package. It requires a separate Hermes runtime/API integration.")
st.subheader("🌐 All-market live signal dashboard")
news_on = bool(st.session_state.get("news_protection_enabled", True))
if news_on:
    st.info("News Protection is ON. New paper entries are blocked unless a valid calendar is loaded and the market is outside its high-impact event window.")
else:
    st.warning("News Protection is OFF. Scheduled-news entry blocking is disabled.")
st.caption("The app script reruns about every 30 seconds while this page stays open and connected. Yahoo Finance is an indicative data source, not guaranteed real-time broker data.")
with st.spinner("Analyzing all configured markets..."):
    overview = scan_all_market_signals()
if not overview.empty:
    overview_display = overview.copy()
    overview_display.insert(0, "Auto-trade", ["ON" if st.session_state.symbol_trade_enabled.get(n, True) else "OFF" for n in overview_display["Market"]])
    st.dataframe(overview_display, use_container_width=True, hide_index=True)
    usable = int((overview["Data status"] == "OK").sum())
    st.caption(f"Markets with usable data: {usable}/{len(overview)}. Provider data may be delayed or unavailable for some symbols.")
else:
    st.warning("No market scan results are available yet.")

with st.expander("🎛️ Per-symbol automatic trading switches", expanded=True):
    st.caption("Turn a symbol OFF to prevent new paper entries on that market. Existing open positions will continue to be managed for stop-loss/targets.")
    toggle_cols = st.columns(3)
    for idx, sym_name in enumerate(MARKETS):
        with toggle_cols[idx % 3]:
            st.toggle(f"{sym_name}", value=st.session_state.symbol_trade_enabled.get(sym_name, True), key=f"trade_toggle_{idx}", on_change=None)
            # Copy widget value to the shared settings map used by the scanner.
            st.session_state.symbol_trade_enabled[sym_name] = bool(st.session_state.get(f"trade_toggle_{idx}", True))

# Calendar-style daily results report from saved paper trade history.
st.divider()
st.subheader("🗓️ Daily trading calendar & performance report")
trades_df = pd.DataFrame(st.session_state.trades)
if not trades_df.empty:
    if "Time" in trades_df.columns:
        trades_df["_opened_dt"] = pd.to_datetime(trades_df.get("Opened", pd.Series(index=trades_df.index, dtype=object)).fillna(trades_df["Time"]), errors="coerce")
    else:
        trades_df["_opened_dt"] = pd.to_datetime(trades_df.get("Opened", pd.Series(index=trades_df.index, dtype=object)), errors="coerce")
    trades_df["_closed_dt"] = pd.to_datetime(trades_df.get("Time", pd.Series(index=trades_df.index, dtype=object)), errors="coerce")
    pnl_col = "PnL (estimated price-units)" if "PnL (estimated price-units)" in trades_df.columns else ("PnL (price units)" if "PnL (price units)" in trades_df.columns else None)
    if pnl_col:
        trades_df["_pnl"] = pd.to_numeric(trades_df[pnl_col], errors="coerce")
    else:
        trades_df["_pnl"] = np.nan
    # Recover estimated P&L from older manual-close status text where no numeric column was saved.
    if "Status" in trades_df.columns:
        _status_pnl = trades_df["Status"].astype(str).str.extract(r"est\. PnL\s+(-?\d+(?:\.\d+)?)", expand=False)
        trades_df["_pnl"] = trades_df["_pnl"].fillna(pd.to_numeric(_status_pnl, errors="coerce"))
    closed_mask = trades_df.get("Status", pd.Series("", index=trades_df.index)).astype(str).str.contains("CLOSED|CLOSED @", case=False, regex=True) | trades_df["_pnl"].notna()
    completed = trades_df[closed_mask].copy()
    if not completed.empty:
        completed["_day"] = completed["_closed_dt"].dt.date.astype(str)
        completed["_win"] = completed["_pnl"] > 0
        completed["_loss"] = completed["_pnl"] < 0
        daily = completed.groupby("_day", dropna=True).agg(Trades=("_pnl", "count"), Wins=("_win", "sum"), Losses=("_loss", "sum"), **{"Net P&L (price-units)": ("_pnl", "sum")}).reset_index().rename(columns={"_day": "Date"})
        daily["Win rate"] = np.where(daily["Trades"] > 0, daily["Wins"] / daily["Trades"] * 100, 0).round(1)
        today_report = daily[daily["Date"] == date.today().isoformat()]
        today_trades = int(today_report["Trades"].sum()) if not today_report.empty else 0
        today_wins = int(today_report["Wins"].sum()) if not today_report.empty else 0
        today_losses = int(today_report["Losses"].sum()) if not today_report.empty else 0
        today_pnl = float(today_report["Net P&L (price-units)"].sum()) if not today_report.empty else 0.0
        today_wr = (today_wins / today_trades * 100) if today_trades else 0.0
        d1,d2,d3,d4,d5 = st.columns(5)
        d1.metric("Today's closed trades", today_trades)
        d2.metric("Today's wins", today_wins)
        d3.metric("Today's losses", today_losses)
        d4.metric("Today's win rate", f"{today_wr:.1f}%" if today_trades else "—")
        d5.metric("Today's net P&L", f"{today_pnl:,.3f}")
        st.caption("Trade P&L is an estimated price-unit proxy, not a broker-calibrated USD return. Contract sizes, conversions, spreads, commissions, swaps and funding are not fully modeled; do not interpret these figures as real-world dollar profit.")
        report_days = st.slider("Calendar report: past days", 7, 90, 30, 1, key="daily_report_days")
        cutoff_day = (date.today() - timedelta(days=report_days - 1)).isoformat()
        report = daily[daily["Date"] >= cutoff_day].sort_values("Date", ascending=False)
        st.dataframe(report, use_container_width=True, hide_index=True, column_config={"Win rate": st.column_config.NumberColumn(format="%.1f%%"), "Net P&L (price-units)": st.column_config.NumberColumn(format="%.3f")})
        st.download_button("Download daily performance report CSV", report.to_csv(index=False).encode("utf-8"), "daily_performance_report.csv", "text/csv")
        available_days = sorted(completed["_day"].dropna().unique().tolist(), reverse=True)
        if available_days:
            selected_report_day = st.selectbox("Inspect completed trades for date", available_days, key="inspect_daily_trade_date")
            day_trades = completed[completed["_day"] == selected_report_day].copy()
            keep_cols = [c for c in ["Time", "Opened", "Market", "Symbol", "TF", "Side", "Entry", "Exit", "Stop", "TP1", "TP2", "Exit reason", "Status", "_pnl"] if c in day_trades.columns]
            day_trades = day_trades[keep_cols].rename(columns={"_pnl": "Net P&L (price-units)"})
            st.markdown(f"**Trade-by-trade result for {selected_report_day}**")
            st.dataframe(day_trades, use_container_width=True, hide_index=True)

    else:
        st.info("No completed paper trades yet. Daily wins, losses, P&L and win rate will appear after positions close. Open positions are not counted as wins or losses.")
else:
    st.info("No paper-trade history yet. The daily report will populate as trades are recorded and completed.")

st.divider()
st.subheader(f"📊 Detailed chart and indicators: {market_name}")
with st.expander("🧠 Multi-strategy engine: strategy votes & market regime", expanded=True):
    st.write("The engine evaluates multiple rule-based strategy families, including optional SMC confluence, detects a broad market regime, then selects up to three strategies per symbol/timeframe. It uses closed-trade outcomes recorded for that exact market and timeframe when enough examples exist; otherwise it uses regime-based priors during warm-up. At least two selected strategies must agree. This is adaptive research logic, not literally every strategy worldwide. Live news trading is not active without an economic-calendar/news feed.")
    if not overview.empty:
        cols = [c for c in ["Market", "Signal", "Market regime", "Strategy BUY votes", "Strategy SELL votes", "Selected strategies", "Selection mode", "Data status"] if c in overview.columns]
        st.dataframe(overview[cols], use_container_width=True, hide_index=True)

st.caption(f"Per-symbol auto-trading is {'ON' if st.session_state.symbol_trade_enabled.get(market_name, True) else 'OFF'} for {market_name}. Change it in the per-symbol switches above. The Market dropdown only selects the detailed chart.")

with st.spinner("Downloading selected-market candles and evaluating the AI Brain..."):
    raw = download_data(cfg["symbol"], signal_tf)
    htf = download_data(cfg["symbol"], htf_tf)
    data = add_smc_features(add_adaptive_levels(add_indicators(raw), signal_tf)) if not raw.empty else pd.DataFrame()
    if not data.empty:
        data["PrevClose"] = data["Close"].shift(1)
        data["PrevSupport"] = data["Low"].rolling(20).min().shift(2)
        data["PrevResistance"] = data["High"].rolling(20).max().shift(2)
    htf = add_adaptive_levels(add_indicators(htf), htf_tf) if not htf.empty else htf
    bias, htf_last = htf_bias(htf) if not htf.empty else ("Unknown", None)

if data.empty or len(data) < 60:
    st.error("Market data is unavailable or too short from Yahoo Finance right now. Try Refresh, another symbol, or another timeframe.")
    st.info("Yahoo Finance intraday data can be delayed, incomplete, or unavailable. Gold uses GC=F futures as a proxy, XAGUSD Silver uses SI=F futures as a proxy, and Crude Oil uses CL=F futures as a proxy; these may differ from broker CFD/spot quotes.")
    st.stop()

last = data.iloc[-1]
learning = learning_profile(st.session_state.trades)
side, score, strategy_votes, regime, buy_votes, sell_votes = evaluate_multi_strategy(last, bias)
checks = strategy_votes
side, ml_probability, ml_note = apply_learning_gate(side, last, bias, score, learning)
daily_loss_reached = ((st.session_state.daily_start - st.session_state.balance) / max(st.session_state.daily_start, 1) * 100) >= max_daily_loss
if st.session_state.trades_today >= max_trades_per_day:
    side = "WAIT"
    checks = {"Daily trade limit reached": True}
elif daily_loss_reached:
    st.session_state.halted = True
    side = "WAIT"
    checks = {"Daily loss limit reached": True}
st.subheader(f"{market_name} — {signal_tf} signal")
m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("Last candle close", f"{float(last['Close']):,.5f}")
m2.metric("HTF bias", bias)
m3.metric("AI Signal strength", f"{score}/100")
m4.metric("RSI / MFI", f"{last['RSI']:.1f} / {last['MFI']:.1f}" if pd.notna(last["MFI"]) else f"{last['RSI']:.1f} / n/a")
m5.metric("ADX / Rel. volume", f"{last['ADX']:.1f} / {last['RelVol']:.2f}x" if pd.notna(last["RelVol"]) else f"{last['ADX']:.1f} / n/a")

with st.expander("🧩 SMC detector — Order Blocks, BOS, CHOCH, FVG & Liquidity Sweeps", expanded=False):
    smc_now = smc_signal(last)
    st.write(f"Optional SMC candidate: **{smc_now}** · SMC voting is {'ON' if st.session_state.get('smc_enabled', True) else 'OFF'}.")
    smc_cols = ["SMC_BOS", "SMC_CHOCH", "SMC_FVG", "SMC_OB", "SMC_OB_Retest", "SMC_LiquiditySweep", "SMC_Active_OB_Bottom", "SMC_Active_OB_Top"]
    st.dataframe(data[smc_cols].tail(12), use_container_width=True)
    st.caption("SMC zones are rule-based OHLC heuristics, not proof of institutional orders. BOS/CHOCH and zones can be noisy; validate them on unseen data before relying on them.")

with st.expander("🧠 Self-learning engine — trade outcomes + machine learning", expanded=True):
    l1, l2, l3 = st.columns(3)
    l1.metric("Completed learning trades", learning["closed_samples"])
    l2.metric("Recent win rate", f"{learning['recent_win_rate']:.1%}" if learning["recent_win_rate"] is not None else "Not enough data")
    l3.metric("ML model", "TRAINED" if learning["model_ready"] else "COLLECTING DATA")
    st.write(learning["model_note"])
    st.write(f"Adaptive rule confirmations: {learning['min_confirmations']} of 10 checks.")
    score_cols = st.columns(2)
    score_cols[0].metric("Chronological holdout AUC", f"{learning['model_auc']:.3f}" if learning.get("model_auc") is not None and pd.notna(learning.get("model_auc")) else "Not qualified")
    score_cols[1].metric("Holdout Brier score", f"{learning['model_brier']:.3f}" if learning.get("model_brier") is not None and pd.notna(learning.get("model_brier")) else "Not available")
    diag = learning.get("learning_diagnostics")
    if diag and diag.get("groups"):
        st.markdown("**Repeated-loss pattern review**")
        flagged = [g for g in diag["groups"] if g.get("pattern_flag")]
        if flagged:
            st.dataframe(pd.DataFrame(flagged)[["dimension", "group", "trades", "win_rate", "expectancy", "action"]], use_container_width=True, hide_index=True)
        else:
            st.info("No repeated-loss pattern passed the minimum sample review yet.")
        st.caption(diag.get("warning", "Pattern analysis is advisory only."))
    if ml_probability is not None:
        st.write(f"Current setup estimated win probability: {ml_probability:.1%}.")
    st.caption("Adaptive learning uses completed paper trades only. A model needs at least 300 trades, at least 50 wins and 50 losses, and chronological holdout AUC ≥ 0.53. Candidate models that fail validation are not promoted. Probabilities are estimates, not guarantees.")

with st.expander("🔬 Controlled strategy discovery — research candidates", expanded=False):
    if discover_strategies is None:
        st.info("Strategy-discovery module is unavailable; active strategies are unchanged.")
    else:
        try:
            candidate_report = discover_strategies(data, cost_bps=2.0, min_validation_trades=8)
            if candidate_report is None or candidate_report.empty:
                st.info("Not enough clean market history or validation trades to rank candidate strategies yet.")
            else:
                st.dataframe(candidate_report, use_container_width=True, hide_index=True)
                st.caption("Discovery evaluates candidates on a chronological validation segment. It is research-only: a candidate is not automatically activated from one report, and all estimates are approximate after-cost simulations.")
        except Exception as exc:
            st.info(f"Candidate research skipped safely: {str(exc)[:120]}")

if st.session_state.halted:
    st.error("Paper bot is halted by the daily loss control. It will reset on the next calendar day.")
elif st.session_state.trades_today >= max_trades_per_day:
    st.error(f"Daily trade limit reached ({max_trades_per_day}). No more paper trades will be allowed today.")
elif side in ("BUY","SELL"):
    st.success(f"Potential {side} setup — paper signal. {ml_note} This is not a guaranteed trade.")
else:
    st.warning(f"WAIT — no setup passed all filters. Best-side Signal strength: {score}/100. {ml_note}")

st.subheader("Automatic paper trading")
st.caption(f"Paper trading only. The app script reruns about every 30 seconds while this page stays open and connected. Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Streamlit reruns the app script rather than updating only individual widgets. Cloud sleep, browser disconnection, or provider delays can interrupt scans.")

if "auto_bot_status" not in st.session_state:
    st.session_state.auto_bot_status = "RUNNING"
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 2])
with ctrl1:
    if st.button("⏸️ Pause", use_container_width=True, disabled=st.session_state.auto_bot_status != "RUNNING"):
        st.session_state.auto_bot_status = "PAUSED"
        st.session_state.last_auto_message = "Bot paused: no new paper entries; existing positions will continue to be managed."
        st.rerun()
with ctrl2:
    if st.button("▶️ Resume / Start", use_container_width=True, disabled=st.session_state.auto_bot_status == "RUNNING"):
        st.session_state.auto_bot_status = "RUNNING"
        st.session_state.last_auto_message = "Bot started/resumed. Automatic paper scanning is enabled."
        st.rerun()
with ctrl3:
    if st.button("⏹️ Stop", use_container_width=True, disabled=st.session_state.auto_bot_status == "STOPPED"):
        st.session_state.auto_bot_status = "STOPPED"
        st.session_state.last_auto_message = "Bot stopped: automatic scans and new entries are disabled. Open paper positions and trade history are preserved; positions are not automatically closed."
        st.rerun()
with ctrl4:
    if st.session_state.auto_bot_status == "RUNNING":
        st.success("Bot status: RUNNING")
    elif st.session_state.auto_bot_status == "PAUSED":
        st.warning("Bot status: PAUSED — managing existing positions only")
    else:
        st.error("Bot status: STOPPED — automatic scanning disabled")

def run_all_market_scan(allow_new_entries=True):
    all_events = []
    messages = []
    # The Market dropdown is for detailed diagnostics only; automatic paper entries
    # are evaluated independently for every configured market.
    for scan_name, scan_cfg in MARKETS.items():
        scan_tf = scan_cfg["signal_tf"]
        try:
            scan_raw = download_data(scan_cfg["symbol"], scan_tf)
            scan_htf_raw = download_data(scan_cfg["symbol"], scan_cfg["htf"])
            if scan_raw.empty or len(scan_raw) < 60:
                all_events.append(f"{scan_name}: skipped; insufficient market data.")
                continue
            scan_data = add_smc_features(add_adaptive_levels(add_indicators(scan_raw), scan_tf))
            scan_data["PrevClose"] = scan_data["Close"].shift(1)
            scan_data["PrevSupport"] = scan_data["Low"].rolling(20).min().shift(2)
            scan_data["PrevResistance"] = scan_data["High"].rolling(20).max().shift(2)
            scan_htf = add_adaptive_levels(add_indicators(scan_htf_raw), scan_cfg["htf"]) if not scan_htf_raw.empty else pd.DataFrame()
            scan_bias, _ = htf_bias(scan_htf) if not scan_htf.empty else ("Unknown", None)
            (st.session_state.balance, st.session_state.trades_today,
             st.session_state.auto_daily_start_balance, st.session_state.auto_open_positions,
             scan_events, scan_message) = simulate_auto_paper_scan(
                scan_data, scan_name, scan_cfg["symbol"], scan_tf, scan_bias, None,
                risk_pct, rr2, st.session_state.balance, st.session_state.trades_today,
                max_trades_per_day, st.session_state.auto_daily_start_balance,
                max_daily_loss, st.session_state.auto_open_positions,
                allow_new_entries=allow_new_entries
            )
            all_events.extend(scan_events)
            if scan_message:
                messages.append(f"{scan_name}: {scan_message}")
        except Exception as exc:
            all_events.append(f"{scan_name}: scan error: {str(exc)[:100]}")
    st.session_state.last_auto_events = all_events
    st.session_state.last_auto_message = "All-market scan completed. " + (" ".join(messages) if messages else "No new entries were opened; see market signals and events.")

# Run once for each 60-second timer tick while the scanner is armed.
if auto_refresh_count > st.session_state.auto_last_refresh_count:
    st.session_state.auto_last_refresh_count = auto_refresh_count
    if st.session_state.auto_bot_status == "RUNNING":
        run_all_market_scan(allow_new_entries=True)
        st.rerun()
    elif st.session_state.auto_bot_status == "PAUSED":
        run_all_market_scan(allow_new_entries=False)
        st.rerun()
# STOPPED disables automatic scans; the dashboard may still rerun for display.

if st.button("🔎 Scan & manage positions now", type="primary",
             disabled=st.session_state.auto_bot_status == "STOPPED"):
    run_all_market_scan(allow_new_entries=st.session_state.auto_bot_status == "RUNNING")
    st.rerun()

if st.session_state.get("last_auto_message"):
    st.info(st.session_state.last_auto_message)
for ev in st.session_state.get("last_auto_events", []):
    st.write("• " + ev)

st.markdown(f"**Open automatic paper positions — all markets ({len([p for p in st.session_state.auto_open_positions if p.get('Status', 'OPEN') == 'OPEN'])}/{MAX_GLOBAL_OPEN_POSITIONS} global limit)**")
all_open_positions = [
    p for p in st.session_state.auto_open_positions
    if p.get("Status", "OPEN") == "OPEN"
]
if all_open_positions:
    # Present one clear Stop-loss column; do not expose the old InitialStop duplicate.
    display_rows = []
    for p in all_open_positions:
        display_rows.append({
            "Opened": p.get("Opened"),
            "Market": p.get("Market"),
            "Symbol": p.get("Symbol"),
            "Timeframe": p.get("Timeframe"),
            "Side": p.get("Side"),
            "Confidence score (%)": p.get("ConfidenceScore", p.get("Score")),
            "Entry": p.get("Entry"),
            "Stop-loss": p.get("Stop"),
            "TP1": p.get("TP1"),
            "TP2": p.get("TP2"),
            "Quantity": p.get("Qty"),
            "TP1 hit": p.get("TP1_Hit"),
            "Realized P&L": p.get("RealizedPnL"),
            "Status": p.get("Status"),
        })
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True,
                 column_config={
                     "Entry": st.column_config.NumberColumn(format="%.5f"),
                     "Stop-loss": st.column_config.NumberColumn(format="%.5f"),
                     "TP1": st.column_config.NumberColumn(format="%.5f"),
                     "TP2": st.column_config.NumberColumn(format="%.5f"),
                     "Quantity": st.column_config.NumberColumn(format="%.4f"),
                     "Realized P&L": st.column_config.NumberColumn(format="%.2f"),
                 })
    st.caption("Only one Stop-loss column is shown. After TP1, the active stop may move to the entry price (break-even) according to the bot's management rule.")
else:
    st.info("No automatic paper positions are currently open across the configured markets.")

if side in ("BUY","SELL"):
    levels = make_trade_levels(last, side, risk_pct, rr2)
    if levels:
        entry, stop, tp1, tp2 = levels["entry"], levels["stop"], levels["tp1"], levels["tp2"]
        risk_per_unit = abs(entry-stop)
        risk_cash = st.session_state.balance * risk_pct/100
        units = risk_cash / max(risk_per_unit, 1e-9)
        a,b,c,d = st.columns(4)
        a.metric("Entry (signal close)", f"{entry:,.5f}")
        b.metric("Stop loss", f"{stop:,.5f}")
        c.metric("TP1 (60%)", f"{tp1:,.5f}")
        d.metric("TP2 (40%)", f"{tp2:,.5f}")
        st.caption(f"Indicative sizing: {units:,.4f} units using price-distance risk math. This is not contract-aware sizing; Forex/Gold contract sizes, pip values, leverage, fees, swaps and slippage differ by broker.")
        can_record_trade = (not st.session_state.halted and st.session_state.trades_today < max_trades_per_day)
        if can_record_trade and st.button(f"➕ Record {side} paper trade"):
            st.session_state.trades.append({
                "Time": datetime.now().isoformat(timespec="seconds"),
                "Market": market_name, "Symbol": cfg["symbol"], "TF": signal_tf, "Side": side,
                "Score": score, "Entry": entry, "Stop": stop, "TP1": tp1, "TP2": tp2,
                "Status": "OPEN (manual tracking)", "Virtual risk budget": risk_cash
            })
            st.session_state.trades_today += 1
            persist_trade_log(st.session_state.trades)
            st.success("Paper setup recorded. No real order was sent.")
else:
    levels = None

with st.expander("AI Brain — condition-by-condition explanation", expanded=True):
    st.write("Directional signals use trend, momentum, DI, VWAP, Bollinger Bands and higher-timeframe alignment. Stop-loss and target levels are validated so BUY targets are above entry and SELL targets are below entry.")
    for name, passed in checks.items():
        st.write(("✅ " if passed else "⬜ ") + name)

with st.expander("🧪 Advanced Strategy Discovery Lab", expanded=False):
    st.write("Research-only candidate testing. It compares a small, explicit strategy library (including an SMC candidate) on chronological train/validation segments; it does not guarantee profitability and does not silently replace the running strategy.")
    st.caption("Candidates are assessed after estimated round-trip costs. A candidate is only research-eligible when it has enough validation trades and positive validation expectancy. Final adoption still requires longer walk-forward and paper-trading validation.")
    if st.button("Run strategy discovery on selected market", key="run_strategy_discovery"):
        try:
            from trading_engine.strategy_discovery import discover_strategies
            _disc = discover_strategies(data, cost_bps=2.0, min_validation_trades=8)
            if _disc.empty:
                st.warning("No candidate met the minimum validation sample. Collect more history or try another timeframe.")
            else:
                st.dataframe(_disc, use_container_width=True, hide_index=True)
                st.info("Research results only: candidates are not automatically promoted to the active strategy by this screen.")
        except Exception as _exc:
            st.error(f"Strategy discovery failed safely: {str(_exc)[:160]}")

st.subheader("Candlestick chart — indicators")
chart_df = data.tail(180).copy()
fig = go.Figure(data=[go.Candlestick(x=chart_df.index, open=chart_df["Open"], high=chart_df["High"], low=chart_df["Low"], close=chart_df["Close"], name="Candles")])
for col, label in [("EMA9","EMA 9"),("EMA21","EMA 21"),("EMA50","EMA 50"),("EMA200","EMA 200"),("BB_Upper","BB Upper"),("BB_Mid","BB Middle"),("BB_Lower","BB Lower"),("Support","Support"),("Resistance","Resistance"),("PowerUpper","Power Upper"),("PowerMid","Power Mid"),("PowerLower","Power Lower")]:
    if col in chart_df:
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[col], mode="lines", name=label, connectgaps=False))
for col, label in [("SMC_Active_OB_Top", "SMC Active OB Top"), ("SMC_Active_OB_Bottom", "SMC Active OB Bottom")]:
    if col in chart_df and chart_df[col].notna().any():
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[col], mode="lines", name=label, line=dict(dash="dash"), connectgaps=True))
fig.update_layout(height=620, xaxis_rangeslider_visible=False, legend_orientation="h", margin=dict(l=10,r=10,t=30,b=10))
# Draw active automatic position levels when they belong to the selected market.
for pos in st.session_state.auto_open_positions:
    if pos.get("Market") == market_name:
        for key, label in [("Entry", "Paper Entry"), ("Stop", "Paper Stop"), ("TP1", "Paper TP1"), ("TP2", "Paper TP2")]:
            if key in pos:
                fig.add_hline(y=float(pos[key]), line_dash="dash", annotation_text=label, annotation_position="top left")
# Re-render after adding trade-level lines.
st.plotly_chart(fig, use_container_width=True, key="candlestick_chart_with_trade_levels")
st.subheader("Recent candles and indicators")
show_cols = ["Open","High","Low","Close","Volume","RSI","MFI","ADX","EMA9","EMA21","EMA50","EMA200","BB_Lower","BB_Mid","BB_Upper","VWAP","ATR","RelVol","Support","Resistance","PowerLower","PowerMid","PowerUpper","PLUS_DI","MINUS_DI"]
st.dataframe(data[show_cols].tail(25).sort_index(ascending=False), use_container_width=True)

st.subheader("Paper trade log")
if st.session_state.trades:
    log = pd.DataFrame(st.session_state.trades)
    st.dataframe(log.iloc[::-1], use_container_width=True)
    csv = log.to_csv(index=False).encode("utf-8")
    st.download_button("Download paper log CSV", csv, "paper_trade_log.csv", "text/csv")
    c1,c2 = st.columns(2)
    with c1:
        if st.button("Mark selected/latest trade as TP1 hit"):
            for tr in reversed(st.session_state.trades):
                if tr["Status"].startswith("OPEN"):
                    tr["Status"] = "TP1 HIT — 60% closed; stop to breakeven (manual simulation)"
                    persist_trade_log(st.session_state.trades)
                    break
            st.rerun()
    with c2:
        if st.button("Close latest open trade at current close"):
            for tr in reversed(st.session_state.trades):
                if tr["Status"].startswith("OPEN"):
                    exit_price = float(last["Close"])
                    pnl_per_unit = (exit_price-tr["Entry"]) if tr["Side"]=="BUY" else (tr["Entry"]-exit_price)
                    # Placeholder estimate only; real instrument contract value is unknown.
                    estimated_pnl = pnl_per_unit * (tr["Virtual risk budget"] / max(abs(tr["Entry"]-tr["Stop"]),1e-9))
                    tr["Status"] = f"CLOSED @ {exit_price:.5f} | est. PnL {estimated_pnl:.2f} price-units"
                    st.session_state.balance += estimated_pnl
                    persist_trade_log(st.session_state.trades)
                    break
            st.rerun()
else:
    st.info("No paper trades recorded yet. When a setup passes the filters, use the Record button to add it to the log.")

with st.expander("Backtest (historical, simplified)"):
    st.warning("Backtest is approximate and educational: no spread, commissions, swaps, funding, broker contract specs, or realistic slippage. Intrabar stop/target ordering is conservative, but this is not a substitute for a proper event-driven backtester.")
    bt_period = st.selectbox("Backtest range (based on available Yahoo candles)", ["Current loaded history"], index=0)
    cost_bps = st.slider("Estimated round-trip costs (basis points of price × units)", 0.0, 20.0, 2.0, 0.5)
    if st.button("Run backtest"):
        bt, final_balance, err = run_backtest(raw, htf, signal_tf, htf_tf, score_threshold, risk_pct, rr2, float(st.session_state.starting_balance_applied), max_daily_loss, cost_bps)
        diagnostics = diagnose_backtest_filters(raw, htf, signal_tf, htf_tf, score_threshold, max_daily_loss)
        st.subheader("Backtest diagnostics")
        st.caption("Counts explain why bars were skipped or rejected. They are diagnostic counts, not independent trades, and do not change the strategy rules.")
        st.dataframe(diagnostics, hide_index=True, use_container_width=True)
        if err:
            st.error(err)
        elif bt.empty:
            st.warning("No trades passed the filters in the loaded history. Use the diagnostics above to identify the main blocker.")
        else:
            wins = (bt["PnL (price units)"] > 0).sum()
            win_rate = 100*wins/len(bt)
            gross_profit = bt.loc[bt["PnL (price units)"]>0,"PnL (price units)"].sum()
            gross_loss = abs(bt.loc[bt["PnL (price units)"]<0,"PnL (price units)"].sum())
            pf = gross_profit/gross_loss if gross_loss else np.inf
            x1,x2,x3,x4 = st.columns(4)
            x1.metric("Trades", len(bt))
            x2.metric("Win rate", f"{win_rate:.1f}%")
            x3.metric("Profit factor", f"{pf:.2f}" if np.isfinite(pf) else "∞")
            x4.metric("Final simulated balance (USD proxy)", f"${final_balance:,.2f}")
            equity = bt["Balance"]
            peak = equity.cummax()
            drawdown = ((peak - equity) / peak.replace(0, np.nan) * 100).max() if len(equity) else 0.0
            expectancy = bt["PnL (price units)"].mean()
            x5,x6 = st.columns(2)
            x5.metric("Max drawdown (trade-equity proxy)", f"{drawdown:.2f}%")
            x6.metric("Average P&L / trade (price units)", f"{expectancy:.5f}")
            st.dataframe(bt.iloc[::-1], use_container_width=True)
            st.download_button("Download backtest CSV", bt.to_csv(index=False).encode("utf-8"), "backtest_results.csv", "text/csv")

st.divider()
st.caption("Important: this version is paper-only. No broker API, exchange API, wallet, private key, or live trading is connected. Validate data quality and out-of-sample performance before considering any real-money use.")
