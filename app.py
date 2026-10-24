import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import json
from pathlib import Path

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
STARTING_BALANCE = 100_000.0  # INR virtual balance
MARKETS = {
    "Gold (XAUUSD)": {"symbol": "GC=F", "signal_tf": "5m", "allowed_tfs": ["5m", "15m", "30m"], "htf": "30m", "kind": "gold"},
    "US Oil (WTI proxy)": {"symbol": "CL=F", "signal_tf": "15m", "allowed_tfs": ["5m", "15m", "30m"], "htf": "30m", "kind": "oil"},
    "EUR/USD": {"symbol": "EURUSD=X", "signal_tf": "15m", "allowed_tfs": ["5m", "15m", "30m"], "htf": "30m", "kind": "forex"},
    "GBP/USD": {"symbol": "GBPUSD=X", "signal_tf": "15m", "allowed_tfs": ["5m", "15m", "30m"], "htf": "30m", "kind": "forex"},
    "AUD/USD": {"symbol": "AUDUSD=X", "signal_tf": "15m", "allowed_tfs": ["5m", "15m", "30m"], "htf": "30m", "kind": "forex"},
    "USD/JPY": {"symbol": "JPY=X", "signal_tf": "15m", "allowed_tfs": ["5m", "15m", "30m"], "htf": "30m", "kind": "forex"},
    "BTC/USD": {"symbol": "BTC-USD", "signal_tf": "15m", "allowed_tfs": ["5m", "15m", "30m"], "htf": "30m", "kind": "crypto"},
    "SOL/USD": {"symbol": "SOL-USD", "signal_tf": "15m", "allowed_tfs": ["5m", "15m", "30m"], "htf": "30m", "kind": "crypto"},
    "ETH/USD": {"symbol": "ETH-USD", "signal_tf": "15m", "allowed_tfs": ["5m", "15m", "30m"], "htf": "30m", "kind": "crypto"},
}
INTERVAL_MAP = {"5m": ("5m", "5d"), "15m": ("15m", "60d"), "30m": ("30m", "60d")}

# -----------------------------
# Market session intelligence (Yahoo Finance schedule estimates + MT5-ready)
# -----------------------------
from zoneinfo import ZoneInfo

def market_session_status(market_name, config, now_utc=None):
    """Estimate session state from public market conventions.
    Yahoo Finance does not expose broker tradability; MT5 can later override this
    with broker-specific symbol sessions when an MT5 connector is configured.
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    kind = config.get("kind", "")
    if kind == "crypto":
        return {"Session status": "OPEN", "Session detail": "24/7 market (provider data still checked)", "Session source": "Schedule estimate"}
    if kind == "forex":
        now_ny = now_utc.astimezone(ZoneInfo("America/New_York"))
        wd = now_ny.weekday()  # Monday=0, Sunday=6
        t = now_ny.time()
        open_now = not (wd == 5 or (wd == 6 and t < datetime.strptime("17:00", "%H:%M").time())
                        or (wd == 4 and t >= datetime.strptime("17:00", "%H:%M").time()))
        detail = "Approx. Sun 17:00–Fri 17:00 New York time"
        return {"Session status": "OPEN" if open_now else "CLOSED", "Session detail": detail, "Session source": "Schedule estimate"}
    if kind in ("gold", "oil"):
        now_ct = now_utc.astimezone(ZoneInfo("America/Chicago"))
        wd, t = now_ct.weekday(), now_ct.time()
        t1600 = datetime.strptime("16:00", "%H:%M").time()
        t1700 = datetime.strptime("17:00", "%H:%M").time()
        closed_weekend = wd == 5 or (wd == 6 and t < t1700) or (wd == 4 and t >= t1600)
        daily_break = t >= t1600 and t < t1700
        open_now = not closed_weekend and not daily_break
        detail = "Approx. Sun–Fri, 17:00–16:00 Chicago time; daily break 16:00–17:00"
        return {"Session status": "OPEN" if open_now else "CLOSED", "Session detail": detail, "Session source": "Futures schedule estimate"}
    return {"Session status": "UNKNOWN", "Session detail": "No configured schedule", "Session source": "Unknown"}

def candle_freshness_status(df, timeframe, now_utc=None):
    """Reject entries on old candles; tolerate some provider lag."""
    if df is None or df.empty:
        return False, "No candles"
    now_utc = now_utc or datetime.now(timezone.utc)
    try:
        last_ts = pd.Timestamp(df.index[-1])
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        last_ts = last_ts.tz_convert("UTC")
        mins = int(timeframe.replace("m", ""))
        age_min = max(0.0, (now_utc - last_ts.to_pydatetime()).total_seconds() / 60.0)
        # Candles are timestamped at their start; allow two bars plus 10 minutes.
        limit = max(20, mins * 2 + 10)
        return age_min <= limit, f"Latest candle age {age_min:.0f} min (limit {limit} min)"
    except Exception:
        return False, "Unable to verify candle timestamp"


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
    return {"5m": 48, "15m": 40, "30m": 36}.get(interval, 40)

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
MIN_ML_TRADES = 30
LEARNING_MODEL_VERSION = "logistic-v1"


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
    """Return conservative learned confirmation setting and an optional trained ML model."""
    rows = _closed_learning_rows(trades)
    profile = {"closed_samples": int(len(rows)), "recent_win_rate": None,
               "min_confirmations": 6, "model": None, "model_ready": False,
               "model_note": "Collecting completed paper trades with entry features."}
    if len(rows):
        recent = rows.tail(20)
        win_rate = float(recent["_label"].mean())
        profile["recent_win_rate"] = win_rate
        # A small, bounded adjustment: tighten after poor recent results; relax only slightly after good results.
        if len(recent) >= 10 and win_rate < 0.40:
            profile["min_confirmations"] = 7
        elif len(recent) >= 10 and win_rate > 0.65:
            profile["min_confirmations"] = 5
    if not SKLEARN_AVAILABLE:
        profile["model_note"] = "scikit-learn is not installed; adaptive rule analysis is active, ML is disabled."
        return profile
    if len(rows) < MIN_ML_TRADES or rows["_label"].nunique() < 2 or rows["_label"].value_counts().min() < 8:
        profile["model_note"] = f"ML needs at least {MIN_ML_TRADES} completed feature-rich trades and at least 8 wins and 8 losses."
        return profile
    try:
        x = rows[ML_FEATURES].apply(pd.to_numeric, errors="coerce")
        y = rows["_label"].astype(int)
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ])
        model.fit(x, y)
        profile["model"] = model
        profile["model_ready"] = True
        profile["model_note"] = f"ML model trained on {len(rows)} completed trades ({int(y.sum())} wins, {int((1-y).sum())} losses)."
    except Exception as exc:
        profile["model_note"] = f"ML training unavailable: {str(exc)[:100]}"
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
def scan_all_market_signals():
    """Fetch and evaluate each configured market for the overview dashboard.
    This reports signals only; it does not open paper positions for every market.
    """
    rows = []
    profile = learning_profile(st.session_state.get("trades", load_trade_log())) if "trades" in st.session_state else learning_profile(load_trade_log())
    for name, config in MARKETS.items():
        tf = config["signal_tf"]
        session = market_session_status(name, config)
        try:
            raw_market = download_data(config["symbol"], tf)
            htf_market = download_data(config["symbol"], config["htf"])
            if raw_market.empty or len(raw_market) < 60:
                rows.append({
                    "Market": name, "Symbol": config["symbol"], "TF": tf,
                    "Last close": None, "HTF bias": "Unknown",
                    "Signal": "DATA UNAVAILABLE", "Signal strength (diagnostic)": None,
                    "RSI": None, "ADX": None, "Data status": "Insufficient candles or provider returned no data",
                    **session
                })
                continue
            market_data = add_adaptive_levels(add_indicators(raw_market), tf)
            htf_processed = add_adaptive_levels(add_indicators(htf_market), config["htf"]) if not htf_market.empty else htf_market
            bias_value, _ = htf_bias(htf_processed) if not htf_processed.empty else ("Unknown", None)
            fresh, freshness_note = candle_freshness_status(raw_market, tf)
            last_row = market_data.iloc[-1]
            best_side, best_score, _ = score_signal(last_row, bias_value, threshold=None, min_confirmations=profile["min_confirmations"])
            best_side, ml_prob, ml_note = apply_learning_gate(best_side, last_row, bias_value, best_score, profile)
            if session["Session status"] != "OPEN":
                best_side = "WAIT"
            elif not fresh:
                best_side = "WAIT"
            signal_value = best_side if best_side in ("BUY", "SELL") else "WAIT"
            rows.append({
                "Market": name, "Symbol": config["symbol"], "TF": tf,
                "Last close": round(float(last_row["Close"]), 5),
                "HTF bias": bias_value, "Signal": (best_side if best_side in ("BUY", "SELL") else "WAIT"),
                "Signal strength (diagnostic)": int(best_score),
                "ML win probability": round(ml_prob * 100, 1) if ml_prob is not None else None,
                "RSI": round(float(last_row["RSI"]), 1) if pd.notna(last_row["RSI"]) else None,
                "ADX": round(float(last_row["ADX"]), 1) if pd.notna(last_row["ADX"]) else None,
                "Data status": "OK" if fresh else f"STALE: {freshness_note}",
                **session
            })
        except Exception as exc:
            rows.append({
                "Market": name, "Symbol": config["symbol"], "TF": tf,
                "Last close": None, "HTF bias": "Unknown",
                "Signal": "DATA ERROR", "Signal strength (diagnostic)": None,
                "RSI": None, "ADX": None, "Data status": str(exc)[:120],
                **market_session_status(name, config)
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


def simulate_auto_paper_scan(data, market_name, symbol, timeframe, bias, threshold,
                             risk_pct, rr2, balance, trades_today, max_trades_per_day,
                             daily_start_balance, max_daily_loss_pct, open_positions):
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
                **{k: pos.get(k, np.nan) for k in ML_FEATURES}
            })
            events.append(f"Closed {side} {market_name} at stop-loss; estimated paper P&L {pnl:.2f} price-units.")
            continue
        if tp1_hit and not pos["TP1_Hit"]:
            qty = pos["Qty"]
            partial = ((pos["TP1"]-pos["Entry"]) if side=="BUY" else (pos["Entry"]-pos["TP1"])) * qty * 0.6
            pos["RealizedPnL"] += partial
            pos["TP1_Hit"] = True
            pos["Stop"] = pos["Entry"]  # move stop to break-even
            events.append(f"TP1 hit on {market_name}; simulated 60% close and stop moved to break-even.")
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

    if already_processed:
        processed[market_key] = bar_time
        st.session_state.auto_last_processed = processed
        return balance, trades_today, daily_start_balance, open_positions, events, "Existing positions checked; this candle was already evaluated for entry."

    if any(p["Market"] == market_name for p in open_positions):
        events.append("An open simulated position already exists for this market; no duplicate entry.")
        processed[market_key] = bar_time
        st.session_state.auto_last_processed = processed
        return balance, trades_today, daily_start_balance, open_positions, events, "Position already open for this market."

    current_trades = st.session_state.get("trades", load_trade_log())
    profile = learning_profile(current_trades)
    side, score, checks = score_signal(last, bias, threshold, profile["min_confirmations"])
    side, ml_probability, ml_note = apply_learning_gate(side, last, bias, score, profile)
    if side in ("BUY", "SELL"):
        events.append(ml_note)
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
                    "Symbol": symbol, "Timeframe": timeframe, "Side": side, "Score": score,
                    "Entry": entry, "Stop": stop, "TP1": tp1, "TP2": tp2,
                    "Qty": qty, "RiskBudget": risk_cash, "TP1_Hit": False, "RealizedPnL": 0.0,
                    **extract_ml_features(last, bias, score, side), "Status": "OPEN"
                }
                open_positions.append(pos)
                trades_today += 1
                events.append(f"Automatically opened paper {side} on {market_name} at {entry:.5f} (directional signal).")
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
                pnl = partial_pnl + ((exit_price-entry) if side=="BUY" else (entry-exit_price))*qty*0.4
                break
            if tp1_hit and partial_pnl == 0.0:
                partial_pnl = ((tp1-entry) if side=="BUY" else (entry-tp1))*qty*0.6
                stop = entry  # simulated break-even after TP1
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
st.caption("Rule-based scoring with standard indicators + multi-timeframe filters + automatic simulated trade entry/exit. Paper trading only; no broker, wallet, or live-order connection.")
with st.sidebar:
    st.header("Bot settings")
    # Keep changes staged until the user explicitly confirms them.
    with st.form("bot_settings_form"):
        market_name = st.selectbox("Market", list(MARKETS.keys()), key="settings_market")
        cfg = MARKETS[market_name]
        allowed_tfs = cfg["allowed_tfs"]
        default_tf = cfg["signal_tf"]
        current_tf = st.session_state.get("settings_signal_tf")
        tf_index = allowed_tfs.index(current_tf) if current_tf in allowed_tfs else allowed_tfs.index(default_tf)
        signal_tf = st.selectbox("Signal timeframe", allowed_tfs, index=tf_index, key="settings_signal_tf_widget")
        htf_tf = st.selectbox("Higher-timeframe filter", ["5m", "15m", "30m"], index=1, key="settings_htf_tf")
        score_threshold = st.slider("Minimum Signal strength", 50, 95, 70, 1, key="settings_score_threshold")
        risk_pct = st.slider("Risk per trade (%)", 0.25, 1.0, 0.5, 0.25, key="settings_risk_pct")
        max_daily_loss = st.slider("Daily loss stop (%)", 1.0, 3.0, 2.0, 0.5, key="settings_max_daily_loss")
        max_trades_per_day = st.slider("Maximum paper trades per day", 1, 30, 20, 1, key="settings_max_trades_per_day")
        rr2 = st.slider("TP2 risk/reward", 1.5, 2.5, 2.0, 0.1, key="settings_rr2")
        confirm_settings = st.form_submit_button("✅ Confirm settings", type="primary", use_container_width=True)
    if confirm_settings:
        st.session_state.settings_market_applied = market_name
        st.session_state.settings_signal_tf = signal_tf
        st.session_state.settings_htf_tf_applied = htf_tf
        st.session_state.settings_score_threshold_applied = None
        st.session_state.settings_risk_pct_applied = risk_pct
        st.session_state.settings_max_daily_loss_applied = max_daily_loss
        st.session_state.settings_max_trades_per_day_applied = max_trades_per_day
        st.session_state.settings_rr2_applied = rr2
        st.success("Settings confirmed for this session.")
    st.caption("The Brain logic evaluates trend, momentum, DI, VWAP, volatility and higher-timeframe alignment. Entries are simulated paper trades only.")

if "balance" not in st.session_state:
    st.session_state.balance = STARTING_BALANCE
if "trades" not in st.session_state:
    st.session_state.trades = load_trade_log()
if "daily_start" not in st.session_state:
    st.session_state.daily_start = STARTING_BALANCE
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

# Streamlit reruns the script on this timer while the browser session is connected.
# Automatic paper scans are gated below so ordinary UI reruns do not trigger new entries.
auto_refresh_count = st_autorefresh(interval=60_000, limit=None, key="ai_brain_auto_refresh_60s")
if "auto_last_refresh_count" not in st.session_state:
    st.session_state.auto_last_refresh_count = auto_refresh_count

today_key = datetime.now().date().isoformat()
if st.session_state.trade_day != today_key:
    st.session_state.trade_day = today_key
    st.session_state.trades_today = 0
    st.session_state.daily_start = st.session_state.balance
    st.session_state.halted = False

col1, col2, col3, col4 = st.columns(4)
col1.metric("Virtual balance", f"₹{st.session_state.balance:,.2f}")
col2.metric("Paper trades today", f"{st.session_state.trades_today}/{max_trades_per_day}")
col3.metric("Daily loss stop", f"{max_daily_loss:.1f}%")
col4.metric("Risk per setup", f"{risk_pct:.2f}%")

if st.button("🔄 Refresh all market data", type="primary"):
    st.cache_data.clear()

st.subheader("🌐 All-market signal dashboard")
st.caption("Scans every configured market for BUY / SELL / WAIT signals. Automatic paper trading also evaluates all configured markets, not only the market selected in Bot settings.")
with st.spinner("Scanning all configured markets..."):
    overview = scan_all_market_signals()
if not overview.empty:
    st.dataframe(overview, use_container_width=True, hide_index=True)
    usable = int((overview["Data status"] == "OK").sum())
    st.caption(f"Markets with usable data: {usable}/{len(overview)}. Provider data may be delayed or unavailable for some symbols.")
else:
    st.warning("No market scan results are available yet.")

st.divider()
st.subheader(f"📊 Detailed chart and indicators: {market_name}")
st.caption("The Market dropdown selects which market's detailed chart and indicator diagnostics to display. It does not limit automatic paper trading.")

with st.spinner("Downloading selected-market candles and evaluating the AI Brain..."):
    raw = download_data(cfg["symbol"], signal_tf)
    htf = download_data(cfg["symbol"], htf_tf)
    data = add_adaptive_levels(add_indicators(raw), signal_tf) if not raw.empty else pd.DataFrame()
    htf = add_adaptive_levels(add_indicators(htf), htf_tf) if not htf.empty else htf
    bias, htf_last = htf_bias(htf) if not htf.empty else ("Unknown", None)

if data.empty or len(data) < 60:
    st.error("Market data is unavailable or too short from Yahoo Finance right now. Try Refresh, another symbol, or another timeframe.")
    st.info("Yahoo Finance intraday data can be delayed, incomplete, or unavailable. Gold uses GC=F futures as a proxy, not a broker's exact XAUUSD CFD quote.")
    st.stop()

last = data.iloc[-1]
learning = learning_profile(st.session_state.trades)
side, score, checks = score_signal(last, bias, score_threshold, learning["min_confirmations"])
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

with st.expander("🧠 Self-learning engine — trade outcomes + machine learning", expanded=True):
    l1, l2, l3 = st.columns(3)
    l1.metric("Completed learning trades", learning["closed_samples"])
    l2.metric("Recent win rate", f"{learning['recent_win_rate']:.1%}" if learning["recent_win_rate"] is not None else "Not enough data")
    l3.metric("ML model", "TRAINED" if learning["model_ready"] else "COLLECTING DATA")
    st.write(learning["model_note"])
    st.write(f"Adaptive rule confirmations: {learning['min_confirmations']} of 10 checks.")
    if ml_probability is not None:
        st.write(f"Current setup estimated win probability: {ml_probability:.1%}.")
    st.caption("The model learns only from completed paper trades that contain entry features. Before enough examples exist, the bot transparently falls back to rule-based signals. Probabilities are estimates, not guarantees.")

if st.session_state.halted:
    st.error("Paper bot is halted by the daily loss control. It will reset on the next calendar day.")
elif st.session_state.trades_today >= max_trades_per_day:
    st.error(f"Daily trade limit reached ({max_trades_per_day}). No more paper trades will be allowed today.")
elif side in ("BUY","SELL"):
    st.success(f"Potential {side} setup — paper signal. {ml_note} This is not a guaranteed trade.")
else:
    st.warning(f"WAIT — no setup passed all filters. Best-side Signal strength: {score}/100. {ml_note}")

st.subheader("Automatic paper trading")
st.caption("Paper trading only. On this Streamlit Cloud deployment, scans run about every 60 seconds only while the app session is active. This is NOT a persistent background worker: cloud sleep or session disconnects can interrupt scans. True unattended operation requires a separately hosted worker (for example, on a VPS) and persistent storage.")
st.success("Automatic paper scanner: RUNNING")


def run_all_market_scan():
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
            session = market_session_status(scan_name, scan_cfg)
            if session["Session status"] != "OPEN":
                all_events.append(f"{scan_name}: no new entries; market session is {session['Session status']} ({session['Session detail']}).")
                continue
            fresh, freshness_note = candle_freshness_status(scan_raw, scan_tf)
            if not fresh:
                all_events.append(f"{scan_name}: entries blocked because data is stale/uncertain ({freshness_note}).")
                continue
            scan_data = add_adaptive_levels(add_indicators(scan_raw), scan_tf)
            scan_htf = add_adaptive_levels(add_indicators(scan_htf_raw), scan_cfg["htf"]) if not scan_htf_raw.empty else pd.DataFrame()
            scan_bias, _ = htf_bias(scan_htf) if not scan_htf.empty else ("Unknown", None)
            (st.session_state.balance, st.session_state.trades_today,
             st.session_state.auto_daily_start_balance, st.session_state.auto_open_positions,
             scan_events, scan_message) = simulate_auto_paper_scan(
                scan_data, scan_name, scan_cfg["symbol"], scan_tf, scan_bias, None,
                risk_pct, rr2, st.session_state.balance, st.session_state.trades_today,
                max_trades_per_day, st.session_state.auto_daily_start_balance,
                max_daily_loss, st.session_state.auto_open_positions
            )
            all_events.extend(scan_events)
            if scan_message:
                messages.append(f"{scan_name}: {scan_message}")
        except Exception as exc:
            all_events.append(f"{scan_name}: scan error: {str(exc)[:100]}")
    st.session_state.last_auto_events = all_events
    st.session_state.last_auto_message = "All-market scan completed. " + (" ".join(messages) if messages else "No new entries were opened; see market signals and events.")

# Run once for each 60-second timer tick while the scanner is armed.
if st.session_state.auto_bot_status == "RUNNING" and auto_refresh_count > st.session_state.auto_last_refresh_count:
    st.session_state.auto_last_refresh_count = auto_refresh_count
    run_all_market_scan()
    st.rerun()
else:
    # Keep the counter in sync while paused/stopped so restarting doesn't replay old ticks.
    st.session_state.auto_last_refresh_count = auto_refresh_count

if st.button("🔎 Scan market & manage paper positions now", type="primary"):
    run_all_market_scan()
    st.rerun()

if st.session_state.get("last_auto_message"):
    st.info(st.session_state.last_auto_message)
for ev in st.session_state.get("last_auto_events", []):
    st.write("• " + ev)

st.markdown("**Open automatic paper positions — all markets**")
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

st.subheader("Candlestick chart — indicators")
chart_df = data.tail(180).copy()
fig = go.Figure(data=[go.Candlestick(x=chart_df.index, open=chart_df["Open"], high=chart_df["High"], low=chart_df["Low"], close=chart_df["Close"], name="Candles")])
for col, label in [("EMA9","EMA 9"),("EMA21","EMA 21"),("EMA50","EMA 50"),("EMA200","EMA 200"),("BB_Upper","BB Upper"),("BB_Mid","BB Middle"),("BB_Lower","BB Lower"),("Support","Support"),("Resistance","Resistance"),("PowerUpper","Power Upper"),("PowerMid","Power Mid"),("PowerLower","Power Lower")]:
    if col in chart_df:
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[col], mode="lines", name=label, connectgaps=False))
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
        bt, final_balance, err = run_backtest(raw, htf, signal_tf, htf_tf, score_threshold, risk_pct, rr2, STARTING_BALANCE, max_daily_loss, cost_bps)
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
            x4.metric("Final model balance", f"₹{final_balance:,.2f}")
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
