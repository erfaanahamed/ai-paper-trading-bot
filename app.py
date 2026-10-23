from __future__ import annotations
import json, math, os
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, brier_score_loss

APP_DIR = Path(__file__).resolve().parent
STATE_FILE = APP_DIR / "paper_state.json"
TRADE_LOG = APP_DIR / "paper_trades.csv"
START_BALANCE = 100_000.0  # virtual INR only
TIMEFRAMES = ["5m", "15m", "30m", "1h"]
TF_META = {
    "5m": {"interval": "5m", "period": "5d", "lookback": 48},
    "15m": {"interval": "15m", "period": "60d", "lookback": 40},
    "30m": {"interval": "30m", "period": "60d", "lookback": 36},
    "1h": {"interval": "60m", "period": "730d", "lookback": 30},
}
MARKETS = {
    "Gold / XAUUSD proxy": {"symbol": "GC=F", "kind": "gold"},
    "US Oil / WTI proxy": {"symbol": "CL=F", "kind": "oil"},
    "EURUSD": {"symbol": "EURUSD=X", "kind": "forex"},
    "GBPUSD": {"symbol": "GBPUSD=X", "kind": "forex"},
    "USDJPY": {"symbol": "JPY=X", "kind": "forex"},
    "AUDUSD": {"symbol": "AUDUSD=X", "kind": "forex"},
    "BTCUSD": {"symbol": "BTC-USD", "kind": "crypto"},
    "ETHUSD": {"symbol": "ETH-USD", "kind": "crypto"},
    "SOLUSD": {"symbol": "SOL-USD", "kind": "crypto"},
}
FEATURES = ["RSI", "MFI", "ADX", "PLUS_DI", "MINUS_DI", "EMA9_gap", "EMA21_gap", "EMA50_gap", "EMA200_gap", "EMA9_21", "BB_pos", "ATR_pct", "VWAP_gap", "RelVol", "Support_gap", "Resistance_gap", "Power_width_pct", "Body_atr", "Upper_wick_atr", "Lower_wick_atr", "Side"]

st.set_page_config(page_title="Brain AI Trading Bot V2", page_icon="🧠", layout="centered")

# Responsive styling for Android/mobile browsers.
st.markdown("""<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 100%;}
[data-testid="stMetric"] {padding: .55rem; border: 1px solid rgba(128,128,128,.25); border-radius: .6rem;}
[data-testid="stDataFrame"] {width: 100%;}
@media (max-width: 640px) {
  .block-container {padding-left: .65rem; padding-right: .65rem;}
  h1 {font-size: 1.65rem !important;}
  h2 {font-size: 1.3rem !important;}
  h3 {font-size: 1.1rem !important;}
}
</style>""", unsafe_allow_html=True)


def load_state():
    default = {"balance": START_BALANCE, "day": date.today().isoformat(), "day_start": START_BALANCE,
               "trades_today": 0, "running": True, "open_positions": [], "last_processed": {}, "events": []}
    try:
        if STATE_FILE.exists():
            saved = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            default.update(saved)
    except Exception:
        pass
    if default.get("day") != date.today().isoformat():
        default["day"] = date.today().isoformat()
        default["day_start"] = float(default.get("balance", START_BALANCE))
        default["trades_today"] = 0
    return default


def save_state():
    st.session_state.bot_state["events"] = st.session_state.bot_state.get("events", [])[-250:]
    STATE_FILE.write_text(json.dumps(st.session_state.bot_state, indent=2, default=str), encoding="utf-8")


def log_trade(record):
    frame = pd.DataFrame([record])
    frame.to_csv(TRADE_LOG, mode="a", header=not TRADE_LOG.exists(), index=False)


@st.cache_data(ttl=50, show_spinner=False)
def download_data(symbol: str, tf: str) -> pd.DataFrame:
    meta = TF_META[tf]
    try:
        df = yf.download(symbol, period=meta["period"], interval=meta["interval"], auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).title() for c in df.columns]
        needed = ["Open", "High", "Low", "Close"]
        if any(c not in df.columns for c in needed):
            return pd.DataFrame()
        if "Volume" not in df.columns:
            df["Volume"] = 0.0
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=needed)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df
    except Exception:
        return pd.DataFrame()


def add_indicators(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    x = df.copy()
    c, h, l, o = x.Close, x.High, x.Low, x.Open
    vol = x.Volume.fillna(0).clip(lower=0)
    delta = c.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    x["RSI"] = (100 - 100/(1+rs)).where(avg_loss.ne(0), 100.0).where(avg_gain.ne(0) | avg_loss.ne(0), 50.0)
    typical = (h+l+c)/3
    flow = typical * vol
    pos = flow.where(typical.diff() > 0, 0).rolling(14, min_periods=14).sum()
    neg = flow.where(typical.diff() < 0, 0).rolling(14, min_periods=14).sum()
    ratio = pos / neg.replace(0, np.nan)
    x["MFI"] = (100 - 100/(1+ratio)).where(neg.ne(0), 100.0).where(pos.ne(0) | neg.ne(0), 50.0)
    if vol.sum() <= 0:
        x["MFI"] = np.nan
    x["BB_Mid"] = c.rolling(20, min_periods=20).mean()
    sd = c.rolling(20, min_periods=20).std(ddof=0)
    x["BB_Upper"], x["BB_Lower"] = x.BB_Mid + 2*sd, x.BB_Mid - 2*sd
    prev = c.shift(1)
    tr = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    x["ATR"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    up, down = h.diff(), -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    plus_sm = plus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    minus_sm = minus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    atr_safe = x.ATR.replace(0, np.nan)
    x["PLUS_DI"], x["MINUS_DI"] = 100*plus_sm/atr_safe, 100*minus_sm/atr_safe
    di_sum = (x.PLUS_DI+x.MINUS_DI).replace(0, np.nan)
    dx = 100*(x.PLUS_DI-x.MINUS_DI).abs()/di_sum
    x["ADX"] = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    for n in (9, 21, 50, 200):
        x[f"EMA{n}"] = c.ewm(span=n, adjust=False).mean()
    # Session VWAP; volume is often unavailable/non-centralized on FX feeds.
    idx = pd.to_datetime(x.index)
    session = pd.Series(idx.date, index=x.index)
    cumulative_vol = vol.groupby(session).cumsum().replace(0, np.nan)
    x["VWAP"] = (typical*vol).groupby(session).cumsum()/cumulative_vol
    x["VWAP"] = x.VWAP.fillna(x.BB_Mid)
    x["VolAvg20"] = vol.rolling(20).mean()
    x["RelVol"] = vol/x.VolAvg20.replace(0, np.nan)
    n = TF_META[tf]["lookback"]
    x["Support"] = l.rolling(n, min_periods=max(8, n//2)).min().shift(1)
    x["Resistance"] = h.rolling(n, min_periods=max(8, n//2)).max().shift(1)
    x["PowerMid"] = (x.Support+x.Resistance)/2
    x["PowerWidth"] = x.Resistance-x.Support
    x["PowerUpper"] = x.Resistance + 0.25*x.ATR
    x["PowerLower"] = x.Support - 0.25*x.ATR
    x["NearSupport"] = x.Close <= x.Support + 0.35*x.ATR
    x["NearResistance"] = x.Close >= x.Resistance - 0.35*x.ATR
    body = (c-o).abs()
    x["EMA9_gap"] = (c-x.EMA9)/c.replace(0, np.nan)
    x["EMA21_gap"] = (c-x.EMA21)/c.replace(0, np.nan)
    x["EMA50_gap"] = (c-x.EMA50)/c.replace(0, np.nan)
    x["EMA200_gap"] = (c-x.EMA200)/c.replace(0, np.nan)
    x["EMA9_21"] = (x.EMA9-x.EMA21)/c.replace(0, np.nan)
    x["BB_pos"] = (c-x.BB_Mid)/(x.BB_Upper-x.BB_Mid).replace(0, np.nan)
    x["ATR_pct"] = x.ATR/c.replace(0, np.nan)
    x["VWAP_gap"] = (c-x.VWAP)/c.replace(0, np.nan)
    x["Support_gap"] = (c-x.Support)/c.replace(0, np.nan)
    x["Resistance_gap"] = (x.Resistance-c)/c.replace(0, np.nan)
    x["Power_width_pct"] = x.PowerWidth/c.replace(0, np.nan)
    candle_range = (h-l).replace(0, np.nan)
    x["Body_atr"] = body/x.ATR.replace(0, np.nan)
    x["Upper_wick_atr"] = (h-np.maximum(o,c))/x.ATR.replace(0, np.nan)
    x["Lower_wick_atr"] = (np.minimum(o,c)-l)/x.ATR.replace(0, np.nan)
    x["BullReversal"] = ((c > o) & ((np.minimum(o,c)-l > body*1.2) | ((c > o.shift(1)) & (o < c.shift(1)) & (c > o.shift(1)))).fillna(False))
    x["BearReversal"] = ((c < o) & ((h-np.maximum(o,c) > body*1.2) | ((c < o.shift(1)) & (o > c.shift(1)) & (c < o.shift(1)))).fillna(False))
    return x


def trend_bias(df: pd.DataFrame) -> str:
    if df.empty or len(df) < 205:
        return "Unknown"
    row = df.iloc[-2]  # use last fully completed candle, not the possibly forming one
    if row.Close > row.EMA50 and row.Close > row.EMA200 and row.EMA50 >= row.EMA200:
        return "Bullish"
    if row.Close < row.EMA50 and row.Close < row.EMA200 and row.EMA50 <= row.EMA200:
        return "Bearish"
    return "Mixed"


def feature_row(row: pd.Series, side: str) -> pd.DataFrame:
    vals = {k: row.get(k, np.nan) for k in FEATURES if k != "Side"}
    vals["Side"] = 1.0 if side == "BUY" else -1.0
    return pd.DataFrame([[vals.get(k, np.nan) for k in FEATURES]], columns=FEATURES)


def build_training_set(data: pd.DataFrame, horizon: int = 12):
    """Create side-specific labels: did TP1 (0.9R) occur before the initial SL in the next bars?"""
    rows, labels = [], []
    if len(data) < 280:
        return pd.DataFrame(columns=FEATURES), np.array([], dtype=int)
    start = max(220, 0)
    end = len(data)-horizon-1
    for i in range(start, end):
        r = data.iloc[i]
        atr = r.get("ATR", np.nan)
        if not np.isfinite(atr) or atr <= 0:
            continue
        for side in ("BUY", "SELL"):
            entry = float(r.Close)
            stop_dist = max(1.2*float(atr), abs(float(r.High)-float(r.Low)))
            stop = entry-stop_dist if side == "BUY" else entry+stop_dist
            tp1 = entry+0.9*stop_dist if side == "BUY" else entry-0.9*stop_dist
            label = 0
            for j in range(i+1, min(i+horizon+1, len(data))):
                bar = data.iloc[j]
                stop_hit = bar.Low <= stop if side == "BUY" else bar.High >= stop
                target_hit = bar.High >= tp1 if side == "BUY" else bar.Low <= tp1
                if stop_hit:  # conservative if both are touched within the same candle
                    label = 0
                    break
                if target_hit:
                    label = 1
                    break
            vals = {k: r.get(k, np.nan) for k in FEATURES if k != "Side"}
            vals["Side"] = 1.0 if side == "BUY" else -1.0
            rows.append(vals)
            labels.append(label)
    return pd.DataFrame(rows, columns=FEATURES), np.asarray(labels, dtype=int)


@st.cache_resource(show_spinner=False)
def train_brain_cached(symbol: str, tf: str, last_index: str, data_for_model: pd.DataFrame):
    # Cache key includes candle index; no feature rows after each sample's timestamp are used.
    X, y = build_training_set(data_for_model)
    if len(X) < 120 or len(np.unique(y)) < 2:
        return None, {"samples": len(X), "holdout_win_rate": None, "holdout_brier": None}
    split = int(len(X)*0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("forest", RandomForestClassifier(n_estimators=180, max_depth=7, min_samples_leaf=8,
                                           class_weight="balanced_subsample", random_state=42,
                                           n_jobs=-1))
    ])
    model.fit(X_train, y_train)
    holdout_prob = model.predict_proba(X_test)[:, 1]
    holdout_pred = (holdout_prob >= 0.5).astype(int)
    holdout = {"samples": len(X), "holdout_win_rate": float(y_test.mean()*100),
               "holdout_accuracy": float(accuracy_score(y_test, holdout_pred)*100),
               "holdout_brier": float(brier_score_loss(y_test, holdout_prob))}
    # Refit on all labeled examples for the live estimate. This is not a guarantee of calibration.
    model.fit(X, y)
    return model, holdout


def signal_quality(row: pd.Series, side: str, biases: dict) -> tuple[bool, list[str]]:
    reasons = []
    h1, m30, m15 = biases.get("1h", "Unknown"), biases.get("30m", "Unknown"), biases.get("15m", "Unknown")
    required = "Bullish" if side == "BUY" else "Bearish"
    if h1 != required: reasons.append(f"H1 is {h1}, not {required}")
    if m30 != required: reasons.append(f"M30 is {m30}, not {required}")
    if m15 not in (required, "Mixed"): reasons.append(f"M15 is against the {side} setup")
    if side == "BUY":
        if not (row.EMA9 > row.EMA21): reasons.append("EMA 9 is not above EMA 21")
        if not (row.PLUS_DI > row.MINUS_DI): reasons.append("+DI is not above -DI")
        if row.RSI >= 78: reasons.append("RSI is extremely overbought")
    else:
        if not (row.EMA9 < row.EMA21): reasons.append("EMA 9 is not below EMA 21")
        if not (row.MINUS_DI > row.PLUS_DI): reasons.append("-DI is not above +DI")
        if row.RSI <= 22: reasons.append("RSI is extremely oversold")
    if not np.isfinite(row.ATR) or row.ATR <= 0: reasons.append("ATR is unavailable")
    if not np.isfinite(row.ADX) or row.ADX < 12: reasons.append("ADX is too weak")
    return not reasons, reasons


def make_levels(row: pd.Series, side: str, buffer_atr: float = 0.05):
    entry = float(row.Close)
    atr = float(row.ATR)
    if not np.isfinite(atr) or atr <= 0:
        return None
    if side == "BUY":
        stop = min(float(row.Low), entry-1.2*atr)
        risk = entry-stop
        if not np.isfinite(risk) or risk <= 0: return None
        return {"Entry": entry, "InitialStop": stop, "Stop": stop, "TP1": entry+0.9*risk,
                "TP2": entry+2.0*risk, "RiskPerUnit": risk, "BEBuffer": buffer_atr*atr}
    stop = max(float(row.High), entry+1.2*atr)
    risk = stop-entry
    if not np.isfinite(risk) or risk <= 0: return None
    return {"Entry": entry, "InitialStop": stop, "Stop": stop, "TP1": entry-0.9*risk,
            "TP2": entry-2.0*risk, "RiskPerUnit": risk, "BEBuffer": buffer_atr*atr}


def close_position(pos: dict, exit_price: float, reason: str, state: dict):
    side = pos["Side"]
    qty = float(pos["Qty"])
    if pos.get("TP1_Hit"):
        pnl = float(pos.get("RealizedPnL", 0.0)) + ((exit_price-pos["Entry"]) if side == "BUY" else (pos["Entry"]-exit_price))*qty*0.4
    else:
        pnl = ((exit_price-pos["Entry"]) if side == "BUY" else (pos["Entry"]-exit_price))*qty
    state["balance"] = float(state["balance"])+pnl
    rec = {"Closed": datetime.now().isoformat(timespec="seconds"), **pos,
           "Exit": float(exit_price), "ExitReason": reason, "PnL_virtual": pnl, "Status": "CLOSED"}
    log_trade(rec)
    state["events"].append(f"{pos['Side']} {pos['Market']} closed: {reason}; virtual P&L ₹{pnl:,.2f}.")


def manage_positions(market_name: str, frames: dict, state: dict):
    kept = []
    for pos in state["open_positions"]:
        if pos["Market"] != market_name:
            kept.append(pos); continue
        position_data = frames.get(pos.get("Timeframe", ""), pd.DataFrame())
        if position_data.empty or len(position_data) < 2:
            kept.append(pos); continue
        bar = position_data.iloc[-2]  # most recent completed candle for this position timeframe
        side = pos["Side"]
        stop_hit = bar.Low <= pos["Stop"] if side == "BUY" else bar.High >= pos["Stop"]
        tp1_hit = bar.High >= pos["TP1"] if side == "BUY" else bar.Low <= pos["TP1"]
        tp2_hit = bar.High >= pos["TP2"] if side == "BUY" else bar.Low <= pos["TP2"]
        # If stop and target are both touched in one candle, conservative assumption: stop first.
        if stop_hit:
            close_position(pos, float(pos["Stop"]), "SL / break-even-plus", state)
            continue
        if tp1_hit and not pos.get("TP1_Hit", False):
            side_pnl = ((pos["TP1"]-pos["Entry"]) if side == "BUY" else (pos["Entry"]-pos["TP1"])) * float(pos["Qty"])*0.6
            pos["RealizedPnL"] = float(pos.get("RealizedPnL", 0.0))+side_pnl
            pos["TP1_Hit"] = True
            pos["Stop"] = pos["Entry"] + pos["BEBuffer"] if side == "BUY" else pos["Entry"] - pos["BEBuffer"]
            pos["RemainingPct"] = 40
            state["events"].append(f"{market_name}: TP1 hit — 60% booked; stop moved to break-even + buffer.")
        if tp2_hit:
            if not pos.get("TP1_Hit", False):
                side_pnl = ((pos["TP1"]-pos["Entry"]) if side == "BUY" else (pos["Entry"]-pos["TP1"])) * float(pos["Qty"])*0.6
                pos["RealizedPnL"] = float(pos.get("RealizedPnL", 0.0))+side_pnl
                pos["TP1_Hit"] = True
            close_position(pos, float(pos["TP2"]), "TP2 — remaining 40% booked", state)
            continue
        kept.append(pos)
    state["open_positions"] = kept


def scan_and_trade(market_name: str, symbol: str, entry_tf: str, frames: dict, model, stats: dict,
                   state: dict, probability_threshold: float, risk_pct: float, max_daily_loss_pct: float,
                   max_trades_per_day: int):
    if entry_tf not in frames or frames[entry_tf].empty or len(frames[entry_tf]) < 230:
        state["events"].append(f"{market_name}: not enough {entry_tf} candles; no new entry.")
        return
    entry_data = frames[entry_tf]
    # Last bar is excluded from signals to avoid trading a potentially unfinished candle.
    closed = entry_data.iloc[:-1]
    if closed.empty: return
    row = closed.iloc[-1]
    bar_key = f"{market_name}|{entry_tf}"
    stamp = str(closed.index[-1])
    manage_positions(market_name, frames, state)
    if state["last_processed"].get(bar_key) == stamp:
        return
    state["last_processed"][bar_key] = stamp
    if not state.get("running", True): return
    loss_pct = (float(state["day_start"])-float(state["balance"]))/max(float(state["day_start"]), 1)*100
    if loss_pct >= max_daily_loss_pct:
        if not state.get("daily_limit_logged", False):
            state["events"].append(f"Daily loss limit {max_daily_loss_pct:.1f}% reached. New paper entries blocked until the next day.")
            state["daily_limit_logged"] = True
        return
    if int(state["trades_today"]) >= max_trades_per_day:
        state["events"].append("Daily paper trade limit reached; no new entry.")
        return
    if any(p["Market"] == market_name for p in state["open_positions"]):
        return
    if model is None:
        state["events"].append(f"{market_name}: AI model needs more historical examples; no automatic entry.")
        return
    probs = {}
    for side in ("BUY", "SELL"):
        try:
            probs[side] = float(model.predict_proba(feature_row(row, side))[0][1])*100
        except Exception:
            probs[side] = 0.0
    side = max(probs, key=probs.get)
    probability = probs[side]
    biases = {tf: trend_bias(frames[tf]) if not frames[tf].empty else "Unknown" for tf in TIMEFRAMES}
    ok, reasons = signal_quality(row, side, biases)
    if probability <= probability_threshold:
        state["events"].append(f"{market_name}: no trade; best AI estimated win probability {probability:.1f}% (must be > {probability_threshold:.0f}%).")
        return
    if not ok:
        state["events"].append(f"{market_name}: AI probability {probability:.1f}% but filters blocked entry: " + "; ".join(reasons[:3]))
        return
    levels = make_levels(row, side)
    if not levels: return
    risk_cash = float(state["balance"])*risk_pct/100
    qty = risk_cash/max(levels["RiskPerUnit"], 1e-12)
    pos = {"Opened": datetime.now().isoformat(timespec="seconds"), "Market": market_name,
           "Symbol": symbol, "Timeframe": entry_tf, "Side": side,
           "AI_Win_Probability_pct": round(probability, 2), "Entry": levels["Entry"],
           "InitialStop": levels["InitialStop"], "Stop": levels["Stop"], "TP1": levels["TP1"],
           "TP2": levels["TP2"], "Qty": qty, "RiskBudget_virtual": risk_cash,
           "RiskPerUnit": levels["RiskPerUnit"], "BEBuffer": levels["BEBuffer"],
           "TP1_Hit": False, "RealizedPnL": 0.0, "RemainingPct": 100, "Status": "OPEN"}
    state["open_positions"].append(pos)
    state["trades_today"] = int(state["trades_today"])+1
    log_trade(pos)
    state["events"].append(f"AUTO PAPER {side} {market_name} @ {levels['Entry']:.5f}; AI estimate {probability:.1f}%; SL {levels['InitialStop']:.5f}, TP1 {levels['TP1']:.5f}, TP2 {levels['TP2']:.5f}.")


# ------------------------------- UI ---------------------------------
st.title("🧠 Brain AI Trading Bot V2")
st.caption("Mobile-friendly automatic paper trading · Simulated orders only · No broker/API or real-money execution")
if "bot_state" not in st.session_state:
    st.session_state.bot_state = load_state()
state = st.session_state.bot_state
# Automatic mode: no Start/Pause/reset buttons or bot-control sidebar.
state["running"] = True

market_name = st.selectbox("Market / symbol", list(MARKETS.keys()), index=0)
entry_tf = st.selectbox("Entry timeframe", TIMEFRAMES, index=0)
probability_threshold = 60.0
risk_pct = 0.5
max_daily_loss_pct = 3.0
max_trades_per_day = 20
symbol = MARKETS[market_name]["symbol"]

# Reset daily limits once date changes.
if state.get("day") != date.today().isoformat():
    state["day"] = date.today().isoformat(); state["day_start"] = float(state["balance"]); state["trades_today"] = 0; state["daily_limit_logged"] = False; state["running"] = True

# Load all four timeframes for selected market.
frames_raw, frames = {}, {}
with st.spinner(f"Loading {market_name} market data and training the AI Brain..."):
    for tf in TIMEFRAMES:
        frames_raw[tf] = download_data(symbol, tf)
        frames[tf] = add_indicators(frames_raw[tf], tf) if not frames_raw[tf].empty else pd.DataFrame()

entry_data = frames.get(entry_tf, pd.DataFrame())
model, model_stats = (None, {"samples": 0, "holdout_win_rate": None, "holdout_accuracy": None, "holdout_brier": None})
if not entry_data.empty and len(entry_data) >= 280:
    model, model_stats = train_brain_cached(symbol, entry_tf, str(entry_data.index[-1]), entry_data.iloc[:-1].copy())

# Auto refresh only while the Streamlit page is open. The app does not run in the background when closed.
refresh_count = st_autorefresh(interval=60_000, limit=None, key="brain_v2_refresh")
scan_and_trade(market_name, symbol, entry_tf, frames, model, model_stats, state,
               probability_threshold, risk_pct, max_daily_loss_pct, max_trades_per_day)
save_state()

# Account summary
loss_today = (float(state["day_start"])-float(state["balance"]))/max(float(state["day_start"]), 1)*100
m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("Virtual balance", f"₹{state['balance']:,.2f}")
m2.metric("Open paper positions", str(len(state["open_positions"])))
m3.metric("Paper trades today", f"{state['trades_today']}/{max_trades_per_day}")
m4.metric("Today's drawdown", f"{loss_today:.2f}%")
m5.metric("Auto mode", "ON" if loss_today < max_daily_loss_pct else "DAILY LIMIT")

st.subheader("AI Brain — win probability")
if model is None:
    st.warning(f"Model is not ready for {entry_tf}. Need more than 280 candles and enough positive/negative training examples. Current candles: {len(entry_data)}.")
else:
    if model_stats.get("holdout_win_rate") is not None:
        c1,c2,c3 = st.columns(3)
        c1.metric("Walk-forward holdout outcome rate", f"{model_stats['holdout_win_rate']:.1f}%")
        c2.metric("Holdout classification accuracy", f"{model_stats['holdout_accuracy']:.1f}%")
        c3.metric("Training samples", f"{model_stats['samples']:,}")
        st.caption("Holdout outcome rate is the proportion of positive labels in the later time-ordered test slice; it is not the bot's live win rate. AI probabilities are estimates and can be poorly calibrated in changing markets.")

if entry_data.empty or len(entry_data) < 230:
    st.error("Market data is unavailable or too short. Yahoo Finance may limit some intraday history; try another symbol/timeframe or refresh later.")
    st.info("Gold and US Oil use GC=F and CL=F futures as proxies; prices may differ from your broker's XAUUSD/USOIL CFD. Forex volume/VWAP/MFI can be incomplete because FX is decentralized.")
else:
    closed = entry_data.iloc[:-1]
    row = closed.iloc[-1]
    biases = {tf: trend_bias(frames[tf]) if not frames[tf].empty else "Unknown" for tf in TIMEFRAMES}
    probs = {"BUY": None, "SELL": None}
    if model is not None:
        for side in probs:
            probs[side] = float(model.predict_proba(feature_row(row, side))[0][1])*100
    best_side = max((s for s in probs if probs[s] is not None), key=lambda s: probs[s] if probs[s] is not None else -1) if model else "WAIT"
    best_prob = probs.get(best_side) if model else None
    ok, reasons = signal_quality(row, best_side, biases) if model and best_side in ("BUY", "SELL") else (False, ["AI model unavailable"])
    auto_ready = bool(model and best_prob is not None and best_prob > probability_threshold and ok)
    st.subheader(f"{market_name} · {entry_tf} entry analysis")
    q1,q2,q3,q4,q5 = st.columns(5)
    q1.metric("Last closed candle", f"{row.Close:,.5f}")
    q2.metric("AI BUY estimate", f"{probs['BUY']:.1f}%" if probs['BUY'] is not None else "n/a")
    q3.metric("AI SELL estimate", f"{probs['SELL']:.1f}%" if probs['SELL'] is not None else "n/a")
    q4.metric("Best side", best_side)
    q5.metric("Entry gate", "PASS" if auto_ready else "WAIT")
    bcols = st.columns(4)
    for col, tf in zip(bcols, ["1h", "30m", "15m", "5m"]):
        col.metric(f"{tf} trend", biases[tf])
    if best_side in ("BUY", "SELL"):
        levels = make_levels(row, best_side)
        if levels:
            p1,p2,p3,p4 = st.columns(4)
            p1.metric("Entry", f"{levels['Entry']:,.5f}")
            p2.metric("Stop loss", f"{levels['InitialStop']:,.5f}")
            p3.metric("Target 1 · close 60%", f"{levels['TP1']:,.5f}")
            p4.metric("Target 2 · close 40%", f"{levels['TP2']:,.5f}")
            st.caption(f"After TP1: book 60% and move the remaining stop to break-even + {levels['BEBuffer']:.5f} price units (0.05 ATR). Risk sizing is simplified and not contract/pip-value aware.")
        if auto_ready:
            st.success(f"Setup passes the current gate: {best_side}, AI estimate {best_prob:.1f}% > {probability_threshold:.0f}%, all required timeframe/indicator filters aligned.")
        else:
            st.warning("No automatic entry: probability threshold or one/more hard filters did not pass.")
        with st.expander("Why this setup is allowed / blocked"):
            st.write("Timeframe bias:", biases)
            if reasons:
                for reason in reasons: st.write("• " + reason)
            else: st.write("All configured hard filters passed.")

    st.subheader("Indicator chart")
    chart = closed.tail(180)
    fig = go.Figure(data=[go.Candlestick(x=chart.index, open=chart.Open, high=chart.High, low=chart.Low, close=chart.Close, name="Candles")])
    for col, label in [("EMA9","EMA 9"),("EMA21","EMA 21"),("EMA50","EMA 50"),("EMA200","EMA 200"),("BB_Upper","BB Upper"),("BB_Mid","BB Middle"),("BB_Lower","BB Lower"),("VWAP","VWAP"),("Support","Support"),("Resistance","Resistance"),("PowerUpper","Power Channel Upper"),("PowerMid","Power Channel Mid"),("PowerLower","Power Channel Lower")]:
        if col in chart:
            fig.add_trace(go.Scatter(x=chart.index, y=chart[col], mode="lines", name=label, connectgaps=False))
    for pos in state["open_positions"]:
        if pos["Market"] == market_name and pos["Timeframe"] == entry_tf:
            for key, label in [("Entry","Paper Entry"),("Stop","Paper Stop"),("TP1","TP1"),("TP2","TP2")]:
                fig.add_hline(y=float(pos[key]), line_dash="dash", annotation_text=label)
    fig.update_layout(height=620, xaxis_rangeslider_visible=False, legend_orientation="h", margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)
    cols = ["Open","High","Low","Close","Volume","EMA9","EMA21","EMA50","EMA200","BB_Lower","BB_Mid","BB_Upper","ATR","ADX","PLUS_DI","MINUS_DI","RSI","MFI","VWAP","RelVol","Support","Resistance","PowerLower","PowerMid","PowerUpper"]
    st.subheader("Recent indicator values")
    st.dataframe(closed[[c for c in cols if c in closed]].tail(20).iloc[::-1], use_container_width=True)

st.subheader("Open paper positions")
if state["open_positions"]:
    st.dataframe(pd.DataFrame(state["open_positions"]), use_container_width=True, hide_index=True)
else:
    st.info("No open paper positions.")

st.subheader("Recent paper-trading events")
if state.get("events"):
    for event in state["events"][-12:][::-1]: st.write("• " + event)
else:
    st.caption("No events yet. Automatic paper entries are enabled when AI probability is above 60% and all hard filters pass.")

if TRADE_LOG.exists():
    st.subheader("Paper trade history")
    try:
        history = pd.read_csv(TRADE_LOG)
        if not history.empty:
            closed_trades = history[history.get("Status", pd.Series(index=history.index, dtype=str)).eq("CLOSED")] if "Status" in history else pd.DataFrame()
            if not closed_trades.empty and "PnL_virtual" in closed_trades:
                wins = int((pd.to_numeric(closed_trades["PnL_virtual"], errors="coerce") > 0).sum())
                total = len(closed_trades)
                st.metric("Closed paper-trade win rate", f"{100*wins/total:.1f}%", help="Historical simulated results only; not a prediction or guarantee.")
            st.dataframe(history.iloc[::-1].head(100), use_container_width=True, hide_index=True)
            st.download_button("Download paper trade CSV", history.to_csv(index=False).encode("utf-8"), "brain_ai_v2_paper_trades.csv", "text/csv")
    except Exception as exc:
        st.warning(f"Could not read paper history: {exc}")

st.divider()
st.caption("Safety: paper-only, virtual INR balance, no broker credentials or live order route. This bot is a research prototype, not financial advice. Data vendor delays, missing volume, spread, slippage, commissions, swap/funding, and instrument contract sizes can materially change results. Keep the page open for auto scans; it does not run in the background when closed.")
