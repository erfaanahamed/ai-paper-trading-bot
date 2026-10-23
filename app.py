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

st.set_page_config(page_title="AI Paper Trading Brain", page_icon="🧠", layout="wide")

# -----------------------------
# Configuration / data helpers
# -----------------------------
STARTING_BALANCE = 100_000.0  # INR virtual balance
MARKETS = {
    "Gold (XAUUSD)": {"symbol": "GC=F", "signal_tf": "5m", "allowed_tfs": ["5m", "15m", "30m", "1h", "4h"], "htf": "4h", "kind": "gold"},
    "US Oil (WTI proxy)": {"symbol": "CL=F", "signal_tf": "15m", "allowed_tfs": ["15m", "30m", "1h", "4h"], "htf": "4h", "kind": "oil"},
    "EUR/USD": {"symbol": "EURUSD=X", "signal_tf": "15m", "allowed_tfs": ["15m", "30m", "1h", "4h"], "htf": "4h", "kind": "forex"},
    "GBP/USD": {"symbol": "GBPUSD=X", "signal_tf": "15m", "allowed_tfs": ["15m", "30m", "1h", "4h"], "htf": "4h", "kind": "forex"},
    "AUD/USD": {"symbol": "AUDUSD=X", "signal_tf": "15m", "allowed_tfs": ["15m", "30m", "1h", "4h"], "htf": "4h", "kind": "forex"},
    "USD/JPY": {"symbol": "JPY=X", "signal_tf": "15m", "allowed_tfs": ["15m", "30m", "1h", "4h"], "htf": "4h", "kind": "forex"},
    "BTC/USD": {"symbol": "BTC-USD", "signal_tf": "15m", "allowed_tfs": ["15m", "30m", "1h", "4h"], "htf": "4h", "kind": "crypto"},
    "SOL/USD": {"symbol": "SOL-USD", "signal_tf": "15m", "allowed_tfs": ["15m", "30m", "1h", "4h"], "htf": "4h", "kind": "crypto"},
    "ETH/USD": {"symbol": "ETH-USD", "signal_tf": "15m", "allowed_tfs": ["15m", "30m", "1h", "4h"], "htf": "4h", "kind": "crypto"},
}
INTERVAL_MAP = {"5m": ("5m", "5d"), "15m": ("15m", "60d"), "30m": ("30m", "60d"), "1h": ("60m", "730d"), "4h": ("60m", "730d")}


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
        if interval == "4h":
            # Yahoo Finance commonly exposes hourly rather than 4-hour bars; aggregate them.
            df = df.resample("4h").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum"
            }).dropna(subset=["Open", "High", "Low", "Close"])
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
    return {"5m": 48, "15m": 40, "30m": 36, "1h": 30, "4h": 24}.get(interval, 40)

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

def add_smc_features(df):
    """Rule-based SMC approximations. These are price-action proxies, not verified institutional order data."""
    if df.empty:
        return df
    x = df.copy()
    # Confirmed swing levels are shifted so current candle does not define its own reference.
    swing_n = 3
    x["SwingHigh"] = x["High"].rolling(2*swing_n+1, center=True).max().where(
        x["High"].eq(x["High"].rolling(2*swing_n+1, center=True).max())).shift(swing_n)
    x["SwingLow"] = x["Low"].rolling(2*swing_n+1, center=True).min().where(
        x["Low"].eq(x["Low"].rolling(2*swing_n+1, center=True).min())).shift(swing_n)
    prev_swing_high = x["SwingHigh"].ffill().shift(1)
    prev_swing_low = x["SwingLow"].ffill().shift(1)
    x["BOS_Bull"] = (x["Close"] > prev_swing_high) & prev_swing_high.notna()
    x["BOS_Bear"] = (x["Close"] < prev_swing_low) & prev_swing_low.notna()
    # CHOCH proxy: close crosses a swing level against the prior candle's local direction.
    x["CHOCH_Bull"] = x["BOS_Bull"] & (x["Close"].shift(1) <= prev_swing_high)
    x["CHOCH_Bear"] = x["BOS_Bear"] & (x["Close"].shift(1) >= prev_swing_low)
    # Three-candle fair value gap proxy.
    x["FVG_Bull"] = x["Low"] > x["High"].shift(2)
    x["FVG_Bear"] = x["High"] < x["Low"].shift(2)
    x["FVG_Bull_Lower"] = x["High"].shift(2)
    x["FVG_Bull_Upper"] = x["Low"]
    x["FVG_Bear_Lower"] = x["High"]
    x["FVG_Bear_Upper"] = x["Low"].shift(2)
    # Liquidity sweep: wick crosses a prior swing and close returns inside.
    x["SweepLow"] = (x["Low"] < prev_swing_low) & (x["Close"] > prev_swing_low) & prev_swing_low.notna()
    x["SweepHigh"] = (x["High"] > prev_swing_high) & (x["Close"] < prev_swing_high) & prev_swing_high.notna()
    # Order-block proxy: last opposite candle before a structure break, carried forward briefly.
    bull_ob_low = pd.Series(np.nan, index=x.index); bull_ob_high = pd.Series(np.nan, index=x.index)
    bear_ob_low = pd.Series(np.nan, index=x.index); bear_ob_high = pd.Series(np.nan, index=x.index)
    for i in range(2, len(x)):
        if bool(x["BOS_Bull"].iloc[i]):
            candidates = [j for j in range(max(0, i-5), i) if x["Close"].iloc[j] < x["Open"].iloc[j]]
            if candidates:
                j = candidates[-1]; bull_ob_low.iloc[i] = x["Low"].iloc[j]; bull_ob_high.iloc[i] = x["High"].iloc[j]
        if bool(x["BOS_Bear"].iloc[i]):
            candidates = [j for j in range(max(0, i-5), i) if x["Close"].iloc[j] > x["Open"].iloc[j]]
            if candidates:
                j = candidates[-1]; bear_ob_low.iloc[i] = x["Low"].iloc[j]; bear_ob_high.iloc[i] = x["High"].iloc[j]
    x["BullOB_Low"] = bull_ob_low.ffill(limit=20); x["BullOB_High"] = bull_ob_high.ffill(limit=20)
    x["BearOB_Low"] = bear_ob_low.ffill(limit=20); x["BearOB_High"] = bear_ob_high.ffill(limit=20)
    x["InBullOB"] = x["Close"].between(x["BullOB_Low"], x["BullOB_High"])
    x["InBearOB"] = x["Close"].between(x["BearOB_Low"], x["BearOB_High"])
    # Premium/discount based on prior rolling dealing range.
    range_high = x["High"].rolling(50, min_periods=20).max().shift(1)
    range_low = x["Low"].rolling(50, min_periods=20).min().shift(1)
    x["DealingMid"] = (range_high + range_low) / 2
    x["Discount"] = x["Close"] < x["DealingMid"]
    x["Premium"] = x["Close"] > x["DealingMid"]
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

def score_signal(last, bias, threshold=78, use_smc=False):
    """Rule-fit score using standard indicators and price-action filters; not a probability."""
    def val(key, default=False):
        v = last.get(key, default)
        return bool(v) if pd.notna(v) else False
    long_checks = {
        "RSI oversold (<25)": last["RSI"] < 25,
        "MFI oversold (<20)": pd.notna(last.get("MFI", np.nan)) and last["MFI"] < 20,
        "Close below lower Bollinger Band": last["Close"] < last["BB_Lower"],
        "Relative volume >= 1.5": pd.notna(last["RelVol"]) and last["RelVol"] >= 1.5,
        "ADX range filter (<25)": pd.notna(last["ADX"]) and last["ADX"] < 25,
        "HTF bullish": bias == "Bullish",
        "EMA9 > EMA21": last["EMA9"] > last["EMA21"],
        "+DI > -DI": pd.notna(last.get("PLUS_DI", np.nan)) and pd.notna(last.get("MINUS_DI", np.nan)) and last["PLUS_DI"] > last["MINUS_DI"],
        "Below VWAP": last["Close"] < last["VWAP"],
        "Near support": pd.notna(last.get("Support", np.nan)) and last["Close"] <= last["Support"] + 0.35 * last["ATR"],
        "Bullish reversal candle": val("BullReversal"),
    }
    short_checks = {
        "RSI overbought (>75)": last["RSI"] > 75,
        "MFI overbought (>80)": pd.notna(last.get("MFI", np.nan)) and last["MFI"] > 80,
        "Close above upper Bollinger Band": last["Close"] > last["BB_Upper"],
        "Relative volume >= 1.5": pd.notna(last["RelVol"]) and last["RelVol"] >= 1.5,
        "ADX range filter (<25)": pd.notna(last["ADX"]) and last["ADX"] < 25,
        "HTF bearish": bias == "Bearish",
        "EMA9 < EMA21": last["EMA9"] < last["EMA21"],
        "-DI > +DI": pd.notna(last.get("PLUS_DI", np.nan)) and pd.notna(last.get("MINUS_DI", np.nan)) and last["MINUS_DI"] > last["PLUS_DI"],
        "Above VWAP": last["Close"] > last["VWAP"],
        "Near resistance": pd.notna(last.get("Resistance", np.nan)) and last["Close"] >= last["Resistance"] - 0.35 * last["ATR"],
        "Bearish reversal candle": val("BearReversal"),
    }
    # Category-weighted score, capped at 100; normalize active categories for fair A/B comparison.
    weights = {
        "RSI oversold (<25)": 5, "MFI oversold (<20)": 5, "Close below lower Bollinger Band": 8,
        "Relative volume >= 1.5": 5, "ADX range filter (<25)": 5, "HTF bullish": 10,
        "EMA9 > EMA21": 5, "+DI > -DI": 5, "Below VWAP": 4, "Near support": 7,
        "Bullish reversal candle": 5, "Low liquidity sweep": 8,
        "Discount zone": 8,
        "RSI overbought (>75)": 5, "MFI overbought (>80)": 5, "Close above upper Bollinger Band": 8,
        "HTF bearish": 10, "EMA9 < EMA21": 5, "-DI > +DI": 5, "Above VWAP": 4, "Near resistance": 7,
        "Bearish reversal candle": 5, "High liquidity sweep": 8,
        "Premium zone": 8,
    }
    active_long_max = sum(weights[k] for k in long_checks) or 1
    active_short_max = sum(weights[k] for k in short_checks) or 1
    long_score = min(100, round(100 * sum(weights[k] for k,v in long_checks.items() if v) / active_long_max))
    short_score = min(100, round(100 * sum(weights[k] for k,v in short_checks.items() if v) / active_short_max))
    long_hard = bias != "Bullish" or not (last["ADX"] < 25) or not (last["Close"] < last["BB_Lower"])
    short_hard = bias != "Bearish" or not (last["ADX"] < 25) or not (last["Close"] > last["BB_Upper"])
    if not long_hard and long_score >= threshold:
        return "BUY", int(long_score), long_checks
    if not short_hard and short_score >= threshold:
        return "SELL", int(short_score), short_checks
    return "WAIT", int(max(long_score, short_score)), long_checks if long_score >= short_score else short_checks



@st.cache_data(ttl=55, show_spinner=False)
def scan_all_market_signals():
    """Fetch and evaluate each configured market for the overview dashboard.
    This reports signals only; it does not open paper positions for every market.
    """
    rows = []
    for name, config in MARKETS.items():
        tf = config["signal_tf"]
        try:
            raw_market = download_data(config["symbol"], tf)
            htf_market = download_data(config["symbol"], config["htf"])
            if raw_market.empty or len(raw_market) < 60:
                rows.append({
                    "Market": name, "Symbol": config["symbol"], "TF": tf,
                    "Last close": None, "HTF bias": "Unknown",
                    "Signal": "DATA UNAVAILABLE", "Rule-fit": None,
                    "RSI": None, "ADX": None, "Data status": "Insufficient candles or provider returned no data"
                })
                continue
            market_data = add_adaptive_levels(add_indicators(raw_market), tf)
            htf_processed = add_adaptive_levels(add_indicators(htf_market), config["htf"]) if not htf_market.empty else htf_market
            bias_value, _ = htf_bias(htf_processed) if not htf_processed.empty else ("Unknown", None)
            last_row = market_data.iloc[-1]
            best_side, best_score, _ = score_signal(last_row, bias_value, threshold=78)
            signal_value = best_side if best_side in ("BUY", "SELL") else "WAIT"
            rows.append({
                "Market": name, "Symbol": config["symbol"], "TF": tf,
                "Last close": round(float(last_row["Close"]), 5),
                "HTF bias": bias_value, "Signal": signal_value,
                "Rule-fit": int(best_score),
                "RSI": round(float(last_row["RSI"]), 1) if pd.notna(last_row["RSI"]) else None,
                "ADX": round(float(last_row["ADX"]), 1) if pd.notna(last_row["ADX"]) else None,
                "Data status": "OK"
            })
        except Exception as exc:
            rows.append({
                "Market": name, "Symbol": config["symbol"], "TF": tf,
                "Last close": None, "HTF bias": "Unknown",
                "Signal": "DATA ERROR", "Rule-fit": None,
                "RSI": None, "ADX": None, "Data status": str(exc)[:120]
            })
    return pd.DataFrame(rows)


def make_trade_levels(last, side, risk_pct, rr2):
    entry = float(last["Close"])
    atr = float(last["ATR"])
    if not np.isfinite(atr) or atr <= 0:
        return None
    # Protective stop uses signal candle + ATR buffer; choose wider of the two.
    if side == "BUY":
        stop = min(float(last["Low"]), entry - 1.2 * atr)
        risk = entry - stop
        tp1 = min(float(last["BB_Mid"]), entry + 0.9 * risk) if pd.notna(last["BB_Mid"]) else entry + 0.9*risk
        tp2_candidates = [entry + rr2*risk]
        if pd.notna(last["BB_Upper"]):
            tp2_candidates.append(float(last["BB_Upper"]))
        if pd.notna(last["Resistance"]):
            tp2_candidates.append(float(last["Resistance"]))
        tp2 = min([v for v in tp2_candidates if v > entry] or [entry + rr2*risk])
    else:
        stop = max(float(last["High"]), entry + 1.2 * atr)
        risk = stop - entry
        tp1 = max(float(last["BB_Mid"]), entry - 0.9 * risk) if pd.notna(last["BB_Mid"]) else entry - 0.9*risk
        tp2_candidates = [entry - rr2*risk]
        if pd.notna(last["BB_Lower"]):
            tp2_candidates.append(float(last["BB_Lower"]))
        if pd.notna(last["Support"]):
            tp2_candidates.append(float(last["Support"]))
        tp2 = max([v for v in tp2_candidates if v < entry] or [entry - rr2*risk])
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
                "PnL (estimated price-units)": pnl, "Status": "CLOSED"
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
                "PnL (estimated price-units)": pnl, "Status": "CLOSED"
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

    side, score, checks = score_signal(last, bias, threshold)
    if side in ("BUY", "SELL"):
        levels = make_trade_levels(last, side, risk_pct, rr2)
        if levels:
            entry = float(last["Close"])
            stop = float(levels["stop"])
            # Re-anchor levels to the signal close for the simulator.
            risk_unit = abs(entry - stop)
            if risk_unit > 0:
                risk_cash = balance * risk_pct / 100
                qty = risk_cash / risk_unit
                if side == "BUY":
                    tp1 = min(float(last["BB_Mid"]), entry + 0.9*risk_unit) if pd.notna(last["BB_Mid"]) else entry + 0.9*risk_unit
                    tp2_candidates = [entry + rr2*risk_unit]
                    if pd.notna(last.get("BB_Upper", np.nan)): tp2_candidates.append(float(last["BB_Upper"]))
                    if pd.notna(last.get("Resistance", np.nan)): tp2_candidates.append(float(last["Resistance"]))
                    tp2 = min([v for v in tp2_candidates if v > entry] or [entry + rr2*risk_unit])
                else:
                    tp1 = max(float(last["BB_Mid"]), entry - 0.9*risk_unit) if pd.notna(last["BB_Mid"]) else entry - 0.9*risk_unit
                    tp2_candidates = [entry - rr2*risk_unit]
                    if pd.notna(last.get("BB_Lower", np.nan)): tp2_candidates.append(float(last["BB_Lower"]))
                    if pd.notna(last.get("Support", np.nan)): tp2_candidates.append(float(last["Support"]))
                    tp2 = max([v for v in tp2_candidates if v < entry] or [entry - rr2*risk_unit])
                pos = {
                    "Opened": datetime.now().isoformat(timespec="seconds"), "Market": market_name,
                    "Symbol": symbol, "Timeframe": timeframe, "Side": side, "Score": score,
                    "Entry": entry, "Stop": stop, "InitialStop": stop, "TP1": tp1, "TP2": tp2,
                    "Qty": qty, "RiskBudget": risk_cash, "TP1_Hit": False, "RealizedPnL": 0.0,
                    "Status": "OPEN"
                }
                open_positions.append(pos)
                trades_today += 1
                events.append(f"Automatically opened paper {side} on {market_name} at {entry:.5f} (score {score}/100).")
            else:
                events.append("Signal rejected: stop distance was zero.")
    else:
        events.append(f"No entry: setup did not pass the hard filters (rule-fit score {score}/100).")

    processed[market_key] = bar_time
    st.session_state.auto_last_processed = processed
    return balance, trades_today, daily_start_balance, open_positions, events, "Scan complete."

def run_backtest(df, htf, timeframe, htf_timeframe, threshold, risk_pct, rr2, starting_balance, max_daily_loss_pct, cost_bps=2.0, use_smc=True):
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
        side, score, _ = score_signal(row, bias, threshold, use_smc=use_smc)
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
st.caption("Rule-based confidence scoring + multi-timeframe filters + automatic simulated trade entry/exit. Paper trading only; no broker, wallet, or live-order connection.")
with st.sidebar:
    st.header("Bot settings")
    market_name = st.selectbox("Market", list(MARKETS.keys()))
    cfg = MARKETS[market_name]
    allowed_tfs = cfg["allowed_tfs"]
    default_tf = cfg["signal_tf"]
    signal_tf = st.selectbox("Signal timeframe", allowed_tfs, index=allowed_tfs.index(default_tf))
    htf_tf = st.selectbox("Higher-timeframe filter", ["1h","4h"], index=1)
    score_threshold = st.slider("Minimum rule-fit score", 50, 95, 78, 1)
    risk_pct = st.slider("Risk per trade (%)", 0.25, 1.0, 0.5, 0.25)
    max_daily_loss = st.slider("Daily loss stop (%)", 1.0, 3.0, 2.0, 0.5)
    max_trades_per_day = st.slider("Maximum paper trades per day", 1, 30, 20, 1)
    rr2 = st.slider("TP2 risk/reward", 1.5, 2.5, 2.0, 0.1)
    st.caption("Score is not a win probability. Test before relying on it.")

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
st.caption("Scans each configured market using its default signal timeframe. Results are cached for about 55 seconds to limit provider requests. This dashboard displays signals only; automatic paper-position management below still applies to the selected market.")
with st.spinner("Scanning configured markets..."):
    all_market_rows = scan_all_market_signals()
if not all_market_rows.empty:
    st.dataframe(all_market_rows, use_container_width=True, hide_index=True)
    ready_count = int(all_market_rows["Data status"].eq("OK").sum())
    st.caption(f"Markets with usable data: {ready_count}/{len(MARKETS)}. Yahoo Finance may not provide every symbol/timeframe reliably; unavailable data is shown explicitly.")
else:
    st.warning("No market scan results are available yet.")

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
side, score, checks = score_signal(last, bias, score_threshold)
daily_loss_reached = ((st.session_state.daily_start - st.session_state.balance) / max(st.session_state.daily_start, 1) * 100) >= max_daily_loss
if st.session_state.trades_today >= max_trades_per_day:
    side = "WAIT"
    checks = {"Daily trade limit reached": True}
elif daily_loss_reached:
    st.session_state.halted = True
    side = "WAIT"
    checks = {"Daily loss limit reached": True}
st.markdown(f"**Selected market:** {market_name} · **Timeframe:** {signal_tf} · **Signal**")
m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("Last candle close", f"{float(last['Close']):,.5f}")
m2.metric("HTF bias", bias)
m3.metric("AI rule-fit score", f"{score}/100")
m4.metric("RSI / MFI", f"{last['RSI']:.1f} / {last['MFI']:.1f}" if pd.notna(last["MFI"]) else f"{last['RSI']:.1f} / n/a")
m5.metric("ADX / Rel. volume", f"{last['ADX']:.1f} / {last['RelVol']:.2f}x" if pd.notna(last["RelVol"]) else f"{last['ADX']:.1f} / n/a")

if st.session_state.halted:
    st.error("Paper bot is halted by the daily loss control. It will reset on the next calendar day.")
elif st.session_state.trades_today >= max_trades_per_day:
    st.error(f"Daily trade limit reached ({max_trades_per_day}). No more paper trades will be allowed today.")
elif side in ("BUY","SELL"):
    st.success(f"Potential {side} setup — score {score}/100. This is a paper signal, not a guaranteed trade.")
else:
    st.warning(f"WAIT — no setup passed all hard filters. Best-side rule-fit score: {score}/100.")

st.subheader("Automatic paper trading")
st.caption("Paper trading only. Automatic scanning is enabled by default and checks the selected market about every 60 seconds while this page stays open and connected. Cloud sleep, browser disconnection, or provider delays can interrupt scans.")
st.success("Automatic paper scanner: RUNNING")


def run_selected_market_scan():
    (st.session_state.balance, st.session_state.trades_today,
     st.session_state.auto_daily_start_balance, st.session_state.auto_open_positions,
     auto_events, auto_message) = simulate_auto_paper_scan(
        data, market_name, cfg["symbol"], signal_tf, bias, score_threshold,
        risk_pct, rr2, st.session_state.balance, st.session_state.trades_today,
        max_trades_per_day, st.session_state.auto_daily_start_balance,
        max_daily_loss, st.session_state.auto_open_positions
    )
    st.session_state.last_auto_events = auto_events
    st.session_state.last_auto_message = auto_message

# Run once for each 60-second timer tick while the scanner is armed.
if st.session_state.auto_bot_status == "RUNNING" and auto_refresh_count > st.session_state.auto_last_refresh_count:
    st.session_state.auto_last_refresh_count = auto_refresh_count
    run_selected_market_scan()
    st.rerun()
else:
    # Keep the counter in sync while paused/stopped so restarting doesn't replay old ticks.
    st.session_state.auto_last_refresh_count = auto_refresh_count

if st.button("🔎 Scan market & manage paper positions now", type="primary"):
    run_selected_market_scan()
    st.rerun()

if st.session_state.get("last_auto_message"):
    st.info(st.session_state.last_auto_message)
for ev in st.session_state.get("last_auto_events", []):
    st.write("• " + ev)

st.markdown("**Open automatic paper positions**")
if st.session_state.auto_open_positions:
    st.dataframe(pd.DataFrame(st.session_state.auto_open_positions), use_container_width=True)
else:
    st.info("No automatic paper positions are currently open.")

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
    st.write("The score is a weighted rule-fit score, not a calibrated probability of winning. Hard filters reject a setup if the HTF bias disagrees, ADX is not ranging, or price has not closed outside the relevant Bollinger Band.")
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
        if err:
            st.error(err)
        elif bt.empty:
            st.warning("No trades passed the filters in the loaded history.")
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
