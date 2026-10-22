import json
import os
from datetime import datetime, timezone, date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

APP_DIR = Path(__file__).parent
STATE_FILE = APP_DIR / "paper_state.json"
STARTING_BALANCE = 100_000.0
USD_INR_ASSUMPTION = 85.0  # Approximation for simulation only; editable in sidebar.
RISK_PER_TRADE_PCT = 0.5
MAX_DAILY_LOSS_PCT = 3.0
TIMEFRAMES = {"15 minutes": "15m", "30 minutes": "30m", "1 hour": "60m"}
MARKETS = {
    "Gold": {"XAUUSD=X": "Gold spot proxy (XAUUSD=X)"},
    "Forex": {
        "EURUSD=X": "EUR/USD",
        "GBPUSD=X": "GBP/USD",
        "USDJPY=X": "USD/JPY",
        "AUDUSD=X": "AUD/USD",
    },
    "Crypto": {
        "BTC-USD": "Bitcoin (BTC-USD)",
        "ETH-USD": "Ethereum (ETH-USD)",
        "SOL-USD": "Solana (SOL-USD)",
    },
}

st.set_page_config(page_title="Multi-Market Paper Trading Bot", page_icon="📈", layout="wide")


def load_state():
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            state.setdefault("balance", STARTING_BALANCE)
            state.setdefault("open_trades", {})
            state.setdefault("history", [])
            state.setdefault("last_entry_bar", {})
            state.setdefault("day_start_balance", state["balance"])
            state.setdefault("day", str(date.today()))
            return state
        except Exception:
            pass
    return {
        "balance": STARTING_BALANCE,
        "open_trades": {},
        "history": [],
        "last_entry_bar": {},
        "day_start_balance": STARTING_BALANCE,
        "day": str(date.today()),
    }


def save_state(state):
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.replace(STATE_FILE)


def fetch_bars(symbol, interval, period="30d"):
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in needed):
        return pd.DataFrame()
    df = df[needed].dropna(subset=["Open", "High", "Low", "Close"])
    return df


def add_indicators(df):
    if df.empty:
        return df
    x = df.copy()
    close, high, low, volume = x["Close"], x["High"], x["Low"], x["Volume"].fillna(0)
    for span in (9, 21, 50, 200):
        x[f"EMA{span}"] = close.ewm(span=span, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    x["RSI"] = 100 - (100 / (1 + rs))
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    x["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=x.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=x.index)
    atr_safe = x["ATR"].replace(0, np.nan)
    x["+DI"] = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_safe
    x["-DI"] = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_safe
    dx = 100 * (x["+DI"] - x["-DI"]).abs() / (x["+DI"] + x["-DI"]).replace(0, np.nan)
    x["ADX"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    # Session VWAP approximation: reset by date where the data index has a timezone/session date.
    typical = (high + low + close) / 3
    idx = x.index
    try:
        session_key = pd.Index(idx.date)
    except Exception:
        session_key = pd.Index(pd.to_datetime(idx).date)
    pv = typical * volume
    x["VWAP"] = pv.groupby(session_key).cumsum() / volume.groupby(session_key).cumsum().replace(0, np.nan)
    # Adaptive channel based on a rolling 20-bar mean and standard deviation.
    x["ChannelMid"] = close.rolling(20, min_periods=10).mean()
    x["ChannelWidth"] = close.rolling(20, min_periods=10).std() * 2
    x["ChannelUpper"] = x["ChannelMid"] + x["ChannelWidth"]
    x["ChannelLower"] = x["ChannelMid"] - x["ChannelWidth"]
    # Rolling support/resistance and simple swing structure proxies.
    x["Support"] = low.rolling(20, min_periods=5).min()
    x["Resistance"] = high.rolling(20, min_periods=5).max()
    x["PrevSwingHigh"] = high.shift(1).rolling(10, min_periods=5).max()
    x["PrevSwingLow"] = low.shift(1).rolling(10, min_periods=5).min()
    return x


def score_timeframe(df):
    if df.empty or len(df) < 205:
        return {"score": 0, "label": "INSUFFICIENT DATA", "row": None}
    r = df.iloc[-1]
    prev = df.iloc[-2]
    vals = [r.get("EMA9"), r.get("EMA21"), r.get("EMA50"), r.get("EMA200"), r.get("RSI"), r.get("ATR"), r.get("ADX")]
    if any(pd.isna(v) for v in vals):
        return {"score": 0, "label": "INSUFFICIENT DATA", "row": r}
    score = 0
    score += 1 if r["EMA9"] > r["EMA21"] else -1
    score += 1 if r["EMA21"] > r["EMA50"] else -1
    score += 1 if r["EMA50"] > r["EMA200"] else -1
    score += 1 if r["Close"] > r["VWAP"] else -1 if pd.notna(r["VWAP"]) else 0
    score += 1 if r["+DI"] > r["-DI"] else -1
    if r["ADX"] >= 18:
        score += 1 if r["+DI"] > r["-DI"] else -1
    # RSI confirmation without chasing extreme readings.
    if 52 <= r["RSI"] <= 70:
        score += 1
    elif 30 <= r["RSI"] <= 48:
        score -= 1
    # Structure breakout proxy: close breaks previous rolling swing level.
    if pd.notna(r["PrevSwingHigh"]) and r["Close"] > r["PrevSwingHigh"]:
        score += 1
    elif pd.notna(r["PrevSwingLow"]) and r["Close"] < r["PrevSwingLow"]:
        score -= 1
    label = "BUY" if score >= 4 else "SELL" if score <= -4 else "NO TRADE"
    return {"score": int(score), "label": label, "row": r}


def make_signal(frames):
    scored = {tf: score_timeframe(df) for tf, df in frames.items()}
    labels = [v["label"] for v in scored.values()]
    # Require all three timeframes to agree to avoid weak/mixed entries.
    if labels == ["BUY", "BUY", "BUY"]:
        direction = "BUY"
    elif labels == ["SELL", "SELL", "SELL"]:
        direction = "SELL"
    else:
        direction = "NO TRADE"
    base = scored["15 minutes"]["row"]
    if direction == "NO TRADE" or base is None:
        return direction, scored, None
    price = float(base["Close"])
    atr = float(base["ATR"])
    if not np.isfinite(price) or not np.isfinite(atr) or atr <= 0:
        return "NO TRADE", scored, None
    stop_dist = max(1.5 * atr, price * 0.0005)
    if direction == "BUY":
        sl, t1, t2 = price - stop_dist, price + stop_dist, price + 2 * stop_dist
    else:
        sl, t1, t2 = price + stop_dist, price - stop_dist, price - 2 * stop_dist
    # Avoid entering too close to an obvious rolling S/R level; this is a heuristic.
    support, resistance = base.get("Support"), base.get("Resistance")
    if direction == "BUY" and pd.notna(resistance) and (float(resistance) - price) < 0.5 * stop_dist:
        return "NO TRADE", scored, None
    if direction == "SELL" and pd.notna(support) and (price - float(support)) < 0.5 * stop_dist:
        return "NO TRADE", scored, None
    return direction, scored, {"entry": price, "sl": float(sl), "t1": float(t1), "t2": float(t2), "atr": atr}


def close_trade(state, symbol, trade, exit_price, reason, usd_inr):
    remaining_qty = float(trade.get("remaining_qty", trade["qty"]))
    pnl_quote = (exit_price - trade["entry"]) * remaining_qty
    if trade["side"] == "SELL":
        pnl_quote = -pnl_quote
    pnl_inr = pnl_quote * usd_inr
    state["balance"] += pnl_inr
    state["history"].append({
        "closed_at": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "side": trade["side"],
        "entry": trade["entry"],
        "exit": float(exit_price),
        "qty_remaining": remaining_qty,
        "pnl_inr": round(pnl_inr, 2),
        "reason": reason,
    })
    state["open_trades"].pop(symbol, None)


def update_paper_trade(state, symbol, bar, usd_inr):
    trade = state["open_trades"].get(symbol)
    if not trade:
        return
    high, low = float(bar["High"]), float(bar["Low"])
    side = trade["side"]
    # Conservative assumption: if SL and TP are both touched in one candle, assume SL first.
    if side == "BUY":
        if low <= trade["sl"]:
            close_trade(state, symbol, trade, trade["sl"], "STOP LOSS", usd_inr)
            return
        if not trade.get("t1_hit") and high >= trade["t1"]:
            qty_close = trade["qty"] * 0.5
            pnl = (trade["t1"] - trade["entry"]) * qty_close * usd_inr
            state["balance"] += pnl
            trade["remaining_qty"] = trade["qty"] - qty_close
            trade["t1_hit"] = True
            state["history"].append({"closed_at": datetime.now(timezone.utc).isoformat(), "symbol": symbol, "side": side, "entry": trade["entry"], "exit": trade["t1"], "qty_remaining": qty_close, "pnl_inr": round(pnl, 2), "reason": "TARGET 1 (PARTIAL)"})
        if symbol in state["open_trades"] and high >= trade["t2"]:
            close_trade(state, symbol, trade, trade["t2"], "TARGET 2", usd_inr)
    else:
        if high >= trade["sl"]:
            close_trade(state, symbol, trade, trade["sl"], "STOP LOSS", usd_inr)
            return
        if not trade.get("t1_hit") and low <= trade["t1"]:
            qty_close = trade["qty"] * 0.5
            pnl = (trade["entry"] - trade["t1"]) * qty_close * usd_inr
            state["balance"] += pnl
            trade["remaining_qty"] = trade["qty"] - qty_close
            trade["t1_hit"] = True
            state["history"].append({"closed_at": datetime.now(timezone.utc).isoformat(), "symbol": symbol, "side": side, "entry": trade["entry"], "exit": trade["t1"], "qty_remaining": qty_close, "pnl_inr": round(pnl, 2), "reason": "TARGET 1 (PARTIAL)"})
        if symbol in state["open_trades"] and low <= trade["t2"]:
            close_trade(state, symbol, trade, trade["t2"], "TARGET 2", usd_inr)


def open_paper_trade(state, symbol, direction, levels, usd_inr):
    risk_inr = max(0.0, state["balance"] * RISK_PER_TRADE_PCT / 100)
    stop_distance = abs(levels["entry"] - levels["sl"])
    if risk_inr <= 0 or stop_distance <= 0:
        return False
    qty = risk_inr / (stop_distance * usd_inr)
    if not np.isfinite(qty) or qty <= 0:
        return False
    state["open_trades"][symbol] = {
        "side": direction,
        "entry": levels["entry"],
        "sl": levels["sl"],
        "t1": levels["t1"],
        "t2": levels["t2"],
        "qty": qty,
        "remaining_qty": qty,
        "t1_hit": False,
        "opened_at": datetime.now(timezone.utc).isoformat(),
    }
    return True


st.title("📈 Multi-Market AI-Assisted Paper Trading Bot")
st.caption("Mobile-friendly research prototype • Virtual money only • No broker orders or wallet connection")

with st.sidebar:
    st.header("Settings")
    usd_inr = st.number_input("USD → INR simulation rate", min_value=1.0, max_value=200.0, value=USD_INR_ASSUMPTION, step=0.5, help="Approximation used only to translate simulated quote-currency P/L into INR. Not a live FX conversion.")
    refresh_seconds = st.select_slider("Refresh while page is open", options=[30, 60, 120, 300], value=60, format_func=lambda x: f"{x} seconds")
    risk_pct = st.select_slider("Risk per new trade", options=[0.25, 0.5, 0.75, 1.0], value=RISK_PER_TRADE_PCT, format_func=lambda x: f"{x}% of balance")
    st.markdown("**Starting balance:** ₹1,00,000")
    st.markdown("**Timeframes:** 15m · 30m · 1h")
    reset = st.button("Reset virtual account", type="secondary")

if reset:
    state = {"balance": STARTING_BALANCE, "open_trades": {}, "history": [], "last_entry_bar": {}, "day_start_balance": STARTING_BALANCE, "day": str(date.today())}
    save_state(state)
    st.success("Virtual account reset to ₹1,00,000.")
    st.rerun()

st_autorefresh(interval=refresh_seconds * 1000, key="auto_refresh")
state = load_state()
if state.get("day") != str(date.today()):
    state["day"] = str(date.today())
    state["day_start_balance"] = state["balance"]

# Apply selected risk for new positions.
RISK_PER_TRADE_PCT = risk_pct

market_tabs = st.tabs(list(MARKETS.keys()))
all_signal_rows = []
all_frames = {}
fetch_errors = []

for tab, (market_name, symbols) in zip(market_tabs, MARKETS.items()):
    with tab:
        for symbol, friendly_name in symbols.items():
            with st.expander(f"{friendly_name} · {symbol}", expanded=(symbol == "XAUUSD=X")):
                frames = {}
                for tf_name, interval in TIMEFRAMES.items():
                    try:
                        raw = fetch_bars(symbol, interval)
                        frames[tf_name] = add_indicators(raw)
                    except Exception as exc:
                        frames[tf_name] = pd.DataFrame()
                        fetch_errors.append(f"{symbol} {tf_name}: {exc}")
                all_frames[symbol] = frames
                direction, scored, levels = make_signal(frames)
                latest = frames["15 minutes"].iloc[-1] if not frames["15 minutes"].empty else None
                if latest is not None:
                    update_paper_trade(state, symbol, latest, usd_inr)
                # Enforce daily loss guardrail for new entries.
                day_start = float(state.get("day_start_balance", STARTING_BALANCE))
                daily_loss_pct = max(0.0, (day_start - float(state["balance"])) / max(day_start, 1.0) * 100)
                can_open = daily_loss_pct < MAX_DAILY_LOSS_PCT
                current_bar_id = str(latest.name) if latest is not None else ""
                last_entry_bar = state.setdefault("last_entry_bar", {}).get(symbol)
                if direction in ("BUY", "SELL") and levels and symbol not in state["open_trades"] and can_open and current_bar_id != last_entry_bar:
                    if open_paper_trade(state, symbol, direction, levels, usd_inr):
                        state["last_entry_bar"][symbol] = current_bar_id
                        state["history"].append({"closed_at": datetime.now(timezone.utc).isoformat(), "symbol": symbol, "side": direction, "entry": levels["entry"], "exit": None, "qty_remaining": state["open_trades"][symbol]["qty"], "pnl_inr": 0.0, "reason": "PAPER ENTRY"})
                elif direction in ("BUY", "SELL") and not can_open:
                    st.warning("Daily loss guardrail reached: no new paper trades will be opened today.")
                st.metric("Current signal", direction)
                c1, c2, c3 = st.columns(3)
                if levels:
                    c1.metric("Entry", f"{levels['entry']:.5f}")
                    c2.metric("Target 1", f"{levels['t1']:.5f}")
                    c3.metric("Target 2", f"{levels['t2']:.5f}")
                    st.metric("Stop Loss", f"{levels['sl']:.5f}")
                else:
                    st.info("No aligned setup across all three timeframes. No new trade signal.")
                tf_data = []
                for tf_name, result in scored.items():
                    r = result["row"]
                    tf_data.append({
                        "Timeframe": tf_name,
                        "Signal": result["label"],
                        "Score": result["score"],
                        "Close": round(float(r["Close"]), 5) if r is not None and pd.notna(r.get("Close")) else None,
                        "RSI(14)": round(float(r["RSI"]), 2) if r is not None and pd.notna(r.get("RSI")) else None,
                        "ATR(14)": round(float(r["ATR"]), 5) if r is not None and pd.notna(r.get("ATR")) else None,
                        "ADX(14)": round(float(r["ADX"]), 2) if r is not None and pd.notna(r.get("ADX")) else None,
                        "+DI": round(float(r["+DI"]), 2) if r is not None and pd.notna(r.get("+DI")) else None,
                        "-DI": round(float(r["-DI"]), 2) if r is not None and pd.notna(r.get("-DI")) else None,
                    })
                st.dataframe(pd.DataFrame(tf_data), use_container_width=True, hide_index=True)
                if latest is not None:
                    st.caption(f"Latest 15m candle close: {latest.name} · Data from Yahoo Finance; may be delayed or unavailable.")
                all_signal_rows.append({"Market": market_name, "Instrument": symbol, "Signal": direction, "Open paper trade": "Yes" if symbol in state["open_trades"] else "No"})

save_state(state)

# Main account summary
st.divider()
st.subheader("💰 Paper Account")
open_trades = state["open_trades"]
history = state["history"]
closed = [h for h in history if h.get("exit") is not None]
realized = sum(float(h.get("pnl_inr", 0) or 0) for h in closed)
entry_count = sum(1 for h in history if h.get("reason") == "PAPER ENTRY")
win_count = sum(1 for h in closed if float(h.get("pnl_inr", 0) or 0) > 0)
win_rate = (win_count / len(closed) * 100) if closed else 0.0
m1, m2, m3, m4 = st.columns(4)
m1.metric("Virtual balance", f"₹{state['balance']:,.2f}")
m2.metric("Realized P/L", f"₹{realized:,.2f}")
m3.metric("Open positions", str(len(open_trades)))
m4.metric("Closed-trade win rate", f"{win_rate:.1f}%" if closed else "N/A")

if open_trades:
    open_rows = []
    for symbol, trade in open_trades.items():
        open_rows.append({"Instrument": symbol, "Side": trade["side"], "Entry": round(trade["entry"], 5), "Target 1": round(trade["t1"], 5), "Target 2": round(trade["t2"], 5), "Stop Loss": round(trade["sl"], 5), "T1 hit": trade["t1_hit"], "Opened (UTC)": trade["opened_at"]})
    st.markdown("#### Open positions")
    st.dataframe(pd.DataFrame(open_rows), use_container_width=True, hide_index=True)
else:
    st.info("No open paper positions right now.")

st.markdown("#### Trade history")
if history:
    hist_df = pd.DataFrame(history)
    cols = [c for c in ["closed_at", "symbol", "side", "entry", "exit", "qty_remaining", "pnl_inr", "reason"] if c in hist_df.columns]
    hist_df = hist_df[cols].rename(columns={"closed_at": "Timestamp (UTC)", "symbol": "Instrument", "side": "Side", "entry": "Entry", "exit": "Exit", "qty_remaining": "Quantity", "pnl_inr": "P/L (INR)", "reason": "Event"})
    st.dataframe(hist_df.iloc[::-1], use_container_width=True, hide_index=True)
    csv = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download trade history CSV", data=csv, file_name="paper_trade_history.csv", mime="text/csv")
else:
    st.write("No trades recorded yet. The bot waits for all three timeframes to align.")

st.markdown("#### Market watch")
if all_signal_rows:
    st.dataframe(pd.DataFrame(all_signal_rows), use_container_width=True, hide_index=True)

with st.expander("Important limitations and safety notes"):
    st.markdown("""
- This is a **rule-based research prototype**, not a proven profitable AI strategy.
- It does **not** connect to MT5, a broker, a wallet, or real-money execution.
- Yahoo Finance data can be delayed, limited, or unavailable; crypto/forex/gold coverage varies by symbol and region.
- The free app only refreshes while the dashboard is open and the hosting process is awake. It is not a guaranteed 24/7 background worker.
- The VWAP is a per-date approximation from the provider's bars; forex and some feeds may report zero or missing volume.
- SMC/BOS/CHoCH/order blocks are represented only by simple rolling swing/breakout proxies here. They are not a full institutional SMC implementation.
- Power Channel is a rolling mean ± 2 standard deviations, not a validated predictive channel.
- The INR P/L conversion uses the editable approximate USD/INR rate and simplified instrument units. It is not broker-accurate lot sizing. Validate contract multipliers, spreads, commissions, slippage, and currency conversion before drawing conclusions.
- The simulator uses candle high/low to approximate fills. It assumes stop-loss first if stop and target are touched in the same candle.
- Never add wallet keys or broker passwords to this app. Real-money trading is intentionally not implemented.
""")

if fetch_errors:
    with st.expander("Data warnings"):
        for err in fetch_errors:
            st.write(err)

st.caption(f"Last dashboard refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} local device time · Auto-refresh: {refresh_seconds}s while page is active")
