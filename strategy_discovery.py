"""Bounded strategy-candidate research. Does not modify/promote the active strategy."""
import numpy as np
import pandas as pd

def _max_drawdown(returns):
    if len(returns) == 0:
        return 0.0
    curve = np.cumprod(1.0 + np.clip(np.asarray(returns, dtype=float), -0.99, None))
    peak = np.maximum.accumulate(curve)
    return float(np.min(curve / np.maximum(peak, 1e-12) - 1.0))

def _candidate_signals(df, name):
    close = pd.to_numeric(df["Close"], errors="coerce")
    ema9 = df.get("EMA9", close.ewm(span=9, adjust=False).mean())
    ema21 = df.get("EMA21", close.ewm(span=21, adjust=False).mean())
    ema50 = df.get("EMA50", close.ewm(span=50, adjust=False).mean())
    rsi = df.get("RSI", pd.Series(np.nan, index=df.index))
    atr = df.get("ATR", pd.Series(np.nan, index=df.index))
    support = df.get("Support", close.rolling(20).min().shift(1))
    resistance = df.get("Resistance", close.rolling(20).max().shift(1))
    upper = df.get("BB_Upper", close.rolling(20).mean() + 2.5 * close.rolling(20).std())
    lower = df.get("BB_Lower", close.rolling(20).mean() - 2.5 * close.rolling(20).std())
    signal = pd.Series(0, index=df.index, dtype=int)
    if name == "Trend EMA 9/21/50":
        signal[(ema9 > ema21) & (ema21 > ema50)] = 1
        signal[(ema9 < ema21) & (ema21 < ema50)] = -1
    elif name == "Mean reversion RSI + Bollinger":
        signal[(close < lower) & (rsi < 30)] = 1
        signal[(close > upper) & (rsi > 70)] = -1
    elif name == "Breakout 20-bar levels":
        signal[(close > resistance.shift(1))] = 1
        signal[(close < support.shift(1))] = -1
    elif name == "Momentum 5-bar":
        momentum = close.pct_change(5)
        signal[momentum > 0.003] = 1
        signal[momentum < -0.003] = -1
    elif name == "Volatility expansion":
        ret = close.pct_change()
        vol = ret.rolling(20).std()
        expanding = vol > vol.rolling(50).median()
        signal[(ret > 0) & expanding] = 1
        signal[(ret < 0) & expanding] = -1
    return signal, atr

def discover_strategies(df, cost_bps=2.0, min_validation_trades=8):
    """Use first 60% for research context and last 40% for chronological validation."""
    if df is None or len(df) < 180 or "Close" not in df:
        return pd.DataFrame()
    candidates = ["Trend EMA 9/21/50", "Mean reversion RSI + Bollinger",
                  "Breakout 20-bar levels", "Momentum 5-bar", "Volatility expansion"]
    split = int(len(df) * 0.60)
    rows = []
    close = pd.to_numeric(df["Close"], errors="coerce")
    next_ret = close.shift(-1) / close - 1.0
    for name in candidates:
        sig, _ = _candidate_signals(df, name)
        # Only take a new position when the candidate changes from flat/opposite; no look-ahead.
        pos = sig.replace(0, np.nan).ffill().fillna(0)
        strat_ret = pos.shift(1).fillna(0) * next_ret
        # Conservative fixed cost charged when signal changes into a non-zero position.
        turnover = (pos.diff().abs().fillna(pos.abs()) > 0).astype(float)
        strat_ret = strat_ret - turnover * (float(cost_bps) / 10000.0)
        val = strat_ret.iloc[split:-1].replace([np.inf, -np.inf], np.nan).dropna()
        active = sig.iloc[split:-1].ne(0)
        trades = int(active.sum())
        if trades < min_validation_trades:
            continue
        # Report returns in percentage terms; no leverage assumed.
        mean_ret = float(val.mean())
        std_ret = float(val.std(ddof=1)) if len(val) > 1 else 0.0
        rows.append({
            "Candidate strategy": name,
            "Validation trades (signal bars)": trades,
            "Validation mean return / bar (%)": mean_ret * 100,
            "Validation total return (%)": (float(np.prod(1 + np.clip(val.to_numpy(), -0.99, None))) - 1) * 100,
            "Validation Sharpe-like": (mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0.0),
            "Validation max drawdown (%)": _max_drawdown(val.to_numpy()) * 100,
            "Research eligible": bool(mean_ret > 0 and np.isfinite(mean_ret)),
            "Note": "Validation-only estimate; costs approximate; not a live-trading claim"
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["Research eligible", "Validation mean return / bar (%)", "Validation max drawdown (%)"],
        ascending=[False, False, False]
    ).reset_index(drop=True)
