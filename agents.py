"""Deterministic analysis agents used by the Hermes AI Agent paper-trading dashboard.

These are local, rule-based analytical components, not external LLM services.
Their confidence values describe indicator/pattern agreement, not calibrated win probabilities.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
try:
    from trading_engine.fibonacci import fibonacci_analysis
except Exception:
    try:
        from .trading_engine.fibonacci import fibonacci_analysis
    except Exception:
        fibonacci_analysis = None


def chart_analysis_agent(df: pd.DataFrame, htf_bias: str = "Unknown") -> dict:
    """Combine trend, momentum, directional-strength and candle evidence."""
    if df is None or df.empty:
        return {"name": "Chart Analysis AI Agent", "side": "WAIT", "confidence": 0,
                "status": "NO DATA", "reason": "No candle data", "fibonacci": {}}
    r = df.iloc[-1]
    fib = fibonacci_analysis(df) if fibonacci_analysis else {"side": "WAIT", "status": "UNAVAILABLE", "levels": {}, "extensions": {}, "zone": None}
    checks = []
    def val(key):
        try:
            x = float(r.get(key, np.nan))
            return x if np.isfinite(x) else np.nan
        except Exception:
            return np.nan
    close, e9, e21, e50 = (val("Close"), val("EMA9"), val("EMA21"), val("EMA50"))
    e200 = val("EMA200")
    rsi, adx, pdi, mdi = val("RSI"), val("ADX"), val("PLUS_DI"), val("MINUS_DI")
    if all(np.isfinite(x) for x in (e9, e21, e50)):
        checks.append("BUY" if e9 > e21 > e50 else "SELL" if e9 < e21 < e50 else "WAIT")
    if np.isfinite(close) and np.isfinite(e200):
        checks.append("BUY" if close > e200 else "SELL" if close < e200 else "WAIT")
    if np.isfinite(pdi) and np.isfinite(mdi) and np.isfinite(adx):
        checks.append(("BUY" if pdi > mdi else "SELL") if adx >= 15 and pdi != mdi else "WAIT")
    if np.isfinite(rsi):
        checks.append("BUY" if 52 <= rsi <= 72 else "SELL" if 28 <= rsi <= 48 else "WAIT")
    if htf_bias in ("Bullish", "Bearish"):
        checks.append("BUY" if htf_bias == "Bullish" else "SELL")
    directional = [x for x in checks if x in ("BUY", "SELL")]
    buys, sells = directional.count("BUY"), directional.count("SELL")
    total = len(directional)
    if total == 0 or buys == sells:
        return {"name": "Chart Analysis AI Agent", "side": "WAIT", "confidence": 0,
                "status": "ANALYZED", "reason": "Indicators are mixed or lack sufficient evidence", "fibonacci": fib}
    side = "BUY" if buys > sells else "SELL"
    confidence = round(100 * max(buys, sells) / max(len(checks), 1))
    evidence = f"{buys} bullish / {sells} bearish directional checks"
    return {"name": "Chart Analysis AI Agent", "side": side, "confidence": confidence,
            "status": "ANALYZED", "reason": evidence, "fibonacci": fib}
