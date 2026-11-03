"""Automatic Fibonacci retracement, extension and dynamic-zone analysis.

The module is a confluence input, never a standalone trade trigger.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

RETRACEMENTS = (0.236, 0.382, 0.500, 0.618, 0.786)
EXTENSIONS = (1.272, 1.618, 2.000)

def fibonacci_analysis(df: pd.DataFrame, lookback: int = 120) -> dict:
    if df is None or len(df) < 20:
        return {"side": "WAIT", "status": "NO DATA", "levels": {}, "zone": None, "reason": "Insufficient candles"}
    d = df.tail(max(lookback, 20)).copy()
    hi_idx, lo_idx = d["High"].idxmax(), d["Low"].idxmin()
    hi, lo, close = float(d.loc[hi_idx, "High"]), float(d.loc[lo_idx, "Low"]), float(d["Close"].iloc[-1])
    if not np.isfinite([hi, lo, close]).all() or hi <= lo:
        return {"side": "WAIT", "status": "INVALID SWING", "levels": {}, "zone": None, "reason": "Invalid swing range"}
    # The most recent extreme determines the active impulse direction.
    bullish = d.index.get_loc(hi_idx) > d.index.get_loc(lo_idx)
    span = hi - lo
    levels = {f"R{r:.3f}": (hi - span*r if bullish else lo + span*r) for r in RETRACEMENTS}
    extensions = {f"E{e:.3f}": (lo + span*e if bullish else hi - span*e) for e in EXTENSIONS}
    atr = float(d["ATR"].iloc[-1]) if "ATR" in d and pd.notna(d["ATR"].iloc[-1]) else span*0.01
    half_width = max(atr*0.20, close*0.0002)
    # Dynamic zone around the golden pocket (0.50–0.618); width adapts to ATR.
    z1, z2 = sorted((levels["R0.500"], levels["R0.618"]))
    zone = {"low": z1-half_width, "high": z2+half_width, "center": (z1+z2)/2}
    in_zone = zone["low"] <= close <= zone["high"]
    side = "BUY" if bullish and in_zone else "SELL" if (not bullish and in_zone) else "WAIT"
    return {"side": side, "status": "ANALYZED", "swing_high": hi, "swing_low": lo,
            "impulse": "BULLISH" if bullish else "BEARISH", "levels": levels,
            "extensions": extensions, "zone": zone, "in_zone": in_zone,
            "reason": "Price is inside the dynamic 0.50–0.618 retracement zone" if in_zone else "Price is outside the golden-pocket zone"}
