"""Deterministic analysis agents used by the AI Brain paper-trading dashboard.

These are local, rule-based analytical components, not external LLM services.
Their confidence values describe indicator/pattern agreement, not calibrated win probabilities.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd


def chart_analysis_agent(df: pd.DataFrame, htf_bias: str = "Unknown") -> dict:
    """Combine trend, momentum, directional-strength and candle evidence."""
    if df is None or df.empty:
        return {"name": "Chart Analysis AI Agent", "side": "WAIT", "confidence": 0,
                "status": "NO DATA", "reason": "No candle data"}
    r = df.iloc[-1]
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
                "status": "ANALYZED", "reason": "Indicators are mixed or lack sufficient evidence"}
    side = "BUY" if buys > sells else "SELL"
    confidence = round(100 * max(buys, sells) / max(len(checks), 1))
    if confidence < 60:
        side = "WAIT"
    evidence = f"{buys} bullish / {sells} bearish directional checks"
    return {"name": "Chart Analysis AI Agent", "side": side, "confidence": confidence,
            "status": "ANALYZED", "reason": evidence}


def _confirmed_pivots(df: pd.DataFrame, left: int = 3, right: int = 3):
    """Return confirmed local swing pivots, requiring bars on both sides."""
    if df is None or len(df) < left + right + 10:
        return []
    highs = pd.to_numeric(df["High"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(df["Low"], errors="coerce").to_numpy(dtype=float)
    pivots = []
    for i in range(left, len(df) - right):
        if not np.isfinite(highs[i]) or not np.isfinite(lows[i]):
            continue
        if highs[i] >= np.nanmax(highs[i-left:i+right+1]) and highs[i] > np.nanmax(highs[i-left:i]):
            pivots.append((i, "H", float(highs[i])))
        if lows[i] <= np.nanmin(lows[i-left:i+right+1]) and lows[i] < np.nanmin(lows[i-left:i]):
            pivots.append((i, "L", float(lows[i])))
    pivots.sort(key=lambda p: p[0])
    # Collapse consecutive pivots of the same type, retaining the more extreme one.
    collapsed = []
    for pivot in pivots:
        if collapsed and collapsed[-1][1] == pivot[1]:
            prev = collapsed[-1]
            if (pivot[1] == "H" and pivot[2] > prev[2]) or (pivot[1] == "L" and pivot[2] < prev[2]):
                collapsed[-1] = pivot
        else:
            collapsed.append(pivot)
    return collapsed


def harmonic_pattern_agent(df: pd.DataFrame) -> dict:
    """Heuristic Gartley/Bat/Butterfly/Crab detector from five confirmed swing pivots."""
    base = {"name": "Harmonic Pattern AI Agent", "side": "WAIT", "confidence": 0,
            "pattern": "None", "status": "NO PATTERN",
            "reason": "No valid five-pivot harmonic structure detected"}
    if df is None or df.empty or not {"High", "Low"}.issubset(df.columns):
        base.update({"status": "NO DATA", "reason": "No candle data"})
        return base
    pivots = _confirmed_pivots(df)
    if len(pivots) < 5:
        return base
    p = pivots[-5:]
    types = "".join(x[1] for x in p)
    if types not in ("LHLHL", "HLHLH"):
        return base
    X, A, B, C, D = [x[2] for x in p]
    xa, ab, bc, cd = abs(A-X), abs(B-A), abs(C-B), abs(D-C)
    ad = abs(D-A)
    if min(xa, ab, bc, cd) <= 0:
        return base
    r_ab_xa, r_bc_ab, r_cd_bc, r_ad_xa = ab/xa, bc/ab, cd/bc, ad/xa
    # Ratios use tolerant zones because market swings are noisy; these are heuristics.
    patterns = {
        "Gartley": (0.618, 0.786, 0.382, 0.886, 1.272, 1.618, 0.786, 0.10),
        "Bat": (0.382, 0.500, 0.382, 0.886, 1.618, 2.618, 0.886, 0.12),
        "Butterfly": (0.786, 0.786, 0.382, 0.886, 1.618, 2.618, 1.270, 0.13),
        "Crab": (0.382, 0.618, 0.382, 0.886, 2.240, 3.618, 1.618, 0.15),
    }
    best = None
    for name, (ab_lo, ab_hi, bc_lo, bc_hi, cd_lo, cd_hi, ad_target, tol) in patterns.items():
        if not (ab_lo-tol <= r_ab_xa <= ab_hi+tol and
                bc_lo-tol <= r_bc_ab <= bc_hi+tol and
                cd_lo-tol <= r_cd_bc <= cd_hi+tol and
                abs(r_ad_xa-ad_target) <= tol):
            continue
        # D is the last pivot; LHLHL suggests bullish reversal, HLHLH bearish reversal.
        side = "BUY" if types == "LHLHL" else "SELL"
        errors = [abs(r_ab_xa-min(max(r_ab_xa, ab_lo), ab_hi)),
                  abs(r_bc_ab-min(max(r_bc_ab, bc_lo), bc_hi)),
                  abs(r_cd_bc-min(max(r_cd_bc, cd_lo), cd_hi)),
                  abs(r_ad_xa-ad_target)]
        confidence = int(round(max(60, min(90, 90 - 100*sum(errors)))))
        candidate = (confidence, name, side)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return base
    confidence, name, side = best
    return {"name": "Harmonic Pattern AI Agent", "side": side, "confidence": confidence,
            "pattern": name, "status": "PATTERN CANDIDATE",
            "reason": f"Heuristic {name} ratios matched; requires independent validation"}
