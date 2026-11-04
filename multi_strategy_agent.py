"""Advisory AI-agent layer for the deterministic Multi-Strategy Engine.

This is a transparent rule-based adviser, not an external LLM. It explains current
strategy votes and closed-trade evidence; it never places orders or changes weights.
"""
from __future__ import annotations
from typing import Iterable, Mapping, Any

import pandas as pd


def analyse_fibonacci(market_data: Any, lookback: int = 100) -> dict:
    """Return advisory Fibonacci levels from recent OHLC data; never creates votes.

    Uses the highest high and lowest low in the recent window. The order of those
    extrema determines whether levels are measured for an upswing or downswing.
    Zones are descriptive bands around retracement levels, not trade triggers.
    """
    if market_data is None or not isinstance(market_data, pd.DataFrame) or market_data.empty:
        return {"available": False, "reason": "No OHLC data supplied", "direction": "UNKNOWN",
                "retracement": {}, "extensions": {}, "zones": []}
    cols = {str(c).lower(): c for c in market_data.columns}
    if not all(k in cols for k in ("high", "low", "close")):
        return {"available": False, "reason": "High, Low and Close columns are required",
                "direction": "UNKNOWN", "retracement": {}, "extensions": {}, "zones": []}
    window = market_data.tail(max(2, int(lookback)))
    highs = pd.to_numeric(window[cols["high"]], errors="coerce")
    lows = pd.to_numeric(window[cols["low"]], errors="coerce")
    valid = highs.notna() & lows.notna()
    if valid.sum() < 2:
        return {"available": False, "reason": "Insufficient valid OHLC data", "direction": "UNKNOWN",
                "retracement": {}, "extensions": {}, "zones": []}
    hi_idx, lo_idx = highs[valid].idxmax(), lows[valid].idxmin()
    swing_high, swing_low = float(highs.loc[hi_idx]), float(lows.loc[lo_idx])
    span = swing_high - swing_low
    if not span > 0:
        return {"available": False, "reason": "Swing range is zero", "direction": "UNKNOWN",
                "retracement": {}, "extensions": {}, "zones": []}
    # If the low occurred first, price most recently completed an upswing.
    uptrend = list(window.index).index(lo_idx) < list(window.index).index(hi_idx)
    direction = "UPSWING" if uptrend else "DOWN SWING"
    ratios = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
    ext_ratios = (1.272, 1.618, 2.0, 2.618)
    if uptrend:
        retr = {f"{r*100:g}%": round(swing_high - span*r, 8) for r in ratios}
        ext = {f"{r*100:g}%": round(swing_low + span*r, 8) for r in ext_ratios}
    else:
        retr = {f"{r*100:g}%": round(swing_low + span*r, 8) for r in ratios}
        ext = {f"{r*100:g}%": round(swing_high - span*r, 8) for r in ext_ratios}
    atr_col = cols.get("atr")
    atr = float(pd.to_numeric(window[atr_col], errors="coerce").dropna().iloc[-1]) if atr_col and pd.to_numeric(window[atr_col], errors="coerce").notna().any() else span * 0.005
    half_width = max(atr * 0.25, span * 0.001)
    zones = [{"level": key, "low": round(value-half_width, 8), "high": round(value+half_width, 8)}
             for key, value in retr.items() if key in ("38.2%", "50%", "61.8%")]
    close = float(pd.to_numeric(window[cols["close"]], errors="coerce").dropna().iloc[-1])
    nearest = min(retr.items(), key=lambda item: abs(item[1]-close))
    return {"available": True, "direction": direction, "swing_high": swing_high, "swing_low": swing_low,
            "retracement": retr, "extensions": ext, "zones": zones,
            "nearest_retracement": {"level": nearest[0], "price": nearest[1]},
            "note": "Advisory chart analysis only; Fibonacci does not add BUY/SELL votes or trigger trades."}


def assess_multi_strategy(votes: Mapping[str, str] | None, side: str = "WAIT",
                          confidence: float = 0, regime: str = "UNKNOWN",
                          trades: Iterable[dict] | None = None,
                          market_data: Any = None, timeframe: str = "UNKNOWN",
                          symbol: str = "UNKNOWN") -> dict:
    votes = dict(votes or {})
    buy = sum(str(v).upper() == "BUY" for v in votes.values())
    sell = sum(str(v).upper() == "SELL" for v in votes.values())
    wait = sum(str(v).upper() not in ("BUY", "SELL") for v in votes.values())
    active = buy + sell
    # Agreement is calculated only among directional votes. It is NOT the
    # directional vote-strength score, which may count WAIT votes and use a separate gate.
    agreement = (max(buy, sell) / active * 100) if active else 0.0
    closed = []
    for row in (trades or []):
        status = str(row.get("Status", "CLOSED")).upper()
        if "CLOSED" in status:
            closed.append(row)

    conf = float(confidence or 0)
    if not active:
        recommendation = "No active BUY/SELL votes. Keep the decision at NO TRADE and collect more valid market data."
    elif buy == sell:
        recommendation = "BUY/SELL votes are tied. Keep the decision at NO TRADE until evidence resolves the conflict."
    elif str(side).upper() == "WAIT":
        recommendation = (
            f"The engine decision is WAIT. Active directional votes agree at {agreement:.1f}%, "
            f"but the directional vote-strength score is {conf:.1f}%. These are different metrics; "
            "review the engine's eligibility gates and do not force a trade."
        )
    elif conf < 70:
        recommendation = (
            f"Directional vote-strength score is {conf:.1f}%, below the configured 70% minimum. "
            f"Active directional votes agree at {agreement:.1f}%, but agreement is not win probability. "
            "Do not treat this setup as eligible."
        )
    else:
        recommendation = (
            f"The engine currently favours {side}. Directional vote strength is {conf:.1f}% and active "
            f"directional vote agreement is {agreement:.1f}%. Verify session, spread, duplicate-order "
            "and risk checks before any paper entry."
        )
    fibonacci = analyse_fibonacci(market_data)
    return {
        "agent": "Multi-Strategy AI Agent",
        "status": "ADVISORY",
        "engine_side": str(side),
        "confidence": conf,
        "regime": str(regime or "UNKNOWN"),
        "buy_votes": buy,
        "sell_votes": sell,
        "wait_votes": wait,
        "active_vote_agreement_pct": round(agreement, 1),
        "closed_trades_reviewed": len(closed),
        "symbol": str(symbol),
        "timeframe": str(timeframe),
        "fibonacci_analysis": fibonacci,
        "fibonacci_affects_votes": False,
        "recommendation": recommendation,
        "execution_authority": False,
        "automatic_weight_changes": False,
    }
