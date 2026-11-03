"""Advisory AI-agent layer for the deterministic Multi-Strategy Engine.

This is a transparent rule-based adviser, not an external LLM. It explains current
strategy votes and closed-trade evidence; it never places orders or changes weights.
"""
from __future__ import annotations
from collections import Counter
from typing import Iterable, Mapping, Any


def assess_multi_strategy(votes: Mapping[str, str] | None, side: str = "WAIT",
                          confidence: float = 0, regime: str = "UNKNOWN",
                          trades: Iterable[dict] | None = None) -> dict:
    votes = dict(votes or {})
    buy = sum(str(v).upper() == "BUY" for v in votes.values())
    sell = sum(str(v).upper() == "SELL" for v in votes.values())
    wait = sum(str(v).upper() not in ("BUY", "SELL") for v in votes.values())
    active = buy + sell
    agreement = (max(buy, sell) / active * 100) if active else 0.0
    closed = []
    for row in (trades or []):
        status = str(row.get("Status", "CLOSED")).upper()
        if "CLOSED" in status:
            closed.append(row)
    if not active:
        recommendation = "No active strategy votes. Keep the decision at NO TRADE and collect more valid market data."
    elif buy == sell:
        recommendation = "BUY/SELL votes are tied. Keep the decision at NO TRADE until evidence resolves the conflict."
    elif side == "WAIT":
        recommendation = "The engine has not confirmed an entry. Do not force a trade; review vote agreement and risk gates."
    elif False:
        recommendation = (
            f"Directional vote-strength score is {float(confidence or 0):.1f}%, below the configured minimum. "
            f"Active directional votes agree at {agreement:.1f}%, but agreement is not win probability. "
            "Do not treat this setup as eligible."
        )
    else:
        recommendation = f"The engine currently favours {side}; verify session, spread, duplicate-order and risk checks before any paper entry."
    return {
        "agent": "Multi-Strategy AI Agent",
        "status": "ADVISORY",
        "engine_side": str(side),
        "confidence": float(confidence or 0),
        "regime": str(regime or "UNKNOWN"),
        "buy_votes": buy,
        "sell_votes": sell,
        "wait_votes": wait,
        "active_vote_agreement_pct": round(agreement, 1),
        "closed_trades_reviewed": len(closed),
        "recommendation": recommendation,
        "execution_authority": False,
        "automatic_weight_changes": False,
    }
