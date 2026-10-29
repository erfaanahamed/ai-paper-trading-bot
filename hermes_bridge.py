"""
Hermes Agent Bridge for AI Brain V7
===================================
This module lets an external agent (Hermes Agent) send trading decisions
while AI Brain V7 keeps full control of risk, news protection, position sizing,
and paper execution.

Usage from Hermes Agent:
  1. Hermes analyses market data
  2. Hermes calls submit_decision(...) with BUY / SELL / WAIT
  3. AI Brain applies risk filters and either accepts or rejects the trade

This is the recommended way to "inbuild" Hermes into your bot.
"""

from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .config import DATA_DIR, DEFAULT_SETTINGS, MARKETS
from .state import load_state, load_settings, log_audit

log = logging.getLogger(__name__)

BRIDGE_DIR = DATA_DIR / "hermes_bridge"
BRIDGE_DIR.mkdir(parents=True, exist_ok=True)

DECISION_FILE = BRIDGE_DIR / "latest_decision.json"
STATUS_FILE = BRIDGE_DIR / "bridge_status.json"
HISTORY_FILE = BRIDGE_DIR / "decision_history.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def submit_decision(
    side: str,
    market: str = "Gold (XAUUSD)",
    confidence: float = 0.0,
    reason: str = "",
    model: str = "Hermes",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Called by Hermes Agent (or any external agent).

    Parameters
    ----------
    side : "BUY" | "SELL" | "WAIT"
    market : must match a key in MARKETS
    confidence : 0.0 – 1.0 (optional)
    reason : short text explanation from the agent
    model : which model produced the decision
    extra : any extra metadata the agent wants to store

    Returns
    -------
    dict with acceptance status and any rejection reason
    """
    side = str(side).upper().strip()
    if side not in ("BUY", "SELL", "WAIT"):
        return {"accepted": False, "reason": f"invalid side: {side}"}

    if market not in MARKETS:
        return {"accepted": False, "reason": f"unknown market: {market}"}

    settings = load_settings()
    if settings.get("paused", False):
        return {"accepted": False, "reason": "bot is paused"}

    decision = {
        "timestamp": _now(),
        "side": side,
        "market": market,
        "confidence": float(confidence),
        "reason": str(reason)[:500],
        "model": str(model),
        "extra": extra or {},
        "status": "pending",
    }

    # Write latest decision
    DECISION_FILE.write_text(json.dumps(decision, indent=2))

    # Append to history
    with HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(decision) + "\n")

    log_audit("hermes_decision", {
        "side": side,
        "market": market,
        "confidence": confidence,
        "model": model,
    })

    log.info("Hermes decision received: %s %s (conf=%.2f)", side, market, confidence)

    return {
        "accepted": True,
        "status": "pending",
        "message": "Decision received. AI Brain risk engine will evaluate it.",
        "decision": decision,
    }


def get_latest_decision() -> Optional[Dict[str, Any]]:
    """Read the most recent decision submitted by Hermes."""
    if not DECISION_FILE.exists():
        return None
    try:
        return json.loads(DECISION_FILE.read_text())
    except Exception as e:
        log.warning("Failed to read decision file: %s", e)
        return None


def clear_decision():
    """Clear the pending decision after it has been processed."""
    if DECISION_FILE.exists():
        DECISION_FILE.unlink(missing_ok=True)


def update_bridge_status(status: str, details: Optional[Dict] = None):
    payload = {
        "timestamp": _now(),
        "status": status,
        "details": details or {},
    }
    STATUS_FILE.write_text(json.dumps(payload, indent=2))


def get_bridge_status() -> Dict[str, Any]:
    if not STATUS_FILE.exists():
        return {"status": "idle", "message": "No Hermes activity yet"}
    try:
        return json.loads(STATUS_FILE.read_text())
    except Exception:
        return {"status": "error", "message": "Could not read status"}


# Convenience helpers for Hermes Agent skills / tools
def hermes_tool_schema() -> Dict[str, Any]:
    """JSON schema that Hermes can use as a tool definition."""
    return {
        "name": "submit_trading_decision",
        "description": (
            "Submit a trading decision (BUY, SELL or WAIT) for the AI Brain V7 engine. "
            "AI Brain will still apply risk management, news protection and position sizing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "side": {
                    "type": "string",
                    "enum": ["BUY", "SELL", "WAIT"],
                    "description": "Trading decision"
                },
                "market": {
                    "type": "string",
                    "description": "Market name, e.g. 'Gold (XAUUSD)'"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence between 0 and 1"
                },
                "reason": {
                    "type": "string",
                    "description": "Short reason for the decision"
                }
            },
            "required": ["side"]
        }
    }
