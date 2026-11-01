"""Self-Learning AI Agent: advisory layer over the existing adaptive learning engine.

It summarizes validated learning diagnostics and recommends research actions. It does
not place orders, bypass risk gates, or autonomously change live trading rules.
"""
from __future__ import annotations
from typing import Iterable


def assess_learning(trades: Iterable[dict], diagnostics_fn=None, model_status: dict | None = None) -> dict:
    """Return an explainable advisory report from completed-trade evidence.

    The engine remains responsible for data storage/training/validation; this agent
    interprets the report and recommends what to investigate next.
    """
    rows = list(trades or [])
    closed = []
    for row in rows:
        status = str(row.get('Status', 'CLOSED')).upper()
        if 'CLOSED' in status:
            closed.append(row)
    diag = {}
    if diagnostics_fn is not None:
        try:
            diag = diagnostics_fn(closed) or {}
        except Exception as exc:
            diag = {'warning': f'Diagnostics unavailable: {str(exc)[:120]}', 'groups': []}
    flags = [g for g in diag.get('groups', []) if g.get('pattern_flag')]
    status = (model_status or {}).get('status', 'collecting')
    if flags:
        recommendation = 'Review flagged market/regime/strategy groups; test alternatives on unseen data before changing weights.'
    elif status == 'promoted':
        recommendation = 'A candidate passed the configured holdout gate; continue paper monitoring and periodic walk-forward validation.'
    else:
        recommendation = 'Continue collecting clean closed-trade data; do not change active rules until sufficient evidence passes validation.'
    return {
        'agent': 'Self-Learning AI Agent',
        'status': 'ADVISORY',
        'closed_trades': len(closed),
        'flagged_patterns': flags,
        'model_status': status,
        'recommendation': recommendation,
        'diagnostics': diag,
        'execution_authority': False,
        'automatic_rule_changes': False,
    }
