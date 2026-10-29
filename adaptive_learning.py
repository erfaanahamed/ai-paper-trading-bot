"""Conservative adaptive learning from completed paper trades only.

This module proposes adjustments; it never routes broker orders. A candidate model
is promoted only after a chronological out-of-sample AUC gate and is saved atomically.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

MIN_SAMPLES = 300
MIN_PER_CLASS = 50
MIN_OOS_AUC = 0.53
MODEL_VERSION = "adaptive-paper-v1"


def _closed_frame(trades: Iterable[dict], feature_columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(list(trades or []))
    if frame.empty or not all(c in frame.columns for c in feature_columns):
        return pd.DataFrame()
    status = frame.get("Status", pd.Series("CLOSED", index=frame.index)).astype(str)
    frame = frame[status.str.contains("CLOSED", case=False, na=False)].copy()
    pnl_col = next((c for c in ("PnL (estimated price-units)", "PnL (price units)", "Net PnL", "PnL") if c in frame), None)
    if not pnl_col:
        return pd.DataFrame()
    frame["_pnl"] = pd.to_numeric(frame[pnl_col], errors="coerce")
    frame = frame.dropna(subset=["_pnl"])
    if "Exit reason" in frame:
        frame = frame[frame["Exit reason"].notna()]
    frame["_label"] = (frame["_pnl"] > 0).astype(int)
    return frame


def _atomic_pickle(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _load_artifact(path: Path):
    try:
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        if isinstance(obj, dict) and obj.get("version") == MODEL_VERSION:
            return obj
    except (OSError, ValueError, pickle.PickleError, EOFError, AttributeError, ImportError):
        pass
    return None


def update_model(trades: Iterable[dict], feature_columns: list[str], model_path: str | Path) -> dict:
    """Validate a new candidate chronologically and promote it only if it qualifies.

    The last 30% of closed trades are kept out of training for the candidate check.
    This is a conservative holdout, not a substitute for repeated purged walk-forward
    testing across multiple market regimes.
    """
    path = Path(model_path)
    frame = _closed_frame(trades, feature_columns)
    previous = _load_artifact(path)
    if frame.empty:
        return {"status": "collecting", "samples": 0, "reason": "No eligible closed trades with saved entry features.", "model": None}

    # Stable digest lets the app avoid retraining on every Streamlit rerun.
    digest_frame = frame[feature_columns + ["_pnl"]].copy()
    digest = hashlib.sha256(pd.util.hash_pandas_object(digest_frame, index=True).values.tobytes()).hexdigest()
    if previous and previous.get("data_digest") == digest:
        return {"status": "loaded", "samples": len(frame), "auc": previous.get("oos_auc"),
                "brier": previous.get("oos_brier"), "reason": "Loaded the previously validated model; no new completed-trade data.",
                "model": previous.get("model"), "promoted": False}

    y = frame["_label"].to_numpy(dtype=int)
    counts = np.bincount(y, minlength=2)
    if len(frame) < MIN_SAMPLES or counts.min() < MIN_PER_CLASS:
        return {"status": "collecting", "samples": len(frame), "class_counts": {"losses": int(counts[0]), "wins": int(counts[1])},
                "reason": f"Needs at least {MIN_SAMPLES} completed trades and at least {MIN_PER_CLASS} wins and {MIN_PER_CLASS} losses.",
                "model": previous.get("model") if previous else None, "promoted": False}

    x = frame[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    split = int(len(frame) * 0.70)
    if split < 200 or len(frame) - split < 50:
        return {"status": "collecting", "samples": len(frame), "reason": "Insufficient chronological holdout size.",
                "model": previous.get("model") if previous else None, "promoted": False}
    train_y, test_y = y[:split], y[split:]
    if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
        return {"status": "collecting", "samples": len(frame), "reason": "Both classes must be present in training and holdout periods.",
                "model": previous.get("model") if previous else None, "promoted": False}

    def make_model():
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()),
                         ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))])

    try:
        candidate = make_model()
        candidate.fit(x.iloc[:split], train_y)
        probabilities = candidate.predict_proba(x.iloc[split:])[:, 1]
        auc = float(roc_auc_score(test_y, probabilities))
        brier = float(brier_score_loss(test_y, probabilities))
        previous_auc = float(previous.get("oos_auc", 0.0)) if previous else 0.0
        # Never replace a qualified champion with an unqualified or materially worse model.
        if not np.isfinite(auc) or auc < MIN_OOS_AUC:
            return {"status": "rejected", "samples": len(frame), "auc": auc, "brier": brier,
                    "reason": f"Candidate rejected: chronological holdout AUC must be at least {MIN_OOS_AUC:.2f}.",
                    "model": previous.get("model") if previous else None, "promoted": False}
        if previous and previous_auc >= MIN_OOS_AUC and auc < previous_auc - 0.02:
            return {"status": "rejected", "samples": len(frame), "auc": auc, "brier": brier,
                    "reason": "Candidate rejected: holdout AUC is more than 0.02 below the current validated model; previous model retained.",
                    "model": previous.get("model"), "promoted": False}
        champion = make_model()
        champion.fit(x, y)
        # Keep a separate last-known-good artifact before replacing the champion.
        if previous:
            _atomic_pickle(path.with_suffix(path.suffix + ".previous.pkl"), previous)
        artifact = {"version": MODEL_VERSION, "model": champion, "features": list(feature_columns),
                    "oos_auc": auc, "oos_brier": brier, "samples": int(len(frame)),
                    "wins": int(counts[1]), "losses": int(counts[0]), "data_digest": digest,
                    "trained_at": pd.Timestamp.now(tz="UTC").isoformat()}
        _atomic_pickle(path, artifact)
        return {"status": "promoted", "samples": len(frame), "auc": auc, "brier": brier,
                "reason": "Candidate passed chronological holdout and was promoted; previous model remains recoverable from the audit history only if separately backed up.",
                "model": champion, "promoted": True}
    except Exception as exc:
        return {"status": "error", "samples": len(frame), "reason": f"Training failed safely: {str(exc)[:160]}",
                "model": previous.get("model") if previous else None, "promoted": False}


def learning_diagnostics(trades: Iterable[dict], min_group_samples: int = 5) -> dict:
    """Summarize repeat-loss patterns by market, regime, side and attributed strategy."""
    frame = pd.DataFrame(list(trades or []))
    if frame.empty:
        return {"closed_samples": 0, "groups": [], "warning": "No trade history yet."}
    status = frame.get("Status", pd.Series("CLOSED", index=frame.index)).astype(str)
    frame = frame[status.str.contains("CLOSED", case=False, na=False)].copy()
    pnl_col = next((c for c in ("PnL (estimated price-units)", "PnL (price units)", "Net PnL", "PnL") if c in frame), None)
    if not pnl_col:
        return {"closed_samples": 0, "groups": [], "warning": "Closed trade P&L column not found."}
    frame["_pnl"] = pd.to_numeric(frame[pnl_col], errors="coerce")
    frame = frame.dropna(subset=["_pnl"])
    frame["_win"] = frame["_pnl"] > 0
    groups = []
    dimensions = [c for c in ("Market", "Market regime", "Side", "Timeframe", "TF") if c in frame]
    for col in dimensions:
        for key, part in frame.groupby(col, dropna=False):
            if len(part) < min_group_samples:
                continue
            expectancy = float(part["_pnl"].mean())
            groups.append({"dimension": col, "group": str(key), "trades": int(len(part)),
                           "win_rate": float(part["_win"].mean()), "expectancy": expectancy,
                           "pattern_flag": bool(expectancy < 0 and float(part["_win"].mean()) < 0.45),
                           "action": "Investigate; do not auto-disable from this sample alone." if expectancy < 0 else "No loss-pattern flag."})
    if "Strategy names" in frame:
        strat_rows = []
        for _, row in frame.iterrows():
            names = [s.strip() for s in str(row.get("Strategy names", "")).split("; ") if s.strip() and s.strip().lower() != "nan"]
            for name in names:
                strat_rows.append({"strategy": name, "_pnl": row["_pnl"], "_win": row["_win"]})
        if strat_rows:
            sf = pd.DataFrame(strat_rows)
            for name, part in sf.groupby("strategy"):
                if len(part) >= min_group_samples:
                    expectancy = float(part["_pnl"].mean())
                    groups.append({"dimension": "Strategy", "group": str(name), "trades": int(len(part)),
                                   "win_rate": float(part["_win"].mean()), "expectancy": expectancy,
                                   "pattern_flag": bool(expectancy < 0 and float(part["_win"].mean()) < 0.45),
                                   "action": "Investigate; require out-of-sample evidence before changing weights." if expectancy < 0 else "No loss-pattern flag."})
    groups.sort(key=lambda g: (not g["pattern_flag"], g["expectancy"]))
    return {"closed_samples": int(len(frame)), "groups": groups,
            "warning": "Pattern flags are hypotheses, not proof of causation. No strategy is disabled from a few losses."}
