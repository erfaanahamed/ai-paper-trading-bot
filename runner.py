"""Market scan integrating causal indicators, rule signals, and validated ML gating."""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from .config import MARKETS, DATA_DIR
from .data import fetch_ohlcv, validate_ohlcv
from .indicators import add_indicators
from .features import build_features
from .signals import primary_signal, primary_signals_batch, _regime
from .labels import triple_barrier_labels
from .meta import fit_meta_model, save_artifact, load_artifact, predict_win_prob
from .gates import decide
from .state import load_state, load_settings, log_audit
from .market_hours import entry_session_status

log = logging.getLogger(__name__)
MODEL_DIR = DATA_DIR / "models"

def _train_or_load(market, df, primary, features, config):
    """Retrain periodically from completed historical outcomes; never train on the current incomplete label."""
    path = MODEL_DIR / (market.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "") + ".pkl")
    artifact = load_artifact(path)
    # Fit only when no valid artifact exists; a future scheduled retrainer can refresh from new completed bars.
    if artifact:
        return artifact, "loaded validated model"
    labels = triple_barrier_labels(df, primary, tp_mult=2.0, sl_mult=1.0,
                                   max_bars=int(config.get("max_bars", 48)))
    # Avoid training on labels whose forward outcome window has not completed.
    if len(labels):
        labels = labels.copy()
        cutoff = max(0, len(df) - int(config.get("max_bars", 48)))
        labels.iloc[cutoff:, labels.columns.get_loc("label")] = np.nan
    artifact = fit_meta_model(features, labels, primary, config={"max_bars": int(config.get("max_bars", 48))})
    if artifact.get("ok"):
        save_artifact(artifact, path)
        return artifact, "trained and passed walk-forward AUC gate"
    return None, artifact.get("reason", "model not qualified")

def scan_all_markets(config=None):
    config = config or {}
    settings = load_settings()
    state = load_state()
    history = state.get("trade_history", [])
    events = []
    for market, item in MARKETS.items():
        try:
            if settings.get("paused") or not settings.get("symbol_enabled", {}).get(market, True):
                events.append({"event":"skip","market":market,"reason":"paused or symbol disabled"}); continue
            if settings.get("session_protection", True):
                allowed, session_reason = entry_session_status(
                    market,
                    buffer_minutes=int(settings.get("session_buffer_minutes", 30)),
                    no_weekend_trading=bool(settings.get("no_weekend_trading", True)),
                )
                if not allowed:
                    events.append({"event":"skip","market":market,"reason":session_reason,"action":"WAIT"}); continue
            raw = fetch_ohlcv(item["symbol"], item.get("interval","15m"))
            ok, reason = validate_ohlcv(raw, item.get("interval","15m"))
            if not ok:
                events.append({"event":"skip","market":market,"reason":reason}); continue
            df = add_indicators(raw)
            if df.empty or df[["ATR","RSI","ADX","EMA9","EMA21","EMA50"]].tail(1).isna().any().any():
                events.append({"event":"skip","market":market,"reason":"indicator warm-up / missing values"}); continue
            regime = _regime(df.iloc[-1])
            signal = primary_signal(df, regime, market, item.get("interval","15m"), history)
            primary = primary_signals_batch(df, market, item.get("interval","15m"), history)
            features = build_features(df)
            artifact, model_status = _train_or_load(market, df, primary, features, config)
            side = signal["side"]
            row = features.iloc[-1].to_dict()
            row.update({"Close":float(df.Close.iloc[-1]), "ATR":float(df.ATR.iloc[-1])})
            gate = decide(side, row, artifact, market, config.get("gate", {}))
            events.append({
                "event":"scan","market":market,"data_ok":True,
                "rule_signal":side,"signal":side if gate["action"]=="TRADE" else "WAIT",
                "action":gate["action"],"p_win":gate.get("p_win"),
                "expected_value_atr":gate.get("ev"),"regime":signal.get("regime"),
                "rule_score":signal.get("score"),"selected_strategies":signal.get("selected"),
                "model_status":model_status,"gate_reason":gate.get("reason"),
                "last_bar":str(df.index[-1])
            })
        except Exception as e:
            log.exception("market scan failed: %s", market)
            events.append({"event":"error","market":market,"reason":str(e)})
    log_audit("market_scan", {"events":len(events), "qualified":sum(e.get("action")=="TRADE" for e in events)})
    return events
