import pandas as pd
from trading_engine.adaptive_learning import learning_diagnostics, update_model

FEATURES = ["f1", "f2"]

def test_learning_diagnostics_flags_repeated_loss_patterns():
    trades = []
    for i in range(6):
        trades.append({"Status": "CLOSED", "Market": "Gold", "Market regime": "RANGING",
                       "Side": "BUY", "Timeframe": "5m", "Strategy names": "Mean reversion",
                       "PnL": -1.0, "f1": float(i), "f2": 0.0})
    report = learning_diagnostics(trades, min_group_samples=5)
    assert report["closed_samples"] == 6
    assert any(g["dimension"] == "Market regime" and g["pattern_flag"] for g in report["groups"])
    assert "not proof" in report["warning"]

def test_model_waits_for_minimum_completed_trade_sample(tmp_path):
    trades = [{"Status": "CLOSED", "PnL": 1.0, "f1": 0.2, "f2": 0.1} for _ in range(20)]
    result = update_model(trades, FEATURES, tmp_path / "model.pkl")
    assert result["status"] == "collecting"
    assert result["model"] is None
    assert not (tmp_path / "model.pkl").exists()
