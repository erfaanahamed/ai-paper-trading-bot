from trading_engine.multi_strategy_agent import assess_multi_strategy


def test_agent_reports_votes_without_execution_authority():
    report = assess_multi_strategy({"Trend": "BUY", "Momentum": "BUY", "Range": "SELL", "Other": "WAIT"},
                                   side="BUY", confidence=75, regime="TRENDING", trades=[])
    assert report["buy_votes"] == 2
    assert report["sell_votes"] == 1
    assert report["wait_votes"] == 1
    assert report["execution_authority"] is False
    assert report["automatic_weight_changes"] is False


def test_agent_advises_no_trade_when_votes_tied():
    report = assess_multi_strategy({"A": "BUY", "B": "SELL"}, side="WAIT", confidence=50)
    assert "tied" in report["recommendation"]


def test_fibonacci_analysis_is_advisory_and_does_not_change_votes():
    import pandas as pd
    from trading_engine.multi_strategy_agent import analyse_fibonacci
    data = pd.DataFrame({
        "High": [10, 12, 14, 13, 12],
        "Low": [9, 10, 11, 10, 9.5],
        "Close": [9.5, 11, 13, 11, 10],
        "ATR": [0.5] * 5,
    })
    report = assess_multi_strategy({"Trend": "BUY", "Momentum": "SELL"},
                                   side="WAIT", confidence=50, market_data=data, timeframe="5m", symbol="XAUUSD")
    fib = report["fibonacci_analysis"]
    assert fib["available"] is True
    assert "61.8%" in fib["retracement"]
    assert "161.8%" in fib["extensions"]
    assert {z["level"] for z in fib["zones"]} == {"38.2%", "50%", "61.8%"}
    assert report["fibonacci_affects_votes"] is False
    assert report["buy_votes"] == 1 and report["sell_votes"] == 1


def test_fibonacci_handles_missing_market_data():
    from trading_engine.multi_strategy_agent import analyse_fibonacci
    assert analyse_fibonacci(None)["available"] is False


def test_root_fallback_fibonacci_import():
    # Streamlit deployments may fall back to the repository-root module.
    import importlib.util
    from pathlib import Path
    root_module = Path(__file__).resolve().parents[2] / "multi_strategy_agent.py"
    spec = importlib.util.spec_from_file_location("hermes_root_multi_strategy_agent", root_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import pandas as pd
    frame = pd.DataFrame({"High": [10, 12, 14], "Low": [9, 10, 11], "Close": [9.5, 11, 13]})
    assert module.analyse_fibonacci(frame)["available"] is True
