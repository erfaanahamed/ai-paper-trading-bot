import pandas as pd
from trading_engine.agents import chart_analysis_agent, harmonic_pattern_agent


def test_chart_agent_reports_wait_without_data():
    result = chart_analysis_agent(pd.DataFrame())
    assert result["side"] == "WAIT"
    assert result["status"] == "NO DATA"


def test_chart_agent_can_detect_bullish_alignment():
    df = pd.DataFrame([{
        "Close": 110, "EMA9": 109, "EMA21": 108, "EMA50": 106, "EMA200": 100,
        "RSI": 60, "ADX": 28, "PLUS_DI": 30, "MINUS_DI": 10
    }])
    result = chart_analysis_agent(df, "Bullish")
    assert result["side"] == "BUY"
    assert result["confidence"] >= 70


def test_harmonic_agent_handles_insufficient_data():
    df = pd.DataFrame({"High": [2, 3, 2], "Low": [1, 2, 1]})
    result = harmonic_pattern_agent(df)
    assert result["side"] == "WAIT"
    assert result["pattern"] == "None"
