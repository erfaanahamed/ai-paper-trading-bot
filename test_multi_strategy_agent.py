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
