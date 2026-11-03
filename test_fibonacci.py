import numpy as np
import pandas as pd
from trading_engine.fibonacci import fibonacci_analysis

def test_fibonacci_returns_retracements_extensions_and_zone():
    n=80
    close=np.linspace(100, 120, n)
    df=pd.DataFrame({"High":close+1, "Low":close-1, "Close":close, "ATR":np.full(n, 1.0)})
    result=fibonacci_analysis(df)
    assert result["status"] == "ANALYZED"
    assert "R0.618" in result["levels"]
    assert "E1.618" in result["extensions"]
    assert result["zone"]["high"] > result["zone"]["low"]

def test_fibonacci_handles_insufficient_data():
    df=pd.DataFrame({"High":[1], "Low":[0], "Close":[.5]})
    assert fibonacci_analysis(df)["status"] == "NO DATA"
