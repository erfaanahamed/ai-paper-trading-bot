import numpy as np
import pandas as pd
from trading_engine.indicators import add_indicators

def test_indicator_builder_adds_required_columns():
    idx = pd.date_range("2025-01-01", periods=300, freq="15min", tz="UTC")
    close = 100 + np.cumsum(np.sin(np.arange(300)/7)*0.2 + 0.03)
    df = pd.DataFrame({"Open":close-0.03,"High":close+0.2,"Low":close-0.2,"Close":close,"Volume":np.arange(300)+100}, index=idx)
    out = add_indicators(df)
    for col in ["ATR","RSI","MFI","ADX","PLUS_DI","MINUS_DI","EMA9","EMA21","EMA50","EMA200","BB_Upper","BB_Lower","VWAP","Support","Resistance"]:
        assert col in out.columns
    assert np.isfinite(out["EMA200"].iloc[-1])
    assert out["Support"].iloc[-1] < out["Resistance"].iloc[-1]

def test_support_resistance_use_prior_bars():
    idx = pd.date_range("2025-01-01", periods=40, freq="15min", tz="UTC")
    close = np.arange(40,dtype=float)+100
    df = pd.DataFrame({"Open":close,"High":close+1,"Low":close-1,"Close":close,"Volume":100},index=idx)
    out = add_indicators(df)
    # Current bar's high/low must not leak into the current support/resistance level.
    assert out["Resistance"].iloc[-1] == df["High"].iloc[-2]
    assert out["Support"].iloc[-1] == df["Low"].iloc[-21:-1].min()
