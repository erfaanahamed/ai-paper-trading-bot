import numpy as np
import pandas as pd
from trading_engine.smc import add_smc_features, smc_signal


def sample_ohlc(n=40):
    idx = pd.date_range('2026-01-01', periods=n, freq='5min')
    close = np.linspace(100, 102, n)
    return pd.DataFrame({
        'Open': close - 0.1,
        'High': close + 0.25,
        'Low': close - 0.25,
        'Close': close,
        'Volume': np.full(n, 1000),
    }, index=idx)


def test_smc_features_created_without_lookahead_errors():
    frame = add_smc_features(sample_ohlc())
    for col in ['SMC_BOS','SMC_CHOCH','SMC_FVG','SMC_OB','SMC_OB_Top','SMC_OB_Bottom','SMC_OB_Retest','SMC_LiquiditySweep']:
        assert col in frame.columns
    assert set(frame['SMC_BOS'].dropna().unique()).issubset({-1, 0, 1})
    assert set(frame['SMC_CHOCH'].dropna().unique()).issubset({-1, 0, 1})


def test_smc_signal_waits_without_confluence():
    assert smc_signal({'SMC_OB_Retest': 0, 'SMC_LiquiditySweep': 0, 'SMC_CHOCH': 0,
                       'SMC_BOS': 0, 'SMC_FVG': 0, 'SMC_OB': 0}) == 'WAIT'


def test_smc_signal_requires_confluence():
    assert smc_signal({'SMC_OB_Retest': 1, 'SMC_LiquiditySweep': 0, 'SMC_CHOCH': 0,
                       'SMC_BOS': 0, 'SMC_FVG': 0, 'SMC_OB': 0}) == 'BUY'
    assert smc_signal({'SMC_OB_Retest': -1, 'SMC_LiquiditySweep': 0, 'SMC_CHOCH': 0,
                       'SMC_BOS': 0, 'SMC_FVG': 0, 'SMC_OB': 0}) == 'SELL'
