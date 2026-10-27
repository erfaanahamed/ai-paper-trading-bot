import numpy as np,pandas as pd
from trading_engine.backtest import run_backtest

def test_backtest_returns_structure():
 n=300;i=pd.date_range('2025-01-01',periods=n,freq='15min',tz='UTC');c=100+np.arange(n)*.01+np.sin(np.arange(n));d=pd.DataFrame({'Open':c,'High':c+1,'Low':c-1,'Close':c,'Volume':100,'ATR':1},index=i);r=run_backtest(d,None,'EUR/USD','15m',{});assert 'stats' in r and 'error' in r
