import numpy as np,pandas as pd
from trading_engine.data import validate_ohlcv

def frame(n=260):
 idx=pd.date_range(end=pd.Timestamp.now(tz='UTC'),periods=n,freq='15min'); c=np.linspace(100,120,n)+np.sin(np.arange(n));return pd.DataFrame({'Open':c,'High':c+1,'Low':c-1,'Close':c,'Volume':100},index=idx)
def test_frozen_rejected():
 d=frame();d.iloc[-20:,d.columns.get_loc('Close')]=5;ok,msg=validate_ohlcv(d,'15m');assert not ok

def test_stale_rejected():
 d=frame();d.index=d.index-pd.Timedelta(days=1);ok,msg=validate_ohlcv(d,'15m');assert not ok

def test_valid():
 ok,msg=validate_ohlcv(frame(),'15m');assert ok
