import pandas as pd
import trading_engine.runner as runner

def test_runner_returns_list(monkeypatch, tmp_path):
    idx=pd.date_range(end=pd.Timestamp.now(tz='UTC'),periods=260,freq='15min')
    c=pd.Series(range(100,360),index=idx,dtype=float)
    df=pd.DataFrame({'Open':c,'High':c+1,'Low':c-1,'Close':c,'Volume':100},index=idx)
    monkeypatch.setattr(runner,'fetch_ohlcv',lambda symbol,interval:df)
    monkeypatch.setattr(runner,'validate_ohlcv',lambda df,interval:(True,'ok'))
    monkeypatch.setattr(runner,'load_settings',lambda:{'paused':False,'symbol_enabled':{}})
    monkeypatch.setattr(runner,'load_state',lambda:{'positions':[]})
    r=runner.scan_all_markets({'markets':{}})
    assert isinstance(r,list) and len(r)>0
