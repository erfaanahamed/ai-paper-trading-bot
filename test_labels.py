import pandas as pd
from trading_engine.labels import triple_barrier_labels

def test_same_bar_sl_wins():
 d=pd.DataFrame({'Close':[100,100,100],'High':[101,103,100],'Low':[99,97,99],'ATR':[1,1,1]});r=triple_barrier_labels(d,pd.Series([1,0,0]),2,1,2);assert r.label.iloc[0]==-1 and r.ret_atr.iloc[0]==-1

def test_no_signal_nan():
 d=pd.DataFrame({'Close':[100,100],'High':[101,101],'Low':[99,99],'ATR':[1,1]});r=triple_barrier_labels(d,pd.Series([0,0]));assert r.isna().all().all()
