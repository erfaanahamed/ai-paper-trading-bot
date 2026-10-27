import numpy as np,pandas as pd
from trading_engine.features import build_features,FEATURES

def df(n=400):
 i=pd.date_range('2024-01-01',periods=n,freq='15min',tz='UTC');c=100*np.exp(np.cumsum(np.random.default_rng(2).normal(0,.001,n)));return pd.DataFrame({'Open':c,'High':c*1.002,'Low':c*.998,'Close':c,'Volume':np.ones(n)*100},index=i)
def test_features_shape_and_no_inf():
 f=build_features(df());assert list(f.columns)==FEATURES and not np.isinf(f.to_numpy(dtype=float)).any()
def test_no_lookahead():
 d=df();a=build_features(d);d2=d.copy();d2.iloc[-1,d2.columns.get_loc('Close')]*=1.1;b=build_features(d2);pd.testing.assert_series_equal(a.iloc[-5],b.iloc[-5])
def test_zero_volume_mfi_vwap_nan():
 d=df();d['Volume']=0;f=build_features(d);assert f.mfi.isna().all() and f.vwap_z.isna().all()
