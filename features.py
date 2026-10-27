import numpy as np, pandas as pd
FEATURES=["fret_1","fret_5","fret_20","ffd_close_04","ffd_vol_04","rsi","mfi","macd_hist","adx","di_gap","ema_gap_9_21","ema_gap_21_50","bb_z","vwap_z","atr_pct","vol_of_vol","rel_vol","obv_slope","htf_bull","htf_bear","is_buy"]
def fractional_diff(series,d=0.4,threshold=1e-4):
    s=pd.Series(series,dtype=float); weights=[1.0]
    for k in range(1,min(len(s),1000)):
        w=-weights[-1]*(d-k+1)/k
        if abs(w)<threshold: break
        weights.append(w)
    out=np.full(len(s),np.nan); arr=s.to_numpy()
    for i in range(len(weights)-1,len(s)):
        win=arr[i-len(weights)+1:i+1]
        if np.isfinite(win).all():out[i]=np.dot(np.array(weights[::-1]),win)
    return pd.Series(out,index=s.index)
def _rsi(c,n=14):
    d=c.diff(); up=d.clip(lower=0).ewm(alpha=1/n,min_periods=n,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    rs=up/dn.replace(0,np.nan); return (100-100/(1+rs)).where(dn.ne(0),100).where(up.ne(0),0)
def build_features(df:pd.DataFrame,htf_bias:pd.Series=None)->pd.DataFrame:
    c=df.Close.astype(float); h=df.High.astype(float); l=df.Low.astype(float); v=df.Volume.astype(float).fillna(0); sig=c.pct_change().rolling(20,min_periods=20).std()
    out=pd.DataFrame(index=df.index)
    for n in (1,5,20):out[f"fret_{n}"]=(c/c.shift(n)-1)/(sig*np.sqrt(n))
    out["ffd_close_04"]=fractional_diff(c,.4); out["ffd_vol_04"]=fractional_diff(v+1,.4)
    out["rsi"]=_rsi(c)
    tp=(h+l+c)/3; mf=tp*v; pos=mf.where(tp.diff()>0,0).rolling(14).sum(); neg=mf.where(tp.diff()<0,0).rolling(14).sum()
    out["mfi"]=100-100/(1+pos/neg.replace(0,np.nan)); out.loc[v.rolling(20).sum().fillna(0).eq(0),"mfi"]=np.nan
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean(); macd=e12-e26; out["macd_hist"]=(macd-macd.ewm(span=9,adjust=False).mean())/(sig*c)
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1); atr=tr.rolling(14).mean(); plus=h.diff().clip(lower=0); minus=(-l.diff()).clip(lower=0)
    pdi=100*plus.rolling(14).mean()/atr.replace(0,np.nan); mdi=100*minus.rolling(14).mean()/atr.replace(0,np.nan); dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    out["adx"]=dx.rolling(14).mean(); out["di_gap"]=(pdi-mdi)/100
    e9=c.ewm(span=9,adjust=False).mean(); e21=c.ewm(span=21,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean()
    out["ema_gap_9_21"]=(e9-e21)/atr.replace(0,np.nan); out["ema_gap_21_50"]=(e21-e50)/atr.replace(0,np.nan)
    mid=c.rolling(20).mean(); sd=c.rolling(20).std(); up=mid+2.5*sd; lo=mid-2.5*sd; out["bb_z"]=(c-mid)/((up-lo)/5).replace(0,np.nan)
    volsum=v.rolling(20).sum(); vwap=(tp*v).rolling(20).sum()/volsum.replace(0,np.nan); out["vwap_z"]=(c-vwap)/atr.replace(0,np.nan)
    out["atr_pct"]=atr/c.replace(0,np.nan); out["vol_of_vol"]=sig.rolling(20).std(); out["rel_vol"]=v/v.rolling(20).mean().replace(0,np.nan)
    obv=(np.sign(c.diff()).fillna(0)*v).cumsum(); out["obv_slope"]=obv.diff(10)/v.rolling(10).sum().replace(0,np.nan)
    if htf_bias is None:out["htf_bull"]=0.;out["htf_bear"]=0.
    else:
        b=htf_bias.reindex(df.index).ffill(); out["htf_bull"]=(b=="Bullish").astype(float); out["htf_bear"]=(b=="Bearish").astype(float)
    out["is_buy"]=np.nan
    return out[FEATURES].replace([np.inf,-np.inf],np.nan)
