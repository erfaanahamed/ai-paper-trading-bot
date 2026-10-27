"""Causal OHLCV indicators for the hybrid rule + machine-learning brain."""
import numpy as np
import pandas as pd

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()
    x = df.copy()
    c, h, l, o = (x[k].astype(float) for k in ("Close", "High", "Low", "Open"))
    v = x["Volume"].astype(float).fillna(0).clip(lower=0) if "Volume" in x else pd.Series(0., index=x.index)
    prev = c.shift(1)
    tr = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    x["ATR"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    x["RSI"] = (100 - 100/(1 + up/dn.replace(0, np.nan))).where(dn.ne(0), 100).where(up.ne(0) | dn.ne(0), 50)
    typical = (h+l+c)/3
    mf = typical*v
    pos = mf.where(typical.diff()>0, 0).rolling(14, min_periods=14).sum()
    neg = mf.where(typical.diff()<0, 0).rolling(14, min_periods=14).sum()
    x["MFI"] = 100 - 100/(1 + pos/neg.replace(0, np.nan))
    if v.sum() <= 0:
        x["MFI"] = np.nan
    x["EMA9"] = c.ewm(span=9, adjust=False).mean()
    x["EMA21"] = c.ewm(span=21, adjust=False).mean()
    x["EMA50"] = c.ewm(span=50, adjust=False).mean()
    x["EMA200"] = c.ewm(span=200, adjust=False).mean()
    p_dm = h.diff().where((h.diff()>-l.diff()) & (h.diff()>0), 0.)
    m_dm = (-l.diff()).where(((-l.diff())>h.diff()) & ((-l.diff())>0), 0.)
    atr_safe = x["ATR"].replace(0, np.nan)
    pdi = 100*p_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean()/atr_safe
    mdi = 100*m_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean()/atr_safe
    x["PLUS_DI"], x["MINUS_DI"] = pdi, mdi
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0, np.nan)
    x["ADX"] = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    x["BB_Mid"] = c.rolling(20, min_periods=20).mean()
    sd = c.rolling(20, min_periods=20).std(ddof=0)
    x["BB_Upper"], x["BB_Lower"] = x["BB_Mid"]+2.5*sd, x["BB_Mid"]-2.5*sd
    session = pd.Series(pd.to_datetime(x.index).date, index=x.index)
    den = v.groupby(session).cumsum().replace(0, np.nan)
    x["VWAP"] = (typical*v).groupby(session).cumsum()/den
    x["VWAP"] = x["VWAP"].fillna(x["BB_Mid"])
    x["VolAvg20"] = v.rolling(20, min_periods=20).mean()
    x["RelVol"] = v/x["VolAvg20"].replace(0, np.nan)
    x["PrevClose"] = c.shift(1)
    x["Support"] = l.rolling(20, min_periods=20).min().shift(1)
    x["Resistance"] = h.rolling(20, min_periods=20).max().shift(1)
    x["PrevSupport"], x["PrevResistance"] = x["Support"].shift(1), x["Resistance"].shift(1)
    return x.replace([np.inf, -np.inf], np.nan)
