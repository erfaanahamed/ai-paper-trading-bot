import functools, logging, time
import numpy as np, pandas as pd
from .config import INTERVAL_SECONDS, MARKETS
log=logging.getLogger(__name__)
@functools.lru_cache(maxsize=256)
def _fetch_cached(symbol, interval, bucket):
    try:
        import yfinance as yf
        period_map={"5m":"5d","15m":"60d","30m":"60d","1h":"730d"}
        df=yf.download(symbol,period=period_map.get(interval,"60d"),interval=interval,progress=False,threads=False,auto_adjust=False)
        if df is None or df.empty: return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
        df=df.rename(columns={str(c).title():str(c).title() for c in df.columns})
        df=df[["Open","High","Low","Close","Volume"]].copy()
        idx=pd.to_datetime(df.index,utc=True,errors="coerce"); df.index=idx; df=df.loc[~df.index.isna()]
        df=df[~df.index.duplicated(keep="last")].sort_index()
        return df
    except Exception as e:
        log.warning("OHLCV fetch failed for %s/%s: %s",symbol,interval,e); return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
def fetch_ohlcv(symbol:str,interval:str)->pd.DataFrame:
    return _fetch_cached(str(symbol),str(interval),int(time.time()//55)).copy()
def bar_age_seconds(df):
    if df is None or df.empty:return float("inf")
    idx=pd.to_datetime(df.index,utc=True); return max(0.0,(pd.Timestamp.now(tz="UTC")-idx[-1]).total_seconds())
def validate_ohlcv(df,interval):
    if df is None or len(df)<250:return False,"length < 250"
    x=df.copy(); idx=pd.to_datetime(x.index,utc=True,errors="coerce")
    if idx.isna().any() or not idx.is_monotonic_increasing or idx.has_duplicates:return False,"invalid index"
    if bar_age_seconds(x)>3*INTERVAL_SECONDS.get(interval,900):return False,"stale feed"
    delta=np.diff(idx.asi8)/1e9
    recent=delta[-100:]; med=np.median(recent) if len(recent) else 0
    if med<=0 or np.sum(recent>1.5*med)>=5:return False,"gappy feed"
    cols=[c for c in ["Open","High","Low","Close"] if c in x]
    if len(cols)<4 or x[cols].tail(20).isna().any().any():return False,"NaN in recent OHLC"
    if x.Close.tail(20).nunique()<=1:return False,"zero variance / frozen feed"
    return True,"ok"
def compute_returns_cache(markets:dict)->dict:
    out={}
    for name,item in markets.items():
        symbol=item.get("symbol",name); interval=item.get("interval","15m")
        df=fetch_ohlcv(symbol,interval)
        if not df.empty: out[name]=df.Close.pct_change().dropna().tail(500)
    return out
