import numpy as np,pandas as pd
STRATEGIES=["Trend following","Mean reversion","Breakout","Momentum","Range trading","Volatility expansion","Price action","Liquidity sweep (proxy)"]
def _regime(r):
    adx=r.get("ADX",np.nan); close=r.get("Close",np.nan); mid=r.get("BB_Mid",np.nan); up=r.get("BB_Upper",np.nan); lo=r.get("BB_Lower",np.nan)
    if pd.notna(adx) and adx>=25:return "TRENDING"
    if pd.notna(up) and pd.notna(lo) and pd.notna(mid) and mid and (up-lo)/abs(mid)<.012:return "SQUEEZE / QUIET"
    if pd.notna(adx) and adx<18:return "RANGING"
    return "MIXED"
def evaluate_multi_strategy(last,bias="Unknown"):
    def n(k):
        try:return float(last.get(k,np.nan))
        except:return np.nan
    c,o,h,l=n("Close"),n("Open"),n("High"),n("Low"); prev=n("PrevClose"); e9,e21,e50=n("EMA9"),n("EMA21"),n("EMA50"); rsi,adx,pdi,mdi=n("RSI"),n("ADX"),n("PLUS_DI"),n("MINUS_DI"); mid,up,low=n("BB_Mid"),n("BB_Upper"),n("BB_Lower"); sup,res,atr,rv=n("Support"),n("Resistance"),n("ATR"),n("RelVol"); mfi=n("MFI"); ps,pr=n("PrevSupport"),n("PrevResistance")
    v={s:"WAIT" for s in STRATEGIES}
    if all(np.isfinite(x) for x in [e9,e21,e50,pdi,mdi,adx]):
        if e9>e21>e50 and pdi>mdi and adx>=18 and bias!="Bearish":v[STRATEGIES[0]]="BUY"
        elif e9<e21<e50 and mdi>pdi and adx>=18 and bias!="Bullish":v[STRATEGIES[0]]="SELL"
    if all(np.isfinite(x) for x in [c,low,up,rsi,adx]):
        if c<=low and rsi<=30 and adx<28 and (not np.isfinite(mfi) or mfi<30):v[STRATEGIES[1]]="BUY"
        elif c>=up and rsi>=70 and adx<28 and (not np.isfinite(mfi) or mfi>70):v[STRATEGIES[1]]="SELL"
    if all(np.isfinite(x) for x in [c,sup,res,atr]):
        if c>res and (not np.isfinite(rv) or rv>=1.1):v[STRATEGIES[2]]="BUY"
        elif c<sup and (not np.isfinite(rv) or rv>=1.1):v[STRATEGIES[2]]="SELL"
    if all(np.isfinite(x) for x in [c,prev,rsi,pdi,mdi]):
        if c>prev and 55<=rsi<=78 and pdi>mdi:v[STRATEGIES[3]]="BUY"
        elif c<prev and 22<=rsi<=45 and mdi>pdi:v[STRATEGIES[3]]="SELL"
    if all(np.isfinite(x) for x in [c,sup,res,adx]) and res>sup and adx<20:
        if c<=sup+.18*(res-sup):v[STRATEGIES[4]]="BUY"
        elif c>=res-.18*(res-sup):v[STRATEGIES[4]]="SELL"
    if all(np.isfinite(x) for x in [c,o,h,l,atr]) and atr>0 and h-l>=1.25*atr and c!=o:
        if c>o and (not np.isfinite(rv) or rv>=1):v[STRATEGIES[5]]="BUY"
        elif c<o and (not np.isfinite(rv) or rv>=1):v[STRATEGIES[5]]="SELL"
    if all(np.isfinite(x) for x in [c,o,h,l,e21]) and h>l:
        body=abs(c-o); cp=(c-l)/(h-l)
        if c>o and body/(h-l)>=.6 and cp>=.8 and c>e21:v[STRATEGIES[6]]="BUY"
        elif c<o and body/(h-l)>=.6 and cp<=.2 and c<e21:v[STRATEGIES[6]]="SELL"
    if all(np.isfinite(x) for x in [c,h,l,ps,pr]):
        if l<ps and c>ps:v[STRATEGIES[7]]="BUY"
        elif h>pr and c<pr:v[STRATEGIES[7]]="SELL"
    b=sum(x=="BUY" for x in v.values()); s=sum(x=="SELL" for x in v.values()); side="BUY" if b>=2 and b>s else "SELL" if s>=2 and s>b else "WAIT"
    return side,int(round(100*max(b,s)/len(v))),v,_regime(last),b,s
def adaptive_strategy_selection(votes,regime,market_name,timeframe,history):
    priors={"TRENDING":["Trend following","Momentum","Breakout","Price action","Volatility expansion"],"RANGING":["Mean reversion","Range trading","Liquidity sweep (proxy)"],"SQUEEZE / QUIET":["Breakout","Volatility expansion","Range trading"],"MIXED":["Trend following","Breakout","Mean reversion","Momentum","Price action","Range trading"]}.get(regime,STRATEGIES)
    perf={}; df=pd.DataFrame(history or [])
    if not df.empty and "strategy_names" in df and "net_pnl" in df:
        if "market" in df:df=df[df.market==market_name]
        if "timeframe" in df:df=df[df.timeframe==timeframe]
        for _,r in df.tail(500).iterrows():
            for name in str(r.strategy_names).split("; "):perf.setdefault(name,[]).append(float(r.net_pnl))
    def rank(s):
        vals=perf.get(s,[])[-30:]; prior=.08 if s in priors else 0
        if len(vals)<5:return (prior,0,len(vals))
        wr=(sum(x>0 for x in vals)+2.5)/(len(vals)+5); avg=np.tanh(np.mean(vals));return (prior+wr-.5+.05*avg,avg,len(vals))
    selected=sorted(votes,key=rank,reverse=True)[:3]; active={s:votes.get(s,"WAIT") for s in selected}; b=sum(v=="BUY" for v in active.values()); se=sum(v=="SELL" for v in active.values()); side="BUY" if b>=2 and b>se else "SELL" if se>=2 and se>b else "WAIT"
    return {"side":side,"selected":selected,"votes":votes,"regime":regime,"note":"performance-adaptive" if any(len(perf.get(x,[]))>=5 for x in selected) else "regime-prior warm-up"}
def primary_signal(df,regime,market,tf,history):
    if df is None or df.empty:return {"side":"WAIT","score":0,"votes":{},"selected":[],"regime":regime,"note":"no data"}
    r=df.iloc[-1]; side,score,votes,detected,*_=evaluate_multi_strategy(r)
    chosen=adaptive_strategy_selection(votes,regime or detected,market,tf,history)
    b=sum(v=="BUY" for v in chosen['votes'].values()); s=sum(v=="SELL" for v in chosen['votes'].values())
    return {"side":chosen['side'],"score":score,"votes":votes,"selected":chosen['selected'],"regime":regime or detected,"note":chosen['note']}
def primary_signals_batch(df,market,tf,history):
    out=[]
    for i in range(len(df)):
        row=df.iloc[:i+1]; r=row.iloc[-1]; sig=primary_signal(row,_regime(r),market,tf,history);out.append(1 if sig['side']=='BUY' else -1 if sig['side']=='SELL' else 0)
    return pd.Series(out,index=df.index,dtype=int)
