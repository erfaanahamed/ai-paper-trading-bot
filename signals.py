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
    # 0 Trend following - loosened ADX + EMA stack
    if all(np.isfinite(x) for x in [e9,e21,e50,pdi,mdi,adx]):
        if e9>e21>e50 and pdi>mdi and adx>=16 and bias!="Bearish":v[STRATEGIES[0]]="BUY"
        elif e9<e21<e50 and mdi>pdi and adx>=16 and bias!="Bullish":v[STRATEGIES[0]]="SELL"
    # 1 Mean reversion - slightly wider RSI / BB
    if all(np.isfinite(x) for x in [c,low,up,rsi,adx]):
        if c<=low and rsi<=32 and adx<30 and (not np.isfinite(mfi) or mfi<35):v[STRATEGIES[1]]="BUY"
        elif c>=up and rsi>=68 and adx<30 and (not np.isfinite(mfi) or mfi>65):v[STRATEGIES[1]]="SELL"
    # 2 Breakout
    if all(np.isfinite(x) for x in [c,sup,res,atr]):
        if c>res and (not np.isfinite(rv) or rv>=1.05):v[STRATEGIES[2]]="BUY"
        elif c<sup and (not np.isfinite(rv) or rv>=1.05):v[STRATEGIES[2]]="SELL"
    # 3 Momentum - wider RSI band + DI confirmation
    if all(np.isfinite(x) for x in [c,prev,rsi,pdi,mdi]):
        if c>prev and 52<=rsi<=80 and pdi>mdi:v[STRATEGIES[3]]="BUY"
        elif c<prev and 20<=rsi<=48 and mdi>pdi:v[STRATEGIES[3]]="SELL"
    # 4 Range trading
    if all(np.isfinite(x) for x in [c,sup,res,adx]) and res>sup and adx<22:
        if c<=sup+.22*(res-sup):v[STRATEGIES[4]]="BUY"
        elif c>=res-.22*(res-sup):v[STRATEGIES[4]]="SELL"
    # 5 Volatility expansion
    if all(np.isfinite(x) for x in [c,o,h,l,atr]) and atr>0 and h-l>=1.15*atr and c!=o:
        if c>o and (not np.isfinite(rv) or rv>=0.95):v[STRATEGIES[5]]="BUY"
        elif c<o and (not np.isfinite(rv) or rv>=0.95):v[STRATEGIES[5]]="SELL"
    # 6 Price action
    if all(np.isfinite(x) for x in [c,o,h,l,e21]) and h>l:
        body=abs(c-o); cp=(c-l)/(h-l)
        if c>o and body/(h-l)>=.55 and cp>=.75 and c>e21:v[STRATEGIES[6]]="BUY"
        elif c<o and body/(h-l)>=.55 and cp<=.25 and c<e21:v[STRATEGIES[6]]="SELL"
    # 7 Liquidity sweep
    if all(np.isfinite(x) for x in [c,h,l,ps,pr]):
        if l<ps and c>ps:v[STRATEGIES[7]]="BUY"
        elif h>pr and c<pr:v[STRATEGIES[7]]="SELL"
    b=sum(x=="BUY" for x in v.values()); s=sum(x=="SELL" for x in v.values())
    # Improved decision: require 2+ agreement OR a strong single high-quality strategy in clear regime
    side = "WAIT"
    if b>=2 and b>s:
        side = "BUY"
    elif s>=2 and s>b:
        side = "SELL"
    elif b==1 and s==0 and v.get(STRATEGIES[0])=="BUY" and adx is not None and adx>=22:
        side = "BUY"   # strong trend only
    elif s==1 and b==0 and v.get(STRATEGIES[0])=="SELL" and adx is not None and adx>=22:
        side = "SELL"
    return side,int(round(100*max(b,s)/max(1,len(v)))),v,_regime(last),b,s
def adaptive_strategy_selection(votes,regime,market_name,timeframe,history):
    """Walk-forward-validated weighted voting from completed trade history.

    Weights remain regime-prior based until each strategy has enough outcomes to
    train on an older window and pass a newer chronological holdout window.
    """
    priors={"TRENDING":["Trend following","Momentum","Breakout","Price action","Volatility expansion"],
            "RANGING":["Mean reversion","Range trading","Liquidity sweep (proxy)"],
            "SQUEEZE / QUIET":["Breakout","Volatility expansion","Range trading"],
            "MIXED":["Trend following","Breakout","Mean reversion","Momentum","Price action","Range trading"]}.get(regime,STRATEGIES)
    df=pd.DataFrame(history or [])
    outcomes={}
    if not df.empty and "strategy_names" in df.columns and "net_pnl" in df.columns:
        if "market" in df: df=df[df.market.astype(str)==str(market_name)]
        if "timeframe" in df: df=df[df.timeframe.astype(str)==str(timeframe)]
        time_col=next((c for c in ["closed_at","time","timestamp","closed"] if c in df.columns),None)
        if time_col:
            df=df.assign(_time=pd.to_datetime(df[time_col],errors="coerce",utc=True)).sort_values("_time",kind="stable")
        df["_pnl"]=pd.to_numeric(df["net_pnl"],errors="coerce")
        for _,row in df.dropna(subset=["_pnl"]).iterrows():
            for name in [x.strip() for x in str(row.get("strategy_names","")).split("; ") if x.strip() and x.strip()!="nan"]:
                outcomes.setdefault(name,[]).append(float(row["_pnl"]))
    candidates=list(dict.fromkeys(list(votes.keys())+priors))
    weights={}; validated=[]
    for name in candidates:
        vals=outcomes.get(name,[]); base=1.15 if name in priors else .85
        if len(vals)<20:
            weights[name]=base; continue
        split=int(len(vals)*.70); train=np.asarray(vals[:split],dtype=float); hold=np.asarray(vals[split:],dtype=float)
        if len(train)<12 or len(hold)<5 or not np.isfinite(train).all() or not np.isfinite(hold).all():
            weights[name]=base; continue
        wr=(np.sum(train>0)+2.5)/(len(train)+5.0); edge=float(np.mean(train))
        if float(np.mean(hold))>0 and float(np.mean(hold>0))>=.5 and edge>0:
            weights[name]=float(np.clip(base+4*(max(0.,wr-.5)+.25*np.tanh(edge)),.5,2.5)); validated.append(name)
        else: weights[name]=max(.5,base*.75)
    total=sum(weights.values()) or 1.
    buy_score=100*sum(weights.get(n,1.) for n,v in votes.items() if v=="BUY")/total
    sell_score=100*sum(weights.get(n,1.) for n,v in votes.items() if v=="SELL")/total
    b=sum(v=="BUY" for v in votes.values()); se=sum(v=="SELL" for v in votes.values())
    side="BUY" if b>=2 and buy_score>=60 and buy_score>sell_score else "SELL" if se>=2 and sell_score>=60 and sell_score>buy_score else "WAIT"
    selected=sorted(candidates,key=lambda n:weights.get(n,1.),reverse=True)[:3]
    note="Walk-forward validated weighted voting" if validated else "Walk-forward warm-up: regime priors used until enough validated outcomes"
    return {"side":side,"selected":selected,"votes":votes,"regime":regime,"note":note,"weights":weights,"buy_score":round(buy_score,1),"sell_score":round(sell_score,1),"validated":validated}

def primary_signal(df,regime,market,tf,history):
    if df is None or df.empty:return {"side":"WAIT","score":0,"votes":{},"selected":[],"regime":regime,"note":"no data"}
    r=df.iloc[-1]; side,score,votes,detected,*_=evaluate_multi_strategy(r)
    chosen=adaptive_strategy_selection(votes,regime or detected,market,tf,history)
    weighted_score=max(chosen.get("buy_score",0),chosen.get("sell_score",0))
    return {"side":chosen['side'],"score":weighted_score,"votes":votes,"selected":chosen['selected'],"regime":regime or detected,"note":chosen['note'],"weights":chosen.get("weights",{}),"buy_score":chosen.get("buy_score",0),"sell_score":chosen.get("sell_score",0)}
def primary_signals_batch(df,market,tf,history):
    out=[]
    for i in range(len(df)):
        row=df.iloc[:i+1]; r=row.iloc[-1]; sig=primary_signal(row,_regime(r),market,tf,history);out.append(1 if sig['side']=='BUY' else -1 if sig['side']=='SELL' else 0)
    return pd.Series(out,index=df.index,dtype=int)
