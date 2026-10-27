import numpy as np
from .config import LEVERAGE_CAP
def kelly_fraction(p_win,rr,fraction=.25,cap=.02):
    full=(rr*p_win-(1-p_win))/rr if rr>0 else -1
    return float(min(cap,max(0,full*fraction)))
def drawdown_throttle(balance,peak,soft=.05,hard=.15,min_scale=.25):
    if peak<=0:return 0.0
    dd=max(0,(peak-balance)/peak)
    if dd>=hard:return 0.0
    if dd<=soft:return 1.0
    return float(1-(1-min_scale)*(dd-soft)/(hard-soft))
def correlation_guard(open_positions,new_market,new_side,returns_cache,max_rho=.7,max_same_dir=3,max_positions=5):
    if len(open_positions)>=max_positions:return False,"max positions"
    same=0
    for p in open_positions:
        if p.get('side',p.get('Side'))==new_side:same+=1
        a=returns_cache.get(new_market);b=returns_cache.get(p.get('market',p.get('Market')))
        if a is not None and b is not None:
            joined=__import__('pandas').concat([a.rename('a'),b.rename('b')],axis=1).dropna()
            if len(joined)>=30 and joined.a.corr(joined.b)>max_rho and p.get('side',p.get('Side'))==new_side:return False,"correlated same-direction exposure"
    if same>=max_same_dir:return False,"same-direction position cap"
    return True,"ok"
def size_position(balance,price,atr,p_win,rr,peak,open_positions,market,config=None):
    config=config or {}; ffull=(rr*p_win-(1-p_win))/rr if rr>0 else -1
    if ffull<=0 or atr<=0 or price<=0:return 0.0
    throttle=drawdown_throttle(balance,peak); risk=min(.25*ffull*throttle,.02)
    if risk<.001:return 0.0
    kind=config.get('kind','fx'); max_notional=balance*LEVERAGE_CAP.get(kind,2); units=min(balance*risk/atr,max_notional/price)
    return float(max(0,units))
