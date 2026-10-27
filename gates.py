import numpy as np
from .meta import predict_win_prob
from .costs import total_cost_bps
def expected_value(p_win,rr,price,atr,cost_bps):
    if not all(np.isfinite(x) for x in [p_win,rr,price,atr,cost_bps]) or atr<=0:return float('-inf')
    return float(p_win*rr-(1-p_win)-(price*cost_bps/10000)/atr)
def dynamic_threshold(history=None,drawdown_scale=1.0,trade_count=0):
    vals=list(history or [])[-20:]
    if drawdown_scale<.5:return .60
    if trade_count<100:return .58
    if len(vals)>=20 and np.mean(vals)<.4:return .62
    if len(vals)>=20 and np.mean(vals)>.65:return .52
    return .55
def decide(side,feature_row,artifact,market,config=None):
    config=config or {}
    if side not in ('BUY','SELL'):return {"action":"SKIP","reason":"no primary signal"}
    if not artifact or artifact.get('auc',0)<.53:return {"action":"SKIP","reason":"model disabled"}
    p=predict_win_prob(artifact,feature_row,side); threshold=float(config.get('threshold',.55))
    if not np.isfinite(p):return {"action":"SKIP","reason":"model probability unavailable"}
    if p<threshold:return {"action":"SKIP","reason":f"p_win {p:.2f} < {threshold:.2f}","p_win":p}
    ev=expected_value(p,2.0,float(feature_row.get('Close',np.nan)),float(feature_row.get('ATR',np.nan)),total_cost_bps(market))
    if ev<=0:return {"action":"SKIP","reason":f"EV {ev:.4f} <= 0","p_win":p,"ev":ev}
    return {"action":"TRADE","reason":"p_win and EV passed","p_win":p,"ev":ev}
