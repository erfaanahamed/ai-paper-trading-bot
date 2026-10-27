import numpy as np
def estimate_fill_price(signal_price,side,cost_bps,open_next_bar):
    base=float(open_next_bar); half=float(cost_bps)/20000
    return base*(1+half if side in ('BUY',1) else 1-half)
def manage_bar(position,bar):
    p=dict(position); side=p.get('side',p.get('Side','BUY')); buy=side in ('BUY',1); op=float(bar['Open']);hi=float(bar['High']);lo=float(bar['Low']);cl=float(bar['Close']); stop=float(p.get('stop',p.get('Stop')));entry=float(p.get('entry',p.get('Entry')));tp1=float(p.get('tp1',p.get('TP1')));tp2=float(p.get('tp2',p.get('TP2'))); partial=bool(p.get('tp1_hit',p.get('TP1_Hit',False)))
    if (buy and op<=stop) or (not buy and op>=stop):return {"action":"CLOSE","price":op,"reason":"gap_sl","fraction":1.0,"position":p}
    target=tp2 if partial else tp1
    if (buy and op>=target) or (not buy and op<=target):
        if partial:return {"action":"CLOSE","price":op,"reason":"gap_tp","fraction":1.0,"position":p}
    sl_hit=lo<=stop if buy else hi>=stop; tp_hit=hi>=tp1 if buy else lo<=tp1
    if sl_hit and tp_hit and not partial:return {"action":"CLOSE","price":stop,"reason":"sl_same_bar","fraction":1.0,"position":p}
    if sl_hit:return {"action":"CLOSE","price":stop,"reason":"sl","fraction":1.0,"position":p}
    if not partial and tp_hit:
        p['tp1_hit']=True;p['stop']=entry;p['TP1_Hit']=True;p['Stop']=entry
        return {"action":"PARTIAL","price":tp1,"reason":"tp1","fraction":.6,"position":p}
    tp2_hit=hi>=tp2 if buy else lo<=tp2
    if partial and tp2_hit:return {"action":"CLOSE","price":tp2,"reason":"tp2","fraction":1.0,"position":p}
    if int(p.get('bars_held',p.get('BarsHeld',0)))>=int(p.get('max_bars',48)):return {"action":"CLOSE","price":cl,"reason":"timeout","fraction":1.0,"position":p}
    return {"action":"HOLD","position":p}
def apply_timeout(position,bars_held,max_bars):
    return {"action":"CLOSE" if bars_held>=max_bars else "HOLD","reason":"timeout" if bars_held>=max_bars else "within_time"}
def net_pnl(position,exit_price,cost_bps):
    side=position.get('side',position.get('Side','BUY')); entry=float(position.get('entry',position.get('Entry')));qty=float(position.get('qty',position.get('Qty',0))); sign=1 if side in ('BUY',1) else -1
    gross=(float(exit_price)-entry)*sign*qty; costs=(entry+float(exit_price))*cost_bps/20000*qty
    return gross-costs
