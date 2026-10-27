from trading_engine.execution import manage_bar,net_pnl

def pos():return {'side':'BUY','entry':100,'stop':99,'tp1':101,'tp2':102,'qty':10,'bars_held':0}
def test_stop_wins_same_bar():
 r=manage_bar(pos(),{'Open':100,'High':102,'Low':98,'Close':100});assert r['reason']=='sl_same_bar' and r['price']==99

def test_tp1_partial_moves_stop():
 r=manage_bar(pos(),{'Open':100,'High':101.5,'Low':99.5,'Close':101});assert r['action']=='PARTIAL' and r['fraction']==.6 and r['position']['stop']==100

def test_costs_subtracted():assert net_pnl(pos(),101,5)<10
