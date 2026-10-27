from trading_engine.state import save_state,load_state,save_settings,load_settings,append_trade,log_audit

def test_roundtrip(tmp_path):
 p=tmp_path/'state.json';save_state({'positions':[{'id':'x'}]},p);assert load_state(p)['positions'][0]['id']=='x'
def test_settings(tmp_path):
 p=tmp_path/'settings.json';save_settings({'paused':True},p);assert load_settings(p)['paused'] is True
def test_append(tmp_path):
 p=tmp_path/'trades.csv';append_trade({'id':'1','net_pnl':2},p);append_trade({'id':'2','net_pnl':-1},p);assert p.read_text().count('\n')==3
