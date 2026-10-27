from trading_engine.gates import expected_value,decide

def test_ev_cost():assert expected_value(.5,2,1.1,.001,5)<0

def test_model_disabled():assert decide('BUY',{}, {'ok':True,'auc':.5},'EUR/USD')['reason']=='model disabled'
