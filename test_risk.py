from trading_engine.risk import kelly_fraction,drawdown_throttle,size_position

def test_negative_kelly():assert kelly_fraction(.3,2)==0

def test_hard_drawdown():assert drawdown_throttle(80,100)==0

def test_leverage_cap():assert size_position(1000,100,1,.8,2,1000,[],'BTC/USD',{'kind':'crypto'})<=20
