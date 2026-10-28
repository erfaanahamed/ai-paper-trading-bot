from pathlib import Path
DATA_DIR = Path("trading_engine_data")
MARKETS = {
 "Gold (XAUUSD)":{"symbol":"GC=F","kind":"gold","interval":"15m"}, "XAGUSD Silver (futures proxy)":{"symbol":"SI=F","kind":"silver","interval":"15m"}, "Crude Oil (WTI proxy)":{"symbol":"CL=F","kind":"oil","interval":"15m"},
 "EUR/USD":{"symbol":"EURUSD=X","kind":"fx","interval":"15m"}, "GBP/USD":{"symbol":"GBPUSD=X","kind":"fx","interval":"15m"}, "AUD/USD":{"symbol":"AUDUSD=X","kind":"fx","interval":"15m"},
 "USD/JPY":{"symbol":"JPY=X","kind":"fx","interval":"15m"}, "BTC/USD":{"symbol":"BTC-USD","kind":"crypto","interval":"15m"}, "SOL/USD":{"symbol":"SOL-USD","kind":"crypto","interval":"15m"}, "ETH/USD":{"symbol":"ETH-USD","kind":"crypto","interval":"15m"}}
INTERVAL_SECONDS={"5m":300,"15m":900,"30m":1800,"1h":3600}
COST_BPS={"EUR/USD":1.0,"USD/JPY":1.0,"GBP/USD":1.5,"AUD/USD":1.5,"Gold (XAUUSD)":2.5,"XAGUSD Silver (futures proxy)":3.0,"Crude Oil (WTI proxy)":3.0,"BTC/USD":8.0,"ETH/USD":10.0,"SOL/USD":12.0}
SLIPPAGE_BPS={k:v/2 for k,v in COST_BPS.items()}
LEVERAGE_CAP={"fx":10,"gold":5,"oil":5,"crypto":2}
DEFAULT_SETTINGS={"risk_pct":0.005,"rr2":2.0,"max_daily_loss":0.03,"max_trades_per_day":20,"thresholds":{},"symbol_enabled":{m:True for m in MARKETS},"paused":False,"session_protection":True,"session_buffer_minutes":30,"no_weekend_trading":True}
ML_FEATURES_V2=["fret_1","fret_5","fret_20","ffd_close_04","ffd_vol_04","rsi","mfi","macd_hist","adx","di_gap","ema_gap_9_21","ema_gap_21_50","bb_z","vwap_z","atr_pct","vol_of_vol","rel_vol","obv_slope","htf_bull","htf_bear","is_buy"]
