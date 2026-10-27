import numpy as np,pandas as pd
def performance_metrics(trades):
 df=pd.DataFrame(trades or [])
 if df.empty:return {"n_trades":"—","win_rate":"—","profit_factor":"—","per_market":pd.DataFrame(),"per_strategy":pd.DataFrame(),"per_regime":pd.DataFrame()}
 pnl=pd.to_numeric(df.get('net_pnl',pd.Series(dtype=float)),errors='coerce'); wins=pnl[pnl>0].sum();loss=-pnl[pnl<0].sum()
 return {"n_trades":len(df),"win_rate":float((pnl>0).mean()),"profit_factor":float(wins/loss) if loss else np.nan,"per_market":df.groupby('market',dropna=False).net_pnl.sum().reset_index() if 'market' in df else pd.DataFrame(),"per_strategy":pd.DataFrame(),"per_regime":df.groupby('regime',dropna=False).net_pnl.sum().reset_index() if 'regime' in df and 'net_pnl' in df else pd.DataFrame()}
