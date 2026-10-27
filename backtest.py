import numpy as np,pandas as pd
from .features import build_features
from .labels import triple_barrier_labels
from .signals import primary_signals_batch
from .validation import purged_walk_forward_splits,deflated_sharpe
from .costs import total_cost_bps
def run_backtest(df,htf_df,market,timeframe,config):
 try:
  features=build_features(df); primary=primary_signals_batch(df,market,timeframe,[]); labels=triple_barrier_labels(df,primary,config.get('tp_mult',2),config.get('sl_mult',1),config.get('max_bars',48),config.get('atr_col','ATR'))
  trades=[]; eq=[float(config.get('balance',100000))]; balance=eq[0]; cost=total_cost_bps(market)
  for i in range(len(df)-1):
   side=primary.iloc[i]
   if side==0:continue
   atr=float(df.ATR.iloc[i]) if 'ATR' in df else np.nan
   if not np.isfinite(atr) or atr<=0:continue
   entry=float(df.Open.iloc[i+1]); sl=entry-side*atr; tp=entry+side*2*atr; exitp=None;reason='timeout';end=min(len(df)-1,i+config.get('max_bars',48))
   for j in range(i+1,end+1):
    bar=df.iloc[j]; sh=bar.Low<=sl if side==1 else bar.High>=sl; th=bar.High>=tp if side==1 else bar.Low<=tp
    if sh:exitp=sl;reason='sl';break
    if th:exitp=tp;reason='tp';break
   if exitp is None:exitp=float(df.Close.iloc[end])
   qty=1.; gross=(exitp-entry)*side*qty; costs=(entry+exitp)*cost/20000*qty; net=gross-costs;balance+=net;eq.append(balance);trades.append({"market":market,"timeframe":timeframe,"entry_index":i+1,"exit_index":j,"side":"BUY" if side==1 else "SELL","entry":entry,"exit":exitp,"gross_pnl":gross,"costs":costs,"net_pnl":net,"exit_reason":reason})
  td=pd.DataFrame(trades);pn=td.net_pnl.to_numpy() if len(td) else np.array([]);wins=pn[pn>0].sum();loss=-pn[pn<0].sum();curve=pd.Series(eq)
  ret=curve.pct_change().dropna();peak=curve.cummax();dd=((peak-curve)/peak).max() if len(curve) else 0
  stats={"n_trades":len(td),"win_rate":float(np.mean(pn>0)) if len(pn) else np.nan,"profit_factor":float(wins/loss) if loss else np.nan,"sharpe":float(ret.mean()/ret.std(ddof=1)*np.sqrt(252)) if len(ret)>1 and ret.std(ddof=1)>0 else np.nan,"sortino":float(ret.mean()/ret[ret<0].std(ddof=1)*np.sqrt(252)) if len(ret)>1 and len(ret[ret<0])>1 and ret[ret<0].std(ddof=1)>0 else np.nan,"max_dd":float(dd),"avg_pnl":float(np.mean(pn)) if len(pn) else np.nan,"oos_auc":np.nan,"deflated_sharpe":deflated_sharpe(pn,8)}
  return {"trades":td,"equity_curve":curve,"stats":stats,"splits":list(purged_walk_forward_splits(len(df),5,config.get('max_bars',48)//2,.4)),"error":None}
 except Exception as e:return {"trades":pd.DataFrame(),"equity_curve":pd.Series(dtype=float),"stats":{},"splits":[],"error":str(e)}
