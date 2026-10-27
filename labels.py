import numpy as np,pandas as pd
def triple_barrier_labels(df,primary,tp_mult=2.0,sl_mult=1.0,max_bars=48,atr_col="ATR"):
    out=pd.DataFrame(index=df.index,columns=["label","t_touch","ret_atr","side"]);out[:]=np.nan
    for i,side in enumerate(pd.Series(primary,index=df.index).to_numpy()):
        if side not in (1,-1):continue
        atr=df[atr_col].iloc[i] if atr_col in df else np.nan;entry=df.Close.iloc[i]
        if not np.isfinite(atr) or atr<=0:continue
        tp=entry+side*tp_mult*atr;sl=entry-side*sl_mult*atr; end=min(i+max_bars,len(df)-1); label=0; touch=end; ret=(df.Close.iloc[end]-entry)/atr*side
        for j in range(i+1,end+1):
            hi,lo=df.High.iloc[j],df.Low.iloc[j]
            tp_hit=hi>=tp if side==1 else lo<=tp; sl_hit=lo<=sl if side==1 else hi>=sl
            if sl_hit and tp_hit:label=-1;touch=j;ret=-sl_mult;break
            if tp_hit:label=1;touch=j;ret=tp_mult;break
            if sl_hit:label=-1;touch=j;ret=-sl_mult;break
        out.iloc[i]=[label,touch,ret,side]
    return out
