import numpy as np
from sklearn.metrics import roc_auc_score,brier_score_loss,log_loss
from sklearn.calibration import calibration_curve
def purged_walk_forward_splits(n,n_splits=5,embargo=20,min_train_pct=.4):
    first=max(100,int(n*min_train_pct)); remain=n-first
    if remain<=0:return
    sizes=np.full(n_splits,remain//n_splits,dtype=int);sizes[:remain%n_splits]+=1; start=first
    for size in sizes:
        end=min(n,start+int(size)); train_end=max(0,start-embargo)
        if train_end>=100 and end>start:yield np.arange(train_end),np.arange(start,end)
        start=end
def evaluate_oos(model,X,y,splits):
    X=np.asarray(X);y=np.asarray(y);p=np.full(len(y),np.nan)
    for tr,te in splits:
        model.fit(X[tr],y[tr]);p[te]=model.predict_proba(X[te])[:,1]
    mask=np.isfinite(p); yy=y[mask];pp=p[mask]; auc=float('nan')
    if len(np.unique(yy))==2 and min(np.sum(yy==0),np.sum(yy==1))>=20:auc=float(roc_auc_score(yy,pp))
    result={"y_true":yy,"y_pred_proba":pp,"auc":auc,"brier":float(brier_score_loss(yy,pp)) if len(yy) else float('nan'),"log_loss":float(log_loss(yy,pp,labels=[0,1])) if len(yy) else float('nan'),"calibration_curve":calibration_curve(yy,pp,n_bins=10) if len(yy) else ([],[])}
    result["tradable"]=bool(np.isfinite(auc) and auc>=.53);result["reason"]="" if result["tradable"] else "auc below threshold";return result
def deflated_sharpe(returns,n_trials):
    x=np.asarray(returns,dtype=float);x=x[np.isfinite(x)]
    if len(x)<2 or np.std(x,ddof=1)==0:return float('nan')
    sr=np.mean(x)/np.std(x,ddof=1)*np.sqrt(252); penalty=np.sqrt(2*np.log(max(1,int(n_trials))));return float(sr-penalty)
