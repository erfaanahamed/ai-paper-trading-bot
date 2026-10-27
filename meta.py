import pickle, numpy as np,pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from .config import ML_FEATURES_V2
from .validation import purged_walk_forward_splits

def fit_meta_model(features,labels,primary,config=None):
    config=config or {}; mask=pd.Series(primary,index=features.index).isin([1,-1]) & labels.label.isin([1,-1]); idx=features.index[mask]
    if len(idx)<300:return {"ok":False,"reason":"insufficient data"}
    cols=[c for c in ML_FEATURES_V2 if c in features.columns and c!="is_buy"]
    X=features.loc[idx,cols].replace([np.inf,-np.inf],np.nan); med=X.median().fillna(0); X=X.fillna(med).to_numpy(); y=(labels.loc[idx,"label"].to_numpy()==pd.Series(primary,index=features.index).loc[idx].to_numpy()).astype(int)
    if min(np.sum(y==0),np.sum(y==1))<50:return {"ok":False,"reason":"insufficient data"}
    n=len(y); weights=np.exp(-(n-1-np.arange(n))*np.log(2)/500); pred=np.full(n,np.nan)
    for tr,te in purged_walk_forward_splits(n,5,max(1,int(config.get("max_bars",48))//2),.4):
        model=HistGradientBoostingClassifier(max_depth=6,max_iter=100,learning_rate=.05,min_samples_leaf=20,l2_regularization=1,random_state=42);model.fit(X[tr],y[tr],sample_weight=weights[tr]);pred[te]=model.predict_proba(X[te])[:,1]
    good=np.isfinite(pred); auc=float(roc_auc_score(y[good],pred[good])) if good.sum() and len(np.unique(y[good]))==2 else float('nan')
    if not np.isfinite(auc) or auc<.53:return {"ok":False,"auc":auc,"reason":"below threshold"}
    base=HistGradientBoostingClassifier(max_depth=6,max_iter=300,learning_rate=.05,min_samples_leaf=40,l2_regularization=1,random_state=42)
    method="isotonic" if n>=2000 else "sigmoid"; model=CalibratedClassifierCV(base,method=method,cv=3);model.fit(X,y,sample_weight=weights)
    return {"ok":True,"model":model,"medians":med.to_dict(),"feature_order":cols,"auc":auc,"sample_count":n,"base_rate":float(y.mean()),"trained_at":pd.Timestamp.now(tz="UTC").isoformat()}
def predict_win_prob(artifact,feature_row,side):
    if not artifact or not artifact.get("ok") or artifact.get("auc",0)<.53:return float('nan')
    cols=artifact["feature_order"]; row=pd.DataFrame([{c:feature_row.get(c,np.nan) for c in cols}]).fillna(artifact["medians"]); row["is_buy"]=1 if side in (1,"BUY") else 0
    return float(artifact["model"].predict_proba(row[cols])[:,1][0])
def save_artifact(artifact,path):
    if artifact and artifact.get("ok") and artifact.get("auc",0)>=.53:
        Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_bytes(pickle.dumps(artifact))
def load_artifact(path):
    try:
        a=pickle.loads(Path(path).read_bytes());return a if a.get("ok") and a.get("auc",0)>=.53 else None
    except (OSError,ValueError,pickle.PickleError,EOFError):return None
