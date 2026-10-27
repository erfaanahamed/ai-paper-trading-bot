import numpy as np
from sklearn.linear_model import LogisticRegression
from trading_engine.validation import purged_walk_forward_splits,evaluate_oos

def test_splits_embargo():
 for tr,te in purged_walk_forward_splits(1000,5,20,.4):assert tr.max()<te.min()-20

def test_known_signal_auc():
 rng=np.random.default_rng(5);X=rng.normal(size=(2000,3));y=((X[:,0]+rng.normal(size=2000)*.3)>0).astype(int);r=evaluate_oos(LogisticRegression(),X,y,purged_walk_forward_splits(len(y),5,20,.4));assert r['auc']>.6
