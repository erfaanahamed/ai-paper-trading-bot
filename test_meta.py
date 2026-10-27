from trading_engine.meta import load_artifact,save_artifact

def test_missing_artifact_none(tmp_path):assert load_artifact(tmp_path/'none.pkl') is None

def test_reject_bad_auc(tmp_path):
 p=tmp_path/'x.pkl';save_artifact({'ok':True,'auc':.5},p);assert not p.exists()
