import csv,json,logging,os,tempfile
from pathlib import Path
from datetime import datetime,timezone
from .config import DATA_DIR,DEFAULT_SETTINGS
log=logging.getLogger(__name__)
def _atomic(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(data,encoding='utf-8');os.replace(tmp,path)
def save_state(state=None,path=None):
 path=Path(path or DATA_DIR/'paper_open_positions.json');obj=state or {};obj=dict(obj);obj['saved_at']=datetime.now(timezone.utc).isoformat();_atomic(path,json.dumps(obj,default=str));return obj
def load_state(path=None):
 try:return json.loads(Path(path or DATA_DIR/'paper_open_positions.json').read_text())
 except FileNotFoundError:return {"positions":[],"daily_state":{}}
 except Exception as e:log.warning('state load failed: %s',e);return {"positions":[],"daily_state":{}}
def append_trade(record,path=None):
 path=Path(path or DATA_DIR/'paper_trade_history.csv');path.parent.mkdir(parents=True,exist_ok=True);keys=list(record)
 # single-process append is safe; use OS append mode. Multi-process callers should serialize at service level.
 exists=path.exists() and path.stat().st_size>0
 with path.open('a',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=keys); 
  if not exists:w.writeheader()
  w.writerow(record);f.flush();os.fsync(f.fileno())
def log_audit(event_type,details,path=None):
 path=Path(path or DATA_DIR/'audit_log.jsonl');path.parent.mkdir(parents=True,exist_ok=True);line=json.dumps({"timestamp":datetime.now(timezone.utc).isoformat(),"event_type":event_type,"details":details},default=str)+'\n'
 with path.open('a',encoding='utf-8') as f:f.write(line);f.flush();os.fsync(f.fileno())
def save_settings(settings,path=None):
 obj=dict(DEFAULT_SETTINGS);obj.update(settings or {});_atomic(path or DATA_DIR/'settings.json',json.dumps(obj,default=str));return obj
def load_settings(path=None):
 try:obj=json.loads(Path(path or DATA_DIR/'settings.json').read_text());d=dict(DEFAULT_SETTINGS);d.update(obj);return d
 except FileNotFoundError:return dict(DEFAULT_SETTINGS)
 except Exception as e:log.warning('settings load failed: %s',e);return dict(DEFAULT_SETTINGS)
