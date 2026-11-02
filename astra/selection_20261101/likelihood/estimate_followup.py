"""Label-free pair inventory and conservative time estimate; no model import."""
from pathlib import Path
import json,csv,pickle,unicodedata,datetime,hashlib
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;BASE=ROOT/'astra/selection_20261101/confirm';DATA=BASE/'eval'
def read(p):
 with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def norm(s):return ' '.join(''.join(c if c.isalnum() else ' ' for c in unicodedata.normalize('NFKC',str(s)).upper()).split())
def preds(p):return {r['sample_id']:r['target_vessel_id'] for r in read(p)}
en=preds(DATA/'en_pred.csv');hs=preds(DATA/'hs_pred.csv');v20=preds(BASE/'v20_hard_pred.csv');cat={r['vessel_id']:r['vessel_name'] for r in read(ROOT/'data/open/vessel_catalog.csv')}
features=pickle.loads((ROOT/'astra/selection_20261101/pairwise/confirm112_features.pkl').read_bytes());eligible=[];pairs=set()
for r in features:
 sid=r['sample_id'];cv=en[sid] if en[sid]==hs[sid] else v20[sid]
 if cv not in r['ids']:continue
 eligible.append({'sample_id':sid,'name':r['name'],'cvote':cv,'baseline112':r['baseline112'],'candidate_ids':r['ids']})
 for v in r['ids']:
  for q in (r['name'],'N/A'):pairs.add((norm(cat[v]),q))
assert len(features)==47 and len(eligible)==42,(len(features),len(eligible))
screen={tuple(p) for p in json.loads((HERE/'MODEL_PAIRS.json').read_text())};unseen=sorted(pairs-screen);progress=[]
for line in (HERE/'run.log').read_text().splitlines():
 try:r=json.loads(line)
 except json.JSONDecodeError:continue
 if isinstance(r,dict) and 'scored' in r:progress.append(r)
rates=[(b['seconds']-a['seconds'])/(b['scored']-a['scored']) for a,b in zip(progress,progress[1:]) if b['scored']>a['scored']];assert rates
rate=max(rates);now=datetime.datetime.now(datetime.timezone.utc);pending=max(0,len(screen)-progress[-1]['scored']);screen_remaining=pending*rate;followup_seconds=len(unseen)*rate*1.25+120;projected=now+datetime.timedelta(seconds=screen_remaining+followup_seconds);deadline=datetime.datetime(2026,10,31,21,35,tzinfo=datetime.timezone.utc)
result={'inference_performed':False,'truth_file_read':False,'hard_rows':len(features),'eligible_rows':len(eligible),'protected_outside_pool':len(features)-len(eligible),'total_unique_followup_pairs':len(pairs),'overlap_with_frozen_screen_cache':len(pairs&screen),'unseen_pairs':len(unseen),'steady_state_seconds_per_pair_max':rate,'safety_multiplier':1.25,'load_QA_allowance_seconds':120,'current_screen_pending_upper_estimate':pending,'screen_remaining_seconds':screen_remaining,'followup_seconds_including_safety':followup_seconds,'projected_finish_kst':projected.astimezone(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),'fits_0635_given_current_projection':projected<=deadline,'time_recheck_required_after_OOF':True}
(HERE/'FOLLOWUP_NEEDED_PAIRS.json').write_text(json.dumps(unseen,ensure_ascii=False)+'\n');(HERE/'FOLLOWUP_ELIGIBLE_ROWS.json').write_text(json.dumps(eligible,ensure_ascii=False,indent=2)+'\n');(HERE/'FOLLOWUP_TIME_ESTIMATE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
