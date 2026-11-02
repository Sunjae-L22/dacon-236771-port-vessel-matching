from pathlib import Path
import json,sys,collections
sys.dont_write_bytecode=True
from translit.normalized_channel import path_coefficients
from channel_model import normalize,read
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
cat={r['vessel_id']:r['vessel_name'] for r in read(ROOT/'data/open/vessel_catalog.csv')};table={r['vessel_name']:r for r in read(ROOT/'shared/kocat/out/catalog_ko.csv')};rr=[json.loads(s) for s in (HERE/'local/rows.jsonl').read_text().splitlines()];out=[]
for r in rr:
 new=r['candidate'];veto=False
 if new!=r['p112']:
  t=table[cat[new]]
  for a in {normalize(t.get(f'ko{i}','')) for i in range(4)}:
   c=path_coefficients(a,normalize(r['vessel_name']))
   if any(c.get(k,0)>0 for k in ('','delete','delete/delete')):veto=True
 if veto:new=r['p112']
 out.append(dict(r,veto_delete_only=veto,guarded=new))
s={'scope':'development cohort; deletion-only ambiguity veto formulated after inspecting 3 clean harms; NOT heldout or C_vote','rescue':sum(r['p112']!=r['target_vessel_id'] and r['guarded']==r['target_vessel_id'] for r in out),'harm':sum(r['p112']==r['target_vessel_id'] and r['guarded']!=r['target_vessel_id'] for r in out),'changed':sum(r['guarded']!=r['p112'] for r in out),'rescued_when_112_hsw_agree':sum(r['p112']==r['hsw'] and r['p112']!=r['target_vessel_id'] and r['guarded']==r['target_vessel_id'] for r in out)}
(HERE/'local/DELETION_VETO_DEV.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
