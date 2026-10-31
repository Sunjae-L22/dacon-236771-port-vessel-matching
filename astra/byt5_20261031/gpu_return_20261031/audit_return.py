from pathlib import Path
import json,sys,hashlib
import numpy as np
root=Path('/Users/sunjae/Desktop/항만');base=root/'astra/byt5_20261031';out=base/'gpu_return_20261031'
sys.path.insert(0,str(base/'package'));from neural import decision
sys.path.insert(0,str(root/'astra/noise113'));from common import B
feat=B.FEATURES.index('ko_combined_score');reports={};detail=[]
for fold in [2,4]:
 packets=[json.loads(s) for s in (base/f'package/data/eval_fold{fold}.jsonl').read_text().splitlines()];rows=[json.loads(s) for s in (out/f'fold{fold}/eval/paired_rows.jsonl').read_text().splitlines()];assert len(packets)==len(rows)
 for p,r in zip(packets,rows):
  assert all(p[k]==r[k] for k in ['query_id','name','variant','vessel','target','baseline','gap']);assert [c['id'] for c in p['candidates']]==r['candidate_ids'];assert decision(p,r['scores'])==r['pred']
 ev=root/f'astra/noise113/fold{fold}/eval';enriched=[]
 for path in sorted(ev.glob('[0-9][0-9][0-9][0-9][0-9][0-9].npz')):
  lo=int(path.stem)
  with np.load(path) as z:
   idx=z['idx'];groups=z['groups'];kc=z['X'][:,feat];off=0
   for j,n in enumerate(groups):
    p=packets[lo+j];r=dict(rows[lo+j]);by_index={int(k):float(s) for k,s in zip(idx[off:off+n],kc[off:off+n])};off+=n
    by_id={c['id']:by_index[c['index']] for c in p['candidates']};old=r['baseline'];new=r['pred'];r['old_ko_score']=by_id[old];r['new_ko_score']=by_id[new]
    r['guard_pred']=new if by_id[new]>=by_id[old] else old
    enriched.append(r)
 def metrics(rs,predkey):
  rescue=[r for r in rs if r['baseline']!=r['target'] and r[predkey]==r['target']];harm=[r for r in rs if r['baseline']==r['target'] and r[predkey]!=r['target']]
  return dict(n=len(rs),rescue=len(rescue),harm=len(harm),net=len(rescue)-len(harm),rescue_vessels=len({r['vessel'] for r in rescue}),harm_vessels=len({r['vessel'] for r in harm}),changed=sum(r['baseline']!=r[predkey] for r in rs))
 reports[f'fold{fold}']={method:{group:metrics([r for r in enriched if group=='all' or r['variant'].startswith('clean')==(group=='clean')],predkey) for group in ['all','clean','noise']} for method,predkey in [('original','pred'),('fixed_name_guard_exploratory','guard_pred')]}
 detail.extend(dict(fold=fold,**{k:r[k] for k in ['query_id','name','variant','vessel','target','baseline','pred','guard_pred','old_ko_score','new_ko_score']}) for r in enriched if r['baseline']!=r['pred'])
 print(fold,reports[f'fold{fold}'],flush=True)
(out/'INDEPENDENT_REPLAY_AND_GUARD.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2));(out/'changed_detail.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2))
