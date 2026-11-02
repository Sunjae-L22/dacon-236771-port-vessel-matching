import csv, json, re
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

def read(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))
cat=read(ROOT/'data/open/vessel_catalog.csv'); cat_by_id={x['vessel_id']:x for x in cat}; ci={x['vessel_id']:i for i,x in enumerate(cat)}
train=read(ROOT/'data/open/train.csv'); train_by_id={x['sample_id']:x for x in train}
rows=read(ROOT/'shared/experiments/cache/oof_v5.csv')
folds={}
for r in rows: folds.setdefault(r['target_vessel_id'],set()).add(r['fold'])
assert all(len(f)==1 for f in folds.values())
en=[r for r in rows if not re.search('[가-힣]',r['vessel_name'])]
errors=[]
for fold in range(5):
    rs=[r for r in rows if int(r['fold'])==fold]
    with np.load(ROOT/f'shared/experiments/cache/v5_fold{fold}.npz',allow_pickle=True) as z:
        cva=z['cva']; gva=z['gva']; yva=z['yva']
    assert len(rs)==len(cva)
    off=0
    for r,cands,g in zip(rs,cva,gva):
        true=ci[r['target_vessel_id']]
        labels=yva[off:off+g]; off+=g
        assert len(cands)==g
        assert list(labels)==[int(c==true) for c in cands]
        if r['pred']==r['target_vessel_id'] or re.search('[가-힣]',r['vessel_name']):continue
        q=train_by_id[r['sample_id']]; t=cat_by_id[r['target_vessel_id']]; p=cat_by_id[r['pred']]
        errors.append(dict(sample_id=r['sample_id'],query=r['vessel_name'],truth=t['vessel_name'],prediction=p['vessel_name'],fold=fold,truth_candidate_present=true in cands,truth_insertion_index=list(cands).index(true) if true in cands else None,candidate_count=int(g),query_gt=q['gross_tonnage'],true_gt=t['gross_tonnage'],pred_gt=p['gross_tonnage'],true_type=t['vessel_type'],pred_type=p['vessel_type'],query_type=q['vessel_type']))
result=dict(source_model='historical v5 grouped OOF; NOT current112/C_vote',rows=len(rows),English_rows=len(en),English_vessels=len(set(r['target_vessel_id'] for r in en)),English_errors=len(errors),candidate_present_errors=sum(e['truth_candidate_present'] for e in errors),candidate_absent_errors=sum(not e['truth_candidate_present'] for e in errors),vessel_crossfold_count=0,top20_status='unavailable: cva insertion order is not model rank',Public_row_truth='unavailable',English_one_correct_weight=.3/3568,English_62_correct_weight=62*.3/3568,errors=errors)
(OUT/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='errors'},ensure_ascii=False,indent=2))
