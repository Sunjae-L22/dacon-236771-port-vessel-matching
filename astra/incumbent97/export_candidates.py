"""Export genuine outerfold candidate scores for frozen neural evaluation only."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];SEEDS=[0,5000,1000,2000,6000,7000]
def main(fold,count):
 out=ROOT/f'astra/incumbent97/fold{fold}';catalog=pd.read_csv(ROOT/'data/open/vessel_catalog.csv');ids=catalog.vessel_id.astype(str).to_numpy();receipts=[]
 for kind in ['old9','benchmark','llm_holdout']:
  rows=json.loads((out/f'eval_{kind}_queries.json').read_text());paths=[p for p in out.glob(f'eval_{kind}_*.npz') if '.tmp.' not in p.name];assert len(paths)==1
  scores=sum(np.load(out/f'seed{s}/{kind}_scores.npy') for s in SEEDS[:count])/count;np.save(out/f'{kind}_{count}models_all_scores.npy',scores)
  with np.load(paths[0]) as z:idx=z['idx'];groups=z['groups'];target=z['target'];allowed=z['allowed']
  handles={k:(out/f'{kind}_{count}models_top{k}.jsonl').open('w') for k in [20,50,100]};offset=0
  for n,size in enumerate(groups):
   pool=np.arange(offset,offset+size);valid=pool[allowed[pool]];valid=valid if len(valid) else pool;ranked=sorted(valid,key=lambda j:(-float(scores[j]),int(idx[j])));r=rows[n]
   for k,fh in handles.items():
    use=ranked[:k];record=dict(q=str(r['vessel_name']),baseline_pred=int(idx[ranked[0]]),candidates=[int(idx[j]) for j in use],baseline_scores=[float(scores[j]) for j in use],t=int(target[n]),sample_id=r['sample_id'],query_id=r['query_id'],variant=r.get('variant',''),family=r.get('family',''),cohort=r.get('cohort','old9'),source_language=r.get('source_language',''),allowed=[bool(allowed[j]) for j in use],target_vessel_id=str(ids[target[n]]));fh.write(json.dumps(record,ensure_ascii=False)+'\n')
   offset+=size
  for fh in handles.values():fh.close()
  receipts.append(dict(kind=kind,queries=len(rows),all_candidates=len(idx),feature_cache=str(paths[0]),all_scores=str(out/f'{kind}_{count}models_all_scores.npy'),score_scale='arithmetic mean of raw600tree model scores, not sum or probabilities',models=SEEDS[:count],topk_files={str(k):str(out/f'{kind}_{count}models_top{k}.jsonl') for k in handles}))
 manifest=dict(fold=fold,count=count,catalog_rows=len(ids),catalog_sha256=hashlib.sha256((ROOT/'data/open/vessel_catalog.csv').read_bytes()).hexdigest(),heldout_only=True,purpose='Frozen evaluation; must not train or choose fusion weight on these labels',exports=receipts)
 (out/f'candidate_export_{count}models.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,default=2);p.add_argument('--models',type=int,default=2);a=p.parse_args();main(a.fold,a.models)
