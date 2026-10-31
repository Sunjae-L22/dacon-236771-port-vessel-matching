"""Author: Astra. Paired retrieval pilot, fixed vessel-split fold 0. No test input."""
import os, sys, json, time
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'shared/experiments/pinned/build84_astra'))
os.environ['KO_DICT']='0'
import cv, opus_lib as B, astra_lib as A

def main(fold=0):
    tr, cat=cv.load(); f=cv.folds(tr)
    fit=tr[f!=fold].copy(); va=tr[f==fold].copy()
    assert not set(fit.target_vessel_id)&set(va.target_vessel_id)
    assert not set(fit.call_sign)&set(va.call_sign)
    ix=A.Index(cat,fit)
    # Recompute LLM quality using only this fit partition.
    llm=pd.read_csv(ROOT/'shared/experiments/pinned/build70/llm_korean_scored.csv')
    llm=llm[llm.vessel_id.isin(set(fit.target_vessel_id))].copy()
    llm['score']=[ix.ko.score(n,ix.id2i[v])['ko_combined_score'] for n,v in zip(llm.ko,llm.vessel_id)]
    out=ROOT/'astra'/('recall84' if fold==0 else f'recall84_fold{fold}'); out.mkdir(exist_ok=True)
    llm.to_csv(out/'llm_fit.csv',index=False)
    aug=B.llm_korean_rows(fit,out/'llm_fit.csv')
    ix.ko=B.KA.KoMatcher(cat,pd.concat([fit,aug],ignore_index=True))
    va=pd.concat([va[va.vessel_name.map(B.is_hangul)].drop_duplicates('target_vessel_id'),
                  va[~va.vessel_name.map(B.is_hangul)].drop_duplicates('target_vessel_id').sample(160,random_state=84)])
    result=[]; start=time.time()
    for variant in ['orig','edit2','drop_first','drop_last','trunc','reorder','trunc_noise','edit2_noise','rename']:
        for row in va.to_dict('records'):
            r=dict(row); rng=np.random.default_rng(int(r['sample_id'].split('_')[-1])+84000)
            name=r['vessel_name']; w=name.split()
            if variant.startswith('edit2'): name=B._typo(name,2,rng)
            elif variant=='drop_first' and len(w)>1: name=' '.join(w[1:])
            elif variant=='drop_last' and len(w)>1: name=' '.join(w[:-1])
            elif variant.startswith('trunc'): name=name[:max(3,int(.6*len(name)))]
            elif variant=='reorder': name=' '.join(w[1:]+w[:1])
            elif variant=='rename': name='UNKNOWN NAME'
            r['vessel_name']=name; r['call_sign']='UNSEEN'
            if variant.endswith('noise'):
                scale=float(rng.uniform(.75,1.3))
                for k in ['gross_tonnage','international_gross_tonnage']: r[k]=B.positive(r[k])*scale
            old=B.query_candidates(ix,r); new=set(old)|set(A.extra_candidates(ix,r)); target=ix.id2i[r['target_vessel_id']]
            result.append(dict(sample_id=r['sample_id'],variant=variant,ko=B.is_hangul(row['vessel_name']),old=target in old,new=target in new,n_old=len(old),n_new=len(new)))
        d=pd.DataFrame(result); d.to_csv(out/'rows.csv',index=False)
        print(variant,d[d.variant==variant].groupby('ko')[['old','new','n_old','n_new']].mean().round(4).to_dict(),round(time.time()-start),flush=True)
    d.groupby(['variant','ko'])[['old','new','n_old','n_new']].mean().to_csv(out/'summary.csv')
    (out/'manifest.json').write_text(json.dumps(dict(ko_dict=0,fold=fold,fit_rows=len(fit),validation_rows=len(va),seconds=time.time()-start,validation='candidate recall only, not ranker accuracy'),indent=2))

if __name__=='__main__':
    for fold in map(int,sys.argv[1:] or ['0']): main(fold)
