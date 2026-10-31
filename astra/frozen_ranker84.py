"""Author: Astra. Isolate retrieval effects with an existing train-only frozen model."""
import os,sys,json,time
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[1]
os.environ['KO_DICT']='1'
sys.path.insert(0,str(ROOT/'shared/experiments/pinned/build84_astra'))
import opus_lib as B,astra_lib as A

def main():
    train=pd.read_csv(ROOT/'data/open/train.csv');cat=pd.read_csv(ROOT/'data/open/vessel_catalog.csv')
    split=pd.read_csv(ROOT/'terra/byt5_pilot/split_v1/rows.csv').set_index('sample_id')
    fit=train[train.sample_id.map(split.role).eq('train')]
    ix=A.Index(cat,fit)
    llm=B.llm_korean_rows(fit,ROOT/'shared/experiments/byt5_baseline/llm_scored_trainrole.csv')
    ix.ko=B.KA.KoMatcher(cat,pd.concat([fit,llm],ignore_index=True))
    model=lgb.Booster(model_file=str(ROOT/'shared/experiments/byt5_baseline/ranker.txt'))
    assert model.feature_name()==B.FEATURES
    queries=pd.read_csv(ROOT/'terra/byt5_pilot/colab_v1/delivery/port_byt5_pilot_v1/prepared/queries.csv')
    noisy=queries[queries.variant.isin(['trunc60','typo2'])].copy()
    for j,r in noisy.iterrows():
        rng=np.random.default_rng(int(r.orig_sample_id.split('_')[-1])+84000)
        scale=float(rng.uniform(.75,1.3))
        for k in ['gross_tonnage','international_gross_tonnage']: noisy.loc[j,k]=B.positive(r[k])*scale
    noisy['variant']=noisy.variant+'_noise';noisy['query_sample_id']=noisy.query_sample_id+'_noise'
    queries=pd.concat([queries,noisy],ignore_index=True)
    assert not set(queries.target_vessel_id)&set(fit.target_vessel_id)
    train_ids=set(fit.target_vessel_id);out=ROOT/'astra'/os.environ.get('FROZEN_OUT','frozen84_ranked');out.mkdir(exist_ok=True)
    results=[];start=time.time();cache={}
    for j,row in enumerate(queries.to_dict('records')):
        r={k:v for k,v in row.items() if k not in ['target_vessel_id','role','variant']}
        for name,retrieve in [('baseline',B.query_candidates),('expanded',A.query_candidates)]:
            key=(name,*(str(r.get(k)) for k in ['vessel_name','gross_tonnage','international_gross_tonnage','flag_name','vessel_type']))
            if key not in cache:
                idx,F=B.query_features(ix,r,retrieve(ix,r));scores=model.predict(F,num_threads=1)
                order=sorted(range(len(idx)),key=lambda k:(-scores[k],idx[k]))
                cache[key]=(idx,order)
            idx,order=cache[key]
            allowed=[k for k in order if ix.ids[idx[k]] not in train_ids]
            for exclusion,order2 in [(False,order),(True,allowed or order)]:
                win=idx[order2[0]];t=ix.id2i[row['target_vessel_id']]
                results.append(dict(query_sample_id=row['query_sample_id'],role=row['role'],variant=row['variant'],target=row['target_vessel_id'],method=name,exclude_train=exclusion,correct=ix.ids[win]==row['target_vessel_id'],recall=t in idx,top20=t in [idx[k] for k in order2[:20]],pred=ix.ids[win],n_candidates=len(idx)))
        if (j+1)%100==0:
            print(j+1,len(queries),'seconds',round(time.time()-start),flush=True)
            pd.DataFrame(results).to_csv(out/'predictions.csv',index=False)
    d=pd.DataFrame(results);d.to_csv(out/'predictions.csv',index=False)
    summary=d.groupby(['role','variant','method','exclude_train'])[['correct','recall','top20']].mean()
    summary.to_csv(out/'summary.csv');print(summary.to_string(),flush=True)
    (out/'manifest.json').write_text(json.dumps(dict(ko_dict=1,fold='existing three-way ship holdout',seconds=time.time()-start,queries=len(queries),validation='frozen-model retrieval ablation; not full OOF'),indent=2))

if __name__=='__main__':main()
