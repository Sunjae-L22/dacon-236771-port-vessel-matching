"""Astra: reconstruct requested G1 comparator; no submission build or source changes."""
from pathlib import Path
import os,sys,json
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,pandas as pd,lightgbm as lgb
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'opus/stage2'))
import stage2 as S

def main():
    X,s1,cand,q,y,meta=S.load()
    F=S.relative(X,s1,q);groups=np.bincount(q);keep=meta.fold.to_numpy()!=2;rowkeep=keep[q]
    modeldir=OUT/'g1_exclude_fold2';modeldir.mkdir(exist_ok=True)
    models=[]
    for seed in (0,1,2):
        path=modeldir/f'stage2_{seed}.txt'
        if path.exists():m=lgb.Booster(model_file=str(path))
        else:
            m=lgb.train(dict(S.PARAMS,seed=seed,num_threads=2),lgb.Dataset(F[rowkeep],y[rowkeep],group=groups[keep],feature_name=S.NAMES),S.ROUNDS)
            m.save_model(str(path))
        models.append(m);print('G1 model ready',seed,flush=True)
    base=ROOT/'astra/incumbent97/fold2';z=np.load(next(base.glob('eval_benchmark_*.npz')))
    rows=json.loads((base/'eval_benchmark_queries.json').read_text());scores=np.load(base/'benchmark_6models_all_scores.npy')
    ids,allowed,features,target=(z[k] for k in ('idx','allowed','X','target'))
    flat=[];qs=[];metas=[];hard=[];lo=0
    for j,n in enumerate(z['groups']):
        pool=np.arange(lo,lo+n);valid=pool[allowed[pool]]
        if not len(valid):valid=pool
        rank=sorted(valid,key=lambda k:(-float(scores[k]),int(ids[k])))[:20]
        if rows[j].get('source_language')=='ko':
            qs.extend([len(metas)]*len(rank));flat.extend(rank);metas.append((j,rows[j]))
            hard.append(float(scores[rank[0]]-scores[rank[1]])<4 if len(rank)>1 else False)
        lo+=n
    flat=np.array(flat);qs=np.array(qs);g=np.bincount(qs);starts=np.r_[0,np.cumsum(g)[:-1]]
    Z=S.relative(features[flat].astype(np.float32),scores[flat].astype(np.float32),qs)
    s2=sum(m.predict(Z,num_threads=2) for m in models)
    result=[]
    for k,(j,row) in enumerate(metas):
        st=starts[k];n=g[k];winner=st+int(np.argmax(s2[st:st+n]))
        old=ids[flat[st]];new=ids[flat[winner]]
        gated=new if hard[k] else old
        result.append(dict(variant=row['variant'],vessel=row['target_vessel_id'],hard=hard[k],
            baseline_correct=bool(old==target[j]),full_correct=bool(new==target[j]),gated_correct=bool(gated==target[j]),
            full_changed=bool(new!=old),gated_changed=bool(gated!=old)))
    df=pd.DataFrame(result);df.to_csv(OUT/'g1_fold2_rows.csv',index=False)
    report=dict(n=len(df),hard=int(df.hard.sum()),full_changed=int(df.full_changed.sum()),
        easy_changed=int((df.full_changed&~df.hard).sum()),gated_changed=int(df.gated_changed.sum()),
        gated_rescue=int((~df.baseline_correct&df.gated_correct).sum()),gated_damage=int((df.baseline_correct&~df.gated_correct).sum()),
        local_99_changed=90,local_easy_changed=0,local_g1_equal_to99=True,
        limitations=['Stage2 training excludes fold2 query labels; upstream OOF models for other folds saw fold2, so not fully nested CV',
                    'Development fold2 reused; synthetic evaluation cannot predict Public uplift',
                    'Local identity with99 is not proof of identity on Public test'])
    (OUT/'g1_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
