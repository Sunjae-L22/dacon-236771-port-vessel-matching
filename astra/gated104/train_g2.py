"""Astra: matched fold2 six-model Kanana ranker-row substitution, Qwen dictionary fixed."""
from pathlib import Path
import os, sys, json, time, hashlib, argparse, gc
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'astra'))
import incumbent97 as I
import numpy as np
import pandas as pd
import lightgbm as lgb

def prepare():
    start=time.time()
    tr,cat,fit,va,ix,llm,base=I.context(2)
    raw_path=ROOT/'shared/experiments/pinned/build100k/llm_korean_scored_kanana_clean.csv'
    raw=pd.read_csv(raw_path)
    raw=raw[raw.vessel_id.isin(set(fit.target_vessel_id))].copy()
    assert not set(raw.vessel_id)&set(va.target_vessel_id)
    # Match incumbent97.prepare: score the LLM spellings with native-fit-only dictionary.
    native=I.B.Index(cat,fit)
    raw['score']=[native.ko.score(n,native.id2i[v])['ko_combined_score'] for n,v in zip(raw.ko,raw.vessel_id)]
    raw.to_csv(OUT/'kanana_fit_rescored.csv',index=False)
    rows=I.B.llm_korean_rows(fit,OUT/'kanana_fit_rescored.csv').to_dict('records')
    (OUT/'kanana_training_rows.json').write_text(json.dumps(rows,ensure_ascii=False,default=str))
    cache=OUT/'kanana_features';cache.mkdir(exist_ok=True)
    counts=[]
    for lo in range(0,len(rows),500):
        path=cache/f'{lo:06d}.npz'
        if not path.exists():
            X=[];y=[];groups=[]
            for r in rows[lo:lo+500]:
                idx,F=I.B.query_features(ix,r,I.B.query_candidates(ix,r))
                target=ix.id2i[r['target_vessel_id']]
                X.append(F);y.append(np.asarray([i==target for i in idx],np.int8));groups.append(len(idx))
            np.savez_compressed(path,X=np.concatenate(X),y=np.concatenate(y),groups=np.asarray(groups,np.int32))
        with np.load(path) as z: counts.append(dict(path=str(path),queries=len(z['groups']),candidates=len(z['y'])))
        print('KANANA_FEATURES',lo+min(500,len(rows)-lo),'/',len(rows),'seconds',round(time.time()-start),flush=True)
    plan=json.loads((base/'plan.json').read_text())
    manifest=dict(fold=2,fit_vessels=fit.target_vessel_id.nunique(),heldout_vessels=va.target_vessel_id.nunique(),
        fit_kanana_rows=len(rows),qwen_rows=len(llm),seed_order=I.SEEDS,feature_names=I.B.FEATURES,
        raw_kanana_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),cache=counts,
        baseline_plan_sha256=hashlib.sha256((base/'plan.json').read_bytes()).hexdigest(),
        baseline_training_plan=plan['seeds'],dictionary='incumbent97 fold2 Qwen unchanged',
        heldout_overlap=0,params=I.PARAMS,rounds=600,weight=.5,gate=4.,
        limitations=plan['mismatches_to_production79']+['Kanana scores rescored with native-fit-only dictionary, unlike full-data101','Reused development fold2, not untouched final holdout'])
    (OUT/'train_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))

def train(seed):
    start=time.time();base=ROOT/'astra/incumbent97/fold2'
    plan=json.loads((OUT/'train_manifest.json').read_text())
    dest=OUT/f'seed{seed}';dest.mkdir(exist_ok=True)
    paths=[]
    for block in plan['baseline_training_plan'][str(seed)]['order']:
        if block=='llm':paths += sorted((OUT/'kanana_features').glob('*.npz'))
        else:paths += sorted((base/'features'/block).glob('*.npz'))
    assert paths
    modelpath=dest/'ranker.txt'
    if not modelpath.exists():
        groups=[];ys=[];n=0
        for p in paths:
            with np.load(p) as z:groups.extend(z['groups'].tolist());ys.append(z['y']);n+=len(z['y'])
        matrix=dest/'matrix.npy'
        X=np.lib.format.open_memmap(matrix,mode='w+',dtype=np.float32,shape=(n,60))
        off=0
        for p in paths:
            with np.load(p) as z:X[off:off+len(z['y'])]=z['X'];off+=len(z['y'])
        y=np.concatenate(ys);assert len(y)==sum(groups)==off
        print('TRAIN',seed,'queries',len(groups),'candidates',n,flush=True)
        m=lgb.train(I.PARAMS,lgb.Dataset(X,y,group=groups,feature_name=I.B.FEATURES),600)
        m.save_model(str(modelpath));del X,y,ys;gc.collect();matrix.unlink()
    else:m=lgb.Booster(model_file=str(modelpath))
    for kind in ('benchmark','old9'):
        p=next(base.glob(f'eval_{kind}_*.npz'))
        with np.load(p) as z:scores=m.predict(z['X'],num_threads=2)
        np.save(dest/f'{kind}_scores.npy',scores)
    print('FINISHED',seed,'seconds',round(time.time()-start),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.prepare:prepare()
    else:assert args.seed in I.SEEDS;train(args.seed)
