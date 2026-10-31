"""Author: Astra. Remove retrieval-rank dependence using cached training matrices."""
from pathlib import Path
import sys,json,time,argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'shared/experiments/pinned/build84_astra'))
import astra_lib as A

def main(fold):
    out=ROOT/('astra/rankless87_full' if fold<0 else f'astra/rankless87_fold{fold}');out.mkdir(exist_ok=True)
    import fcntl
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    keep=[i for i,n in enumerate(A.FEATURES) if n!='rank_name'];names=[A.FEATURES[i] for i in keep]
    model_path=out/'ranker.txt';start=time.time()
    if not model_path.exists():
        training=ROOT/('astra/full84' if fold<0 else f'astra/ranker84_fold{fold}')
        paths=list(training.glob('enhanced_features_*/matrix.npy'));assert len(paths)==1,paths
        cache=paths[0].parent;X=np.load(paths[0],mmap_mode='r');ys=[];groups=[]
        for p in sorted(cache.glob('*.npz')):
            with np.load(p) as z:ys.append(z['y']);groups.extend(z['groups'].tolist())
        y=np.concatenate(ys);assert sum(groups)==len(X)==len(y)
        params=dict(objective='lambdarank',metric='ndcg',eval_at=[1],learning_rate=.05,num_leaves=63,min_data_in_leaf=20,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,lambdarank_truncation_level=10,verbose=-1,num_threads=2,seed=84,deterministic=True,force_col_wise=True)
        model=lgb.train(params,lgb.Dataset(X[:,keep],y,group=groups,feature_name=names),600);model.save_model(str(model_path))
        print('TRAINED rankless',fold,round(time.time()-start),flush=True)
    else:model=lgb.Booster(model_file=str(model_path))
    if fold<0:
        (out/'manifest.json').write_text(json.dumps(dict(full=True,features=names,seconds=time.time()-start),indent=2));return
    evaluation=ROOT/f'astra/neural87_fold{fold}/eval_features_enhanced.npz'
    if not evaluation.exists():
        print('WAITING_FOR_EVAL_CACHE',fold,flush=True);return
    rows=[]
    with np.load(evaluation) as z:
        scores=model.predict(z['X'][:,keep],num_threads=1);idx=z['idx'];allowed=z['allowed'];offset=0
        for n,size in enumerate(z['groups']):
            pool=np.arange(offset,offset+size);valid=pool[allowed[pool]];valid=valid if len(valid) else pool
            win=min(valid,key=lambda j:(-float(scores[j]),int(idx[j])))
            rows.append(dict(fold=fold,target=int(z['target'][n]),variant=str(z['variant'][n]),sample_id=str(z['sample_id'][n]),method='rankless',pred=int(idx[win]),correct=bool(idx[win]==z['target'][n])))
            offset+=size
    d=pd.DataFrame(rows);d.to_csv(out/'predictions.csv',index=False)
    print(d.groupby('variant').correct.mean().to_string(),flush=True)
    (out/'manifest.json').write_text(json.dumps(dict(fold=fold,features=names,seconds=time.time()-start,change='remove rank_name; reuse enhanced candidate pool, other features and training rows'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);a=p.parse_args();main(a.fold)
