"""Astra: one prespecified vessel-frequency weighting ablation, cached features only."""
from pathlib import Path
import os,sys,json,time,hashlib,argparse,gc,fcntl
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['KO_DICT']='0'
import numpy as np,pandas as pd,lightgbm as lgb
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'shared/experiments/pinned/build84_astra'))
import cv,astra_lib as A

def main(fold):
    start=time.time();out=ROOT/f'astra/balance94/fold{fold}';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    cached_rows=ROOT/f'astra/coldship91/fold{fold}/queries.json'
    if cached_rows.exists():rows=json.loads(cached_rows.read_text())
    else:
        from coldship91 import training_rows
        tr,cat=cv.load();fit=tr[cv.folds(tr)!=fold];queries,_=training_rows(fit,fold);rows=queries.to_dict('records')
    code=ROOT/'shared/experiments/pinned/build84_astra';signature=hashlib.sha256(json.dumps(rows,sort_keys=True,default=str).encode())
    for name in ['opus_lib.py','astra_lib.py','ko_translit_argon.py']:signature.update((code/name).read_bytes())
    cache=ROOT/f'astra/ranker84_fold{fold}'/f'enhanced_features_{signature.hexdigest()[:16]}'
    assert (cache/'matrix.npy').exists(),f'Exact training query/cache signature mismatch: {cache}'
    counts=pd.Series([r['target_vessel_id'] for r in rows]).value_counts();w=np.array([counts[r['target_vessel_id']]**-.5 for r in rows],np.float64);w/=w.mean()
    keep=[i for i,n in enumerate(A.FEATURES) if n!='rank_name'];names=[A.FEATURES[i] for i in keep]
    groups=[];labels=[]
    for path in sorted(cache.glob('*.npz')):
        with np.load(path) as z:groups.extend(z['groups'].tolist());labels.append(z['y'])
    y=np.concatenate(labels);del labels;assert len(groups)==len(rows);row_weight=np.repeat(w,groups).astype(np.float32)
    model_path=out/'ranker.txt';plan=dict(fold=fold,cache_signature=signature.hexdigest(),queries=len(rows),training_vessels=len(counts),weight='1/sqrt(augmented query count for target vessel), normalized to mean group weight 1',group_weight_min=float(w.min()),group_weight_max=float(w.max()),query_count_min=int(counts.min()),query_count_max=int(counts.max()),num_threads=2,num_trees=600,seed=84,features=names)
    (out/'plan.json').write_text(json.dumps(plan,indent=2));print(json.dumps({k:v for k,v in plan.items() if k!='features'}),flush=True)
    if not model_path.exists():
        source=np.load(cache/'matrix.npy',mmap_mode='r');assert len(source)==len(y)==len(row_weight)
        X=np.lib.format.open_memmap(out/'matrix70.npy',mode='w+',dtype=np.float32,shape=(len(source),len(keep)))
        for lo in range(0,len(source),100000):X[lo:lo+100000]=source[lo:lo+100000][:,keep]
        X.flush();del source;gc.collect()
        params=dict(objective='lambdarank',metric='ndcg',eval_at=[1],learning_rate=.05,num_leaves=63,min_data_in_leaf=20,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,lambdarank_truncation_level=10,verbose=-1,num_threads=2,seed=84,deterministic=True,force_col_wise=True)
        print('TRAIN',fold,len(y),'candidate rows',round(time.time()-start),flush=True)
        model=lgb.train(params,lgb.Dataset(X,y,group=groups,weight=row_weight,feature_name=names),600);model.save_model(str(model_path));del X,y,row_weight;gc.collect()
    else:model=lgb.Booster(model_file=str(model_path))
    print('TRAINED',fold,round(time.time()-start),flush=True)
    with np.load(ROOT/f'astra/neural87_fold{fold}/eval_features_enhanced.npz') as archive:z={key:archive[key] for key in archive.files}
    s=model.predict(z['X'][:,keep],num_threads=1);offset=0;result=[]
    for j,n in enumerate(z['groups']):
        pool=np.arange(offset,offset+n);valid=pool[z['allowed'][pool]];valid=valid if len(valid) else pool;win=min(valid,key=lambda i:(-float(s[i]),int(z['idx'][i])))
        result.append(dict(fold=fold,target=int(z['target'][j]),variant=str(z['variant'][j]),sample_id=str(z['sample_id'][j]),method='balance94',pred=int(z['idx'][win]),correct=bool(z['idx'][win]==z['target'][j])));offset+=n
    d=pd.DataFrame(result);d.to_csv(out/'predictions.csv',index=False)
    old=pd.read_csv(ROOT/f'astra/rankless87_fold{fold}/predictions.csv').rename(columns={'correct':'baseline_correct','pred':'baseline_pred'})
    m=d.merge(old[['target','variant','sample_id','baseline_correct','baseline_pred']],on=['target','variant','sample_id'],validate='one_to_one');m.to_csv(out/'paired_predictions.csv',index=False)
    stress=m[m.variant!='native_all_rows'];native=m[m.variant=='native_all_rows'];orig=m[m.variant=='orig'];delta=float(stress.correct.mean()-stress.baseline_correct.mean());native_delta=float(native.correct.mean()-native.baseline_correct.mean());orig_delta=float(orig.correct.mean()-orig.baseline_correct.mean())
    vessel_delta=(stress.assign(delta=stress.correct.astype(float)-stress.baseline_correct.astype(float)).groupby('target').delta.mean().to_numpy());rng=np.random.default_rng(94);ci=np.quantile(rng.choice(vessel_delta,size=(5000,len(vessel_delta)),replace=True).mean(1),[.025,.975]).tolist()
    report=dict(fold=fold,seconds=time.time()-start,stress_baseline=float(stress.baseline_correct.mean()),stress_balance=float(stress.correct.mean()),stress_delta=delta,bootstrap95=ci,native_baseline=float(native.baseline_correct.mean()),native_balance=float(native.correct.mean()),native_delta=native_delta,original_delta=orig_delta,corrected=int(((~m.baseline_correct)&m.correct).sum()),regressed=int((m.baseline_correct&(~m.correct)).sum()),eligible_for_confirmation=bool(delta>=.005 and native_delta>=0 and orig_delta>=0),variants=m.groupby('variant')[['baseline_correct','correct']].mean().to_dict('index'))
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,default=0);a=p.parse_args();main(a.fold)
