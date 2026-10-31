"""Astra: cold-ship ranker training via three inner vessel partitions; train/catalog only."""
from pathlib import Path
import os,sys,time,json,argparse,hashlib,gc,fcntl
os.environ['KO_DICT']='0';os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,pandas as pd,lightgbm as lgb
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'shared/experiments/pinned/build84_astra'))
import cv,opus_lib as B,astra_lib as A

def training_rows(fit,fold):
    llm=B.llm_korean_rows(fit,ROOT/f'astra/ranker84_fold{fold}/llm_fit.csv')
    ko=fit[fit.vessel_name.map(B.is_hangul)].copy(); llm_fit=llm.sample(min(2500,len(llm)),random_state=84)
    names=pd.concat([ko,llm_fit],ignore_index=True)
    augmented=pd.concat([B.augment_korean(ko,84),llm_fit,B.augment_corrupt(names,84,frac=.7),B.augment_rename(ko,84,frac=.2)],ignore_index=True)
    noisy=B.augment_corrupt(names,184,frac=.3);rng=np.random.default_rng(84);scale=rng.uniform(.75,1.3,len(noisy))
    for k in ['gross_tonnage','international_gross_tonnage']:noisy[k]=noisy[k]*scale
    return pd.concat([augmented,noisy],ignore_index=True),llm

def main(fold):
    start=time.time();out=ROOT/f'astra/coldship91/fold{fold}';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    tr,cat=cv.load();f=cv.folds(tr);fit=tr[f!=fold].copy();held=set(tr[f==fold].target_vessel_id)
    queries,llm=training_rows(fit,fold); rows=queries.to_dict('records');keep=[i for i,n in enumerate(A.FEATURES) if n!='rank_name'];names=[A.FEATURES[i] for i in keep]
    ids=np.array(sorted(fit.target_vessel_id.unique()));np.random.default_rng(91).shuffle(ids);inner={str(v):i%3 for i,v in enumerate(ids)}
    signature=hashlib.sha256(json.dumps(rows,sort_keys=True,default=str).encode());signature.update(json.dumps(inner,sort_keys=True).encode())
    for name in ['opus_lib.py','astra_lib.py','ko_translit_argon.py']:signature.update((ROOT/'shared/experiments/pinned/build84_astra'/name).read_bytes())
    signature=signature.hexdigest();plan=dict(fold=fold,fit_rows=len(fit),queries=len(rows),inner_seed=91,inner_parts=3,features=names,signature=signature,recipe='Identical ranker84 queries; each query uses stats and dictionary from other 2 inner vessel partitions; unfiltered candidate pool')
    mp=out/'plan.json'
    if mp.exists():assert json.loads(mp.read_text())==plan
    else:mp.write_text(json.dumps(plan,indent=2));(out/'queries.json').write_text(json.dumps(rows,ensure_ascii=False,default=str));(out/'inner_folds.json').write_text(json.dumps(inner,sort_keys=True))
    cache=out/'features';cache.mkdir(exist_ok=True); paths=[cache/f'{lo:06d}.npz' for lo in range(0,len(rows),500)]
    indexes=[]
    if not all(p.exists() for p in paths):
      for part in range(3):
        targets={k for k,v in inner.items() if v!=part};sub=fit[fit.target_vessel_id.isin(targets)];gen=llm[llm.target_vessel_id.isin(targets)]
        assert not targets&held
        ix=A.Index(cat,sub);ix.ko=B.KA.KoMatcher(cat,pd.concat([sub,gen],ignore_index=True));indexes.append(ix)
        print('INDEX',fold,part,'fit',len(sub),'pairs',len(sub)+len(gen),flush=True)
      for lo,path in zip(range(0,len(rows),500),paths):
        if path.exists():continue
        xs=[];ys=[];groups=[];catalog_ids=[];targets=[];parts=[];present=[]
        for j,row in enumerate(rows[lo:lo+500],start=lo):
          part=inner[str(row['target_vessel_id'])];ix=indexes[part]
          assert row['target_vessel_id'] not in {k for k,v in inner.items() if v!=part}
          idx,F=A.query_features(ix,row,A.query_candidates(ix,row));target=ix.id2i[row['target_vessel_id']]
          xs.append(F);ys.append(np.array([i==target for i in idx],dtype=np.int8));groups.append(len(idx));catalog_ids.extend(idx);targets.append(target);parts.append(part);present.append(target in idx)
        temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,X=np.concatenate(xs),y=np.concatenate(ys),groups=np.array(groups,dtype=np.int32),idx=np.array(catalog_ids,dtype=np.int32),query_index=np.arange(lo,lo+len(groups),dtype=np.int32),target=np.array(targets,dtype=np.int32),inner_fold=np.array(parts,dtype=np.int8),present=np.array(present,dtype=bool));temp.replace(path)
        print('CHECKPOINT',fold,lo+len(groups),len(rows),'elapsed',round(time.time()-start),'chunk_recall',np.mean(present),flush=True);del xs,ys;gc.collect()
      del indexes;gc.collect()
    model_path=out/'ranker.txt';all_groups=[];all_y=[];all_present=[];total=0
    for path in paths:
      with np.load(path) as z:all_groups.extend(z['groups'].tolist());all_y.append(z['y']);all_present.extend(z['present'].tolist());total+=len(z['y'])
    if not model_path.exists():
      X=np.lib.format.open_memmap(cache/'matrix.npy',mode='w+',dtype=np.float32,shape=(total,len(keep)));offset=0
      for path in paths:
        with np.load(path) as z:n=len(z['y']);X[offset:offset+n]=z['X'][:,keep];offset+=n
      X.flush();y=np.concatenate(all_y);assert len(X)==len(y)==sum(all_groups)
      params=dict(objective='lambdarank',metric='ndcg',eval_at=[1],learning_rate=.05,num_leaves=63,min_data_in_leaf=20,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,lambdarank_truncation_level=10,verbose=-1,num_threads=2,seed=84,deterministic=True,force_col_wise=True)
      print('TRAIN',fold,total,'candidate_rows',round(time.time()-start),flush=True)
      model=lgb.train(params,lgb.Dataset(X,y,group=all_groups,feature_name=names),600);model.save_model(str(model_path));del X,y;gc.collect()
    else:model=lgb.Booster(model_file=str(model_path))
    print('TRAINED',fold,round(time.time()-start),flush=True)
    cache_eval=ROOT/f'astra/neural87_fold{fold}/eval_features_enhanced.npz';result=[]
    with np.load(cache_eval) as z:
      scores=model.predict(z['X'][:,keep],num_threads=1);idx=z['idx'];allowed=z['allowed'];eval_target=z['target'];eval_variant=z['variant'];eval_sample=z['sample_id'];offset=0
      for j,n in enumerate(z['groups']):
        pool=np.arange(offset,offset+n);valid=pool[allowed[pool]];valid=valid if len(valid) else pool;win=min(valid,key=lambda i:(-float(scores[i]),int(idx[i])))
        result.append(dict(fold=fold,target=int(eval_target[j]),variant=str(eval_variant[j]),sample_id=str(eval_sample[j]),method='coldship',pred=int(idx[win]),correct=bool(idx[win]==eval_target[j])));offset+=n
    d=pd.DataFrame(result);d.to_csv(out/'predictions.csv',index=False);old=pd.read_csv(ROOT/f'astra/rankless87_fold{fold}/predictions.csv');old=old.rename(columns={'correct':'rankless_correct','pred':'rankless_pred'});m=d.merge(old[['target','variant','sample_id','rankless_correct','rankless_pred']],on=['target','variant','sample_id'],validate='one_to_one');m.to_csv(out/'paired_predictions.csv',index=False)
    stress=m[m.variant!='native_all_rows'];native=m[m.variant=='native_all_rows'];delta=float(stress.correct.mean()-stress.rankless_correct.mean());native_delta=float(native.correct.mean()-native.rankless_correct.mean());orig=m[m.variant=='orig'];orig_delta=float(orig.correct.mean()-orig.rankless_correct.mean())
    summary=dict(fold=fold,seconds=time.time()-start,queries=len(rows),training_candidate_recall=float(np.mean(all_present)),stress_baseline=float(stress.rankless_correct.mean()),stress_coldship=float(stress.correct.mean()),stress_delta=delta,native_baseline=float(native.rankless_correct.mean()),native_coldship=float(native.correct.mean()),native_delta=native_delta,orig_delta=orig_delta,eligible_for_confirmation=bool(delta>=.005 and native_delta>=0 and orig_delta>=0),validation_caveat='Fold4 is method confirmation; its vessels were used in previous unrelated experiments, not a pristine never-used holdout',variants=m.groupby('variant')[['rankless_correct','correct']].mean().to_dict('index'))
    (out/'report.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,default=0);a=p.parse_args();main(a.fold)
