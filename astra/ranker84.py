"""Author: Astra. Paired Korean ranker pilot, then untouched fold-1 confirmation."""
import os,sys,time,json,argparse,gc,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'shared/experiments/pinned/build84_astra'))
os.environ['KO_DICT']='0'
import cv,opus_lib as B,astra_lib as A

def checkpoint_features(ix,rows,M,out,method,start):
    """Persist bounded chunks; assemble a disk-backed matrix instead of two RAM copies."""
    code=ROOT/'shared/experiments/pinned/build84_astra'
    signature=hashlib.sha256(json.dumps(rows,sort_keys=True,default=str).encode())
    for name in ['opus_lib.py','astra_lib.py','ko_translit_argon.py']:
        signature.update((code/name).read_bytes())
    cache=out/f'{method}_features_{signature.hexdigest()[:16]}'
    cache.mkdir(exist_ok=True)
    paths=[]
    for lo in range(0,len(rows),500):
        path=cache/f'{lo:06d}.npz';paths.append(path)
        if path.exists():continue
        xs=[];ys=[];groups=[]
        for row in rows[lo:lo+500]:
            idx,F=M.query_features(ix,row,M.query_candidates(ix,row));target=ix.id2i[row['target_vessel_id']]
            xs.append(F);ys.append(np.array([i==target for i in idx],dtype=np.int8));groups.append(len(idx))
        temp=path.with_suffix('.tmp.npz')
        np.savez_compressed(temp,X=np.concatenate(xs),y=np.concatenate(ys),groups=np.array(groups,dtype=np.int32))
        temp.replace(path)
        del xs,ys,groups;gc.collect()
        print(method,'CHECKPOINT',min(lo+500,len(rows)),len(rows),'seconds',round(time.time()-start),flush=True)
    count=0;all_groups=[]
    for path in paths:
        with np.load(path) as z:
            groups=z['groups'];count+=int(groups.sum());all_groups.extend(groups.tolist())
    X=np.lib.format.open_memmap(cache/'matrix.npy',mode='w+',dtype=np.float32,shape=(count,len(M.FEATURES)))
    y=np.empty(count,dtype=np.int8);offset=0
    for path in paths:
        with np.load(path) as z:
            n=len(z['y']);X[offset:offset+n]=z['X'];y[offset:offset+n]=z['y'];offset+=n
    X.flush()
    return X,y,all_groups

def queries(va,fit):
    result=[]
    for variant in ['orig','edit2','drop_first','drop_last','trunc','reorder','trunc_noise','edit2_noise','rename']:
        for row in va.to_dict('records'):
            r=dict(row); rng=np.random.default_rng(int(r['sample_id'].split('_')[-1])+84000)
            name=r['vessel_name']; w=name.split()
            if variant.startswith('edit2'): name=B._typo(name,2,rng)
            elif variant=='drop_first' and len(w)>1: name=' '.join(w[1:])
            elif variant=='drop_last' and len(w)>1: name=' '.join(w[:-1])
            elif variant.startswith('trunc'): name=name[:max(3,int(.6*len(name)))]
            elif variant=='reorder': name=' '.join(w[1:]+w[:1])
            elif variant=='rename':
                names=fit.loc[fit.vessel_name.map(B.is_hangul),'vessel_name'].tolist()
                name=str(names[int(rng.integers(len(names)))])
            r['vessel_name']=name; r['call_sign']='UNSEEN'
            if variant.endswith('noise'):
                scale=float(rng.uniform(.75,1.3))
                for k in ['gross_tonnage','international_gross_tonnage']: r[k]=B.positive(r[k])*scale
            r['variant']=variant; result.append(r)
    return result

def main(fold,full,features_only=False):
    tr,cat=cv.load(); f=cv.folds(tr)
    # Fold 0 was used to develop retrieval. Fold 1 is independent confirmation.
    fit=tr if full else tr[f!=fold].copy(); va=tr[f==fold].copy()
    out=ROOT/'astra'/('full84' if full else f'ranker84_fold{fold}');out.mkdir(exist_ok=True)
    ix=A.Index(cat,fit)
    if not full:
        assert not set(fit.target_vessel_id)&set(va.target_vessel_id)
        assert not set(fit.call_sign)&set(va.call_sign)
    gen=pd.read_csv(ROOT/'shared/experiments/pinned/build70/llm_korean_scored.csv')
    gen=gen[gen.vessel_id.isin(set(fit.target_vessel_id))].copy()
    gen['score']=[ix.ko.score(n,ix.id2i[v])['ko_combined_score'] for n,v in zip(gen.ko,gen.vessel_id)]
    gen.to_csv(out/'llm_fit.csv',index=False)
    llm=B.llm_korean_rows(fit,out/'llm_fit.csv')
    ix.ko=B.KA.KoMatcher(cat,pd.concat([fit,llm],ignore_index=True))
    ko=fit[fit.vessel_name.map(B.is_hangul)].copy()
    llm_fit=llm.sample(min(2500,len(llm)),random_state=84)
    names=pd.concat([ko,llm_fit],ignore_index=True)
    augmented=pd.concat([B.augment_korean(ko,84),llm_fit,B.augment_corrupt(names,84,frac=.7),B.augment_rename(ko,84,frac=.2)],ignore_index=True)
    # More training-only joint corruptions; identical examples for both compared models.
    noisy=B.augment_corrupt(names,184,frac=.3)
    rng=np.random.default_rng(84);scale=rng.uniform(.75,1.3,len(noisy))
    for k in ['gross_tonnage','international_gross_tonnage']: noisy[k]=noisy[k]*scale
    augmented=pd.concat([augmented,noisy],ignore_index=True)
    print('FIT',len(fit),'LLM',len(llm),'ranker queries',len(augmented),flush=True)
    rows=augmented.to_dict('records'); models={}; start=time.time()
    methods=['enhanced'] if full or features_only else ['baseline','enhanced']
    for method in methods:
        M=B if method=='baseline' else A
        model_path=out/(method+'.txt')
        if model_path.exists() and not features_only:
            models[method]=lgb.Booster(model_file=str(model_path)); continue
        X,y,groups=checkpoint_features(ix,rows,M,out,method,start)
        if features_only:
            del X,y,groups;gc.collect();continue
        params=dict(objective='lambdarank',metric='ndcg',eval_at=[1],learning_rate=.05,num_leaves=63,min_data_in_leaf=20,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,lambdarank_truncation_level=10,verbose=-1,num_threads=2,seed=84,deterministic=True,force_col_wise=True)
        model=lgb.train(params,lgb.Dataset(X,y,group=groups,feature_name=M.FEATURES),600)
        model.save_model(str(model_path));models[method]=model
        del X,y,groups;gc.collect()
        print('TRAINED',method,round(time.time()-start),flush=True)
    if full:
        import pickle
        with (out/'stats.pkl').open('wb') as fh: pickle.dump(dict(fit=fit,ko_pairs=pd.concat([fit,llm],ignore_index=True)),fh)
        return
    if features_only:return
    va=va[va.vessel_name.map(B.is_hangul)].drop_duplicates('target_vessel_id')
    train_ids=set(fit.target_vessel_id);results=[]
    for j,r in enumerate(queries(va,fit)):
        for method,M in [('baseline',B),('enhanced',A)]:
            idx,F=M.query_features(ix,r,M.query_candidates(ix,r));s=models[method].predict(F,num_threads=1)
            order=sorted(range(len(idx)),key=lambda j:(-s[j],idx[j]))
            win=next((idx[j] for j in order if ix.ids[idx[j]] not in train_ids),idx[order[0]])
            results.append(dict(sample_id=r['sample_id'],target=r['target_vessel_id'],variant=r['variant'],method=method,correct=ix.ids[win]==r['target_vessel_id'],recall=ix.id2i[r['target_vessel_id']] in idx,pred=ix.ids[win]))
        if (j+1)%len(va)==0:
            d=pd.DataFrame(results);d.to_csv(out/'predictions.csv',index=False)
            print('EVAL',r['variant'],d[d.variant==r['variant']].groupby('method')[['correct','recall']].mean().to_dict(),flush=True)
    d.groupby(['variant','method'])[['correct','recall']].mean().to_csv(out/'summary.csv')
    (out/'manifest.json').write_text(json.dumps(dict(fold=fold,ko_dict=0,fit_rows=len(fit),train_queries=len(augmented),validation_vessels=len(va),seconds=time.time()-start,full_oof=False),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,default=0);p.add_argument('--full',action='store_true');p.add_argument('--features-only',action='store_true');a=p.parse_args();main(a.fold,a.full,a.features_only)
