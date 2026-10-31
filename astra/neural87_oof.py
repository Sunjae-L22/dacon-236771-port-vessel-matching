"""Author: Astra. Fold-specific hard-negative encoder training and NumPy rank fusion.
Run train/eval in separate processes to avoid the local PyTorch/LightGBM crash.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import sys,json,time,argparse,random
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'shared/experiments/pinned/build84_astra'))
import cv,opus_lib as B

def fit_pairs(fold):
    tr,cat=cv.load();folds=cv.folds(tr);fit=tr[folds!=fold];va=tr[folds==fold]
    llm_path=ROOT/('astra/full84/llm_fit.csv' if fold<0 else f'astra/ranker84_fold{fold}/llm_fit.csv')
    llm=B.llm_korean_rows(fit,llm_path)
    real=fit[fit.vessel_name.map(B.is_hangul)].copy();real['source']='real';llm=llm.copy();llm['source']='llm'
    pairs=pd.concat([real,llm],ignore_index=True).drop_duplicates(['target_vessel_id','vessel_name','source'])
    assert not set(pairs.target_vessel_id)&set(va.target_vessel_id)
    for source,g in pairs.groupby('source'):
        counts=g.groupby('target_vessel_id').size()
        pairs.loc[g.index,'sample_weight']=.5/len(counts)/g.target_vessel_id.map(counts)
    return pairs,fit,va,cat

def train(fold):
    import torch
    from rapidfuzz import process,fuzz
    import ngram_encoder84 as N
    random.seed(87);np.random.seed(87);torch.manual_seed(87)
    pairs,fit,va,cat=fit_pairs(fold);out=ROOT/('astra/neural87_full' if fold<0 else f'astra/neural87_fold{fold}');out.mkdir(exist_ok=True)
    import fcntl
    lock=(out/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    if (out/'train_manifest.json').exists():return
    names=cat.vessel_name.fillna('').astype(str).tolist();norm=[B.norm(s) for s in names]
    id2i={v:i for i,v in enumerate(cat.vessel_id)};gt=cat.gross_tonnage.map(B.positive).to_numpy()
    # The cached pools depend only on allowed catalog names/attributes.
    hard=json.loads((ROOT/'astra/hardnegative87/hard_negatives.json').read_text());start=time.time()
    for target in pairs.target_vessel_id.unique():
        if target in hard:continue
        t=id2i[target];lex=[i for _,_,i in process.extract(norm[t],norm,scorer=fuzz.ratio,limit=70) if norm[i]!=norm[t]][:24]
        if np.isfinite(gt[t]):
            gaps=np.abs(np.log(np.maximum(gt,1)/max(gt[t],1)));gaps[~np.isfinite(gaps)]=np.inf
            meta=[int(i) for i in np.argsort(gaps,kind='stable')[:100] if norm[i]!=norm[t]][:24]
        else:meta=[]
        hard[target]=dict(lexical=lex,metadata=meta)
    model=N.Encoder();opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.001)
    weights=pairs.sample_weight.to_numpy(float,copy=True);weights/=weights.sum();rng=np.random.default_rng(87)
    for step in range(1,901):
        batch=pairs.iloc[rng.choice(len(pairs),size=64,p=weights)]
        queries=[];positives=[];negatives=[]
        for row in batch.to_dict('records'):
            q=str(row['vessel_name']);t=id2i[row['target_vessel_id']]
            if rng.random()<.4:q=B._typo(q,int(rng.integers(1,3)),rng)
            if rng.random()<.15 and len(q.split())>1:q=' '.join(q.split()[::-1])
            if rng.random()<.1 and len(q)>5:q=q[:max(3,int(.8*len(q)))]
            options=hard[row['target_vessel_id']];chosen=list(rng.choice(options['lexical'],size=3));chosen+=list(rng.choice(options['metadata'] or options['lexical'],size=2))
            while len(chosen)<8:
                i=int(rng.integers(len(names)))
                if norm[i]!=norm[t]:chosen.append(i)
            queries.append(q);positives.append(names[t]);negatives.extend(names[i] for i in chosen)
        opt.zero_grad(set_to_none=True);qv=model(queries);pv=model(positives);nv=model(negatives).reshape(64,8,-1)
        en=np.array([B.norm(s) for s in positives]);eq=torch.tensor(en[:,None]==en[None,:])
        logits=qv@pv.T/.07;neg=torch.einsum('bd,bnd->bn',qv,nv)/.07
        loss=(torch.cat([logits,neg],1).logsumexp(1)-logits.masked_fill(~eq,float('-inf')).logsumexp(1)).mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%300==0:print('FOLD',fold,'STEP',step,'LOSS',float(loss.detach()),'SECONDS',round(time.time()-start),flush=True)
    torch.save(model.state_dict(),out/'best.pt')
    np.savez_compressed(out/'weights.npz',**{k:v.detach().numpy() for k,v in model.state_dict().items()})
    pairs.to_csv(out/'training_pairs.csv',index=False)
    (out/'train_manifest.json').write_text(json.dumps(dict(fold=fold,steps=900,weight=.1,selection='fixed from prior three-way tune; no fold-validation selection',fit_vessels=fit.target_vessel_id.nunique(),pair_rows=len(pairs),seconds=time.time()-start),indent=2))

def evaluate(fold):
    out=ROOT/f'astra/neural87_fold{fold}'
    import fcntl
    lock=(out/'eval.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    if (out/'eval_manifest.json').exists() and all((out/f'eval_features_{m}.npz').exists() for m in ['baseline','enhanced']):return
    import lightgbm as lgb
    import astra_lib as A
    import neural_numpy86 as NN
    import ranker84 as R
    os.environ['KO_DICT']='0'
    pairs,fit,va,cat=fit_pairs(fold);out=ROOT/f'astra/neural87_fold{fold}'
    ix=A.Index(cat,fit);llm=B.llm_korean_rows(fit,ROOT/f'astra/ranker84_fold{fold}/llm_fit.csv');ix.ko=B.KA.KoMatcher(cat,pd.concat([fit,llm],ignore_index=True))
    models={m:lgb.Booster(model_file=str(ROOT/f'astra/ranker84_fold{fold}/{m}.txt')) for m in ['baseline','enhanced']}
    neural=NN.Encoder(out/'weights.npz');catalog_vectors=neural(ix.names_raw)
    train_ids=set(fit.target_vessel_id);va_all=va[va.vessel_name.map(B.is_hangul)].copy();va=va_all.drop_duplicates('target_vessel_id');rows=[];start=time.time()
    query_rows=R.queries(va,fit)
    for row in va_all.to_dict('records'):
        row['call_sign']='UNSEEN';row['variant']='native_all_rows';query_rows.append(row)
    caches={m:dict(X=[],idx=[],allowed=[],groups=[],target=[],variant=[],sample_id=[]) for m in models}
    for n,r in enumerate(query_rows):
        query_vector=neural([str(r['vessel_name'])])[0]
        for method,M in [('baseline',B),('enhanced',A)]:
            idx,F=M.query_features(ix,r,M.query_candidates(ix,r));s=models[method].predict(F,num_threads=1)
            cache=caches[method];cache['X'].append(F);cache['idx'].append(np.array(idx,dtype=np.int32));cache['allowed'].append(np.array([ix.ids[i] not in train_ids for i in idx]));cache['groups'].append(len(idx));cache['target'].append(ix.id2i[r['target_vessel_id']]);cache['variant'].append(r['variant']);cache['sample_id'].append(r['sample_id'])
            order=sorted(range(len(idx)),key=lambda j:(-s[j],idx[j]));allowed=[j for j in order if ix.ids[idx[j]] not in train_ids]
            win=idx[(allowed or order)[0]];fused_win=win;short=[j for j in order[:20] if ix.ids[idx[j]] not in train_ids]
            if len(short)>1:
                ns=catalog_vectors[[idx[j] for j in short]]@query_vector;bs=s[short]
                fusion=(bs-bs.mean())/max(float(bs.std()),1e-6)+.1*(ns-ns.mean())/max(float(ns.std()),1e-6)
                best=min(range(len(short)),key=lambda k:(-float(fusion[k]),idx[short[k]]));fused_win=idx[short[best]]
            for name,w in [(method,win),(method+'_nn',fused_win)]:
                rows.append(dict(fold=fold,target=r['target_vessel_id'],sample_id=r['sample_id'],variant=r['variant'],method=name,pred=ix.ids[w],correct=ix.ids[w]==r['target_vessel_id']))
        if (n+1)%len(va)==0:
            d=pd.DataFrame(rows);d.to_csv(out/'predictions.csv',index=False)
            print('FOLD',fold,r['variant'],d[d.variant.eq(r['variant'])].groupby('method').correct.mean().to_dict(),flush=True)
    pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False)
    for method,cache in caches.items():
        arrays={k:np.concatenate(v) if k in ['X','idx','allowed'] else np.array(v) for k,v in cache.items()}
        np.savez_compressed(out/f'eval_features_{method}.npz',**arrays)
    (out/'eval_manifest.json').write_text(json.dumps(dict(fold=fold,vessels=len(va),queries=n+1,native_rows=len(va_all),seconds=time.time()-start),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);p.add_argument('--train',action='store_true');a=p.parse_args()
    train(a.fold) if a.train else evaluate(a.fold)
