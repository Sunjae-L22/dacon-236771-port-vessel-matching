"""Astra: paired append-only translated-string feature experiment.

Training rows, candidate pools, original features and ranker parameters are
identical to rankless87. No test data is read. New projections are fitted on
the outer training partition only. Existing caches are read, never rewritten.
"""
import os
os.environ['KO_DICT']='0'
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,time,argparse,hashlib,gc,fcntl
from pathlib import Path
import numpy as np
import pandas as pd
from rapidfuzz import process,fuzz
from rapidfuzz.distance import JaroWinkler,Levenshtein
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'shared/experiments/pinned/build84_astra'))
import cv,opus_lib as B,astra_lib as A
import ranker84 as R

EXTRA=['trans_ratio','trans_pratio','trans_tsort','trans_tset','trans_jw',
       'trans_lev','trans_len','trans_len_ratio','trans_prefix','trans_len_diff']
KEEP=[i for i,n in enumerate(A.FEATURES) if n!='rank_name']
FEATURES=[A.FEATURES[i] for i in KEEP]+EXTRA

def extra(ix,row,idx):
    q=str(row.get('vessel_name') or '')
    info=ix.ko.translate_ko(q)
    qc=(info['trans'] or B.norm(B.romanize(B.ko_clean(q)))).upper()
    qt=' '.join(info['trans_tokens']).upper() or B.norm_tokens(B.romanize(q))
    names=[ix.names[i] for i in idx];toks=[ix.names_tok[i] for i in idx]
    n=len(idx);lens=np.array([len(s) for s in names])
    cols=[process.cdist([qc],names,scorer=fuzz.ratio)[0],
          process.cdist([qc],names,scorer=fuzz.partial_ratio)[0],
          process.cdist([qt],toks,scorer=fuzz.token_sort_ratio)[0],
          process.cdist([qt],toks,scorer=fuzz.token_set_ratio)[0],
          process.cdist([qc],names,scorer=JaroWinkler.normalized_similarity)[0],
          process.cdist([qc],names,scorer=Levenshtein.normalized_similarity)[0],
          np.full(n,len(qc)),np.minimum(lens,len(qc))/np.maximum(np.maximum(lens,len(qc)),1),
          np.array([len(qc)>=3 and s.startswith(qc) for s in names]),lens-len(qc)]
    return np.column_stack(cols).astype(np.float32)

def setup(fold):
    tr,cat=cv.load();fs=cv.folds(tr);fit=tr[fs!=fold].copy();va=tr[fs==fold].copy()
    if fold>=0:assert not set(fit.target_vessel_id)&set(va.target_vessel_id)
    training=ROOT/('astra/full84' if fold<0 else f'astra/ranker84_fold{fold}')
    ix=A.Index(cat,fit);llm=B.llm_korean_rows(fit,training/'llm_fit.csv')
    ix.ko=B.KA.KoMatcher(cat,pd.concat([fit,llm],ignore_index=True))
    ko=fit[fit.vessel_name.map(B.is_hangul)].copy()
    llmf=llm.sample(min(2500,len(llm)),random_state=84);names=pd.concat([ko,llmf],ignore_index=True)
    aug=pd.concat([B.augment_korean(ko,84),llmf,B.augment_corrupt(names,84,frac=.7),B.augment_rename(ko,84,frac=.2)],ignore_index=True)
    noisy=B.augment_corrupt(names,184,frac=.3);rng=np.random.default_rng(84);scale=rng.uniform(.75,1.3,len(noisy))
    for k in ['gross_tonnage','international_gross_tonnage']:noisy[k]=noisy[k]*scale
    aug=pd.concat([aug,noisy],ignore_index=True)
    cache=list(training.glob('enhanced_features_*/matrix.npy'));assert len(cache)==1
    return ix,fit,va,aug.to_dict('records'),cache[0].parent

def main(fold,cache_only=False):
    out=ROOT/('astra/translated90_full' if fold<0 else f'astra/translated90_fold{fold}');out.mkdir(exist_ok=True)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    if (out/'report.json').exists():return
    start=time.time();ix,fit,va,rows,cache=setup(fold)
    signature=hashlib.sha256(json.dumps(rows,sort_keys=True,default=str).encode()).hexdigest()
    # ranker84's signature also includes exact code files; recompute to catch drift.
    sig=hashlib.sha256(json.dumps(rows,sort_keys=True,default=str).encode())
    for name in ['opus_lib.py','astra_lib.py','ko_translit_argon.py']:
        sig.update((ROOT/'shared/experiments/pinned/build84_astra'/name).read_bytes())
    assert cache.name==f'enhanced_features_{sig.hexdigest()[:16]}',cache
    paths=sorted(cache.glob('*.npz'));all_y=[];all_groups=[];aux_paths=[]
    for path in paths:
        lo=int(path.stem);auxpath=out/path.name;aux_paths.append(auxpath)
        with np.load(path) as z:
            yy=z['y'];gg=z['groups'];all_y.append(yy);all_groups.extend(gg.tolist())
            if auxpath.exists():continue
            base_x=z['X']
            parts=[];offset=0
            for j,row in enumerate(rows[lo:lo+len(gg)]):
                cand=A.query_candidates(ix,row);idx=list(cand);n=len(idx)
                assert n==gg[j],(lo+j,n,gg[j])
                labels=np.array([i==ix.id2i[row['target_vessel_id']] for i in idx],dtype=np.int8)
                assert np.array_equal(labels,yy[offset:offset+n]),lo+j
                # Check stable candidate order using catalog-derived base columns.
                assert np.array_equal(np.array([len(ix.names[i]) for i in idx]),base_x[offset:offset+n,A.FEATURES.index('len_c')])
                parts.append(extra(ix,row,idx));offset+=n
            np.savez_compressed(auxpath,X=np.concatenate(parts))
        print('CACHE',fold,min(lo+500,len(rows)),len(rows),'seconds',round(time.time()-start),flush=True)
    if cache_only:return
    import lightgbm as lgb
    model_path=out/'ranker.txt';y=np.concatenate(all_y)
    if not model_path.exists():
        old=np.load(cache/'matrix.npy',mmap_mode='r')
        matrix_path=out/'matrix.npy'
        X=np.lib.format.open_memmap(matrix_path,mode='w+',dtype=np.float32,shape=(len(old),len(FEATURES)))
        offset=0
        for path in aux_paths:
            with np.load(path) as z:
                n=len(z['X']);X[offset:offset+n,:len(KEEP)]=old[offset:offset+n,KEEP];X[offset:offset+n,len(KEEP):]=z['X'];offset+=n
        assert offset==len(old)==len(y);X.flush();del old
        params=dict(objective='lambdarank',metric='ndcg',eval_at=[1],learning_rate=.05,num_leaves=63,min_data_in_leaf=20,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,lambdarank_truncation_level=10,verbose=-1,num_threads=2,seed=84,deterministic=True,force_col_wise=True)
        model=lgb.train(params,lgb.Dataset(X,y,group=all_groups,feature_name=FEATURES),600)
        model.save_model(str(model_path));del X;gc.collect();matrix_path.unlink()
        print('TRAINED',fold,'seconds',round(time.time()-start),flush=True)
    else:model=lgb.Booster(model_file=str(model_path))
    if fold<0:
        (out/'report.json').write_text(json.dumps(dict(full=True,features=FEATURES,seconds=time.time()-start),indent=2));return
    vaall=va[va.vessel_name.map(B.is_hangul)].copy()
    evalrows=R.queries(vaall.drop_duplicates('target_vessel_id'),fit)
    for r in vaall.to_dict('records'):r['call_sign']='UNSEEN';r['variant']='native_all_rows';evalrows.append(r)
    report=[]
    with np.load(ROOT/f'astra/neural87_fold{fold}/eval_features_enhanced.npz') as archive:
        z={k:archive[k] for k in archive.files}
        offset=0;xs=[]
        for j,row in enumerate(evalrows):
            assert row['sample_id']==z['sample_id'][j] and row['variant']==z['variant'][j]
            n=int(z['groups'][j]);idx=z['idx'][offset:offset+n]
            xs.append(np.column_stack([z['X'][offset:offset+n,KEEP],extra(ix,row,idx)]));offset+=n
        Xev=np.concatenate(xs);scores=model.predict(Xev,num_threads=1)
        np.savez_compressed(out/'eval_scores.npz',scores=scores)
        offset=0
        for j,row in enumerate(evalrows):
            n=int(z['groups'][j]);pool=np.arange(offset,offset+n);valid=pool[z['allowed'][pool]]
            valid=valid if len(valid) else pool
            win=min(valid,key=lambda k:(-float(scores[k]),int(z['idx'][k])))
            report.append(dict(fold=fold,sample_id=row['sample_id'],variant=row['variant'],target=row['target_vessel_id'],pred=ix.ids[z['idx'][win]],correct=bool(z['idx'][win]==z['target'][j])))
            offset+=n
    d=pd.DataFrame(report);d.to_csv(out/'predictions.csv',index=False)
    result=dict(fold=fold,features=FEATURES,seconds=time.time()-start,queries=len(rows),signature=signature,variants=d.groupby('variant').correct.mean().to_dict(),stress_mean=float(d[d.variant!='native_all_rows'].correct.mean()))
    (out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);p.add_argument('--cache-only',action='store_true');a=p.parse_args();main(a.fold,a.cache_only)
