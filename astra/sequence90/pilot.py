"""Astra: train-only weighted edit/digraph alignment, fold0 tune -> fold4 confirmation.
No external names, no manual mappings, no GPU; rankless87 candidate sets unchanged.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import sys,json,re,ctypes,time,argparse,collections
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'astra'),str(ROOT/'shared/experiments/pinned/build84_astra')]
import cv,opus_lib as B,astra_lib as A,ranker84 as R
from neural87_oof import fit_pairs
import ko_translit_argon as K
ALPH='abcdefghijklmnopqrstuvwxyz0123456789';V=len(ALPH);CHAR={c:i for i,c in enumerate(ALPH)}

def qnorm(s):
    body,_=K.extract_digits_and_clean_ko(str(s))
    return re.sub('[^a-z0-9]','',K.romanize_hangul(body,drop_eu=True).lower())

def enorm(s):
    body,_=K.extract_digits_and_clean_en(str(s))
    return re.sub('[^a-z0-9]','',K.en_to_ko_phonetic_lat(body).lower())

def alignment(a,b):
    ops=[];i=j=0
    for op,x,y in Levenshtein.editops(a,b):
        while i<x and j<y:ops.append((a[i],b[j]));i+=1;j+=1
        if op=='replace':ops.append((a[i],b[j]));i+=1;j+=1
        elif op=='insert':ops.append(('',b[j]));j+=1
        else:ops.append((a[i],''));i+=1
    while i<len(a) and j<len(b):ops.append((a[i],b[j]));i+=1;j+=1
    return ops

def train(fold):
    pairs,fit,va,cat=fit_pairs(fold);names=dict(zip(cat.vessel_id,cat.vessel_name));counts=collections.Counter();source=collections.Counter();target=collections.Counter();used=0
    weights=pairs.sample_weight.to_numpy(float);weights=weights/weights.mean()
    for row,w in zip(pairs.to_dict('records'),weights):
        a=qnorm(row['vessel_name']);b=enorm(names[row['target_vessel_id']])
        if not a or not b or Levenshtein.normalized_distance(a,b)>.65:continue
        used+=1;ops=alignment(a,b)
        for x,y in ops:counts[x,y]+=w;source[x]+=w;target[y]+=w
        for (a1,b1),(a2,b2) in zip(ops,ops[1:]):
            x=a1+a2;y=b1+b2
            if (len(x),len(y)) in [(2,1),(1,2)]:counts[x,y]+=w
    sub=np.ones((V,V));np.fill_diagonal(sub,0);dele=np.ones(V);ins=np.ones(V);c21=np.full((V,V,V),2.);c12=np.full((V,V,V),2.)
    # Symmetric evidence discount with a 10-pair shrinkage prior; identity stays exact.
    for (x,y),n in counts.items():
        if x==y:continue
        if any(c not in CHAR for c in x+y):continue
        denom=max(sum(source[c] for c in x)/max(len(x),1),sum(target[c] for c in y)/max(len(y),1))
        p=min(1.,n/(denom+10.));cost=max(.15,1.-.9*np.sqrt(p))
        if len(x)==1 and len(y)==1:sub[CHAR[x],CHAR[y]]=cost
        elif len(x)==1 and len(y)==0:dele[CHAR[x]]=cost
        elif len(x)==0 and len(y)==1:ins[CHAR[y]]=cost
        elif len(x)==2 and len(y)==1 and n>=3:c21[CHAR[x[0]],CHAR[x[1]],CHAR[y]]=cost
        elif len(x)==1 and len(y)==2 and n>=3:c12[CHAR[x],CHAR[y[0]],CHAR[y[1]]]=cost
    out=OUT/f'fold{fold}';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'costs.npz',sub=sub,dele=dele,ins=ins,c21=c21,c12=c12)
    top=sorted((dict(source=x,target=y,count=float(n)) for (x,y),n in counts.items() if x!=y),key=lambda r:-r['count'])[:100]
    (out/'train_manifest.json').write_text(json.dumps(dict(fold=fold,fit_vessels=fit.target_vessel_id.nunique(),holdout_vessels=va.target_vessel_id.nunique(),pair_rows=len(pairs),aligned_rows=used,train_vessel_overlap=0,method='single Viterbi Levenshtein alignment; shrunk character/digraph cost discounts',top_operations=top),ensure_ascii=False,indent=2))

class DP:
    def __init__(self,path):
        self.lib=ctypes.CDLL(str(OUT/'dp.dylib'));self.fn=self.lib.distance90
        ip=ctypes.POINTER(ctypes.c_int);dp=ctypes.POINTER(ctypes.c_double)
        self.fn.argtypes=[ip,ctypes.c_int,ip,ctypes.c_int,dp,dp,dp,dp,dp,ctypes.c_int];self.fn.restype=ctypes.c_double
        with np.load(path) as z:self.arr=[np.ascontiguousarray(z[k]) for k in ['sub','dele','ins','c21','c12']]
        self.ptr=[x.ctypes.data_as(dp) for x in self.arr];self.cache={}
    def enc(self,s):
        if s not in self.cache:self.cache[s]=np.array([CHAR[c] for c in s if c in CHAR],dtype=np.int32)
        return self.cache[s]
    def scores(self,a,b):
        x=self.enc(a);y=self.enc(b);ip=ctypes.POINTER(ctypes.c_int)
        args=[x.ctypes.data_as(ip),len(x),y.ctypes.data_as(ip),len(y),*self.ptr]
        full=1-self.fn(*args,0);local=1-self.fn(*args,1)
        return full,.5*(full+local),local

def score(fold):
    import lightgbm as lgb
    out=OUT/f'fold{fold}';start=time.time();dp=DP(out/'costs.npz')
    pairs,fit,va,cat=fit_pairs(fold);va=va[va.vessel_name.map(B.is_hangul)].copy();queries=R.queries(va.drop_duplicates('target_vessel_id'),fit)
    for row in va.to_dict('records'):row['call_sign']='UNSEEN';row['variant']='native_all_rows';queries.append(row)
    names=[enorm(s) for s in cat.vessel_name.fillna('')]
    model=lgb.Booster(model_file=str(ROOT/f'astra/rankless87_fold{fold}/ranker.txt'));cols=[A.FEATURES.index(n) for n in model.feature_name()]
    with np.load(ROOT/f'astra/neural87_fold{fold}/eval_features_enhanced.npz') as archive:z={key:archive[key] for key in archive.files}
    bs=model.predict(z['X'][:,cols],num_threads=1);offset=0;rows=[]
    assert len(queries)==len(z['groups'])
    for k,(row,n) in enumerate(zip(queries,z['groups'])):
        assert row['sample_id']==z['sample_id'][k] and row['variant']==z['variant'][k]
        pool=np.arange(offset,offset+n);pool=pool[z['allowed'][pool]];short=sorted(pool,key=lambda j:(-float(bs[j]),int(z['idx'][j])))[:20];offset+=n
        a=qnorm(row['vessel_name'])
        for rank,j in enumerate(short):
            ci=int(z['idx'][j]);full,balanced,local=dp.scores(a,names[ci])
            rows.append(dict(fold=fold,qid=k,sample_id=row['sample_id'],variant=row['variant'],target=int(z['target'][k]),candidate=ci,correct=ci==z['target'][k],rank=rank,base=float(bs[j]),full=full,balanced=balanced,local=local))
    pd.DataFrame(rows).to_csv(out/'scored.csv',index=False)
    print('SCORED',fold,len(queries),'seconds',round(time.time()-start),flush=True)

def predict(d,feature,weight):
    rows=[]
    for _,g in d.groupby('qid',sort=False):
        b=g.base.to_numpy();n=g[feature].to_numpy();s=(b-b.mean())/max(b.std(),1e-6)+weight*(n-n.mean())/max(n.std(),1e-6)
        win=g.iloc[np.lexsort((g.candidate.to_numpy(),-s))[0]]
        rows.append(dict(fold=int(win.fold),qid=int(win.qid),sample_id=win.sample_id,target=int(win.target),variant=win.variant,pred=int(win.candidate),correct=bool(win.correct)))
    return pd.DataFrame(rows)

def collect(select_only=False):
    d=pd.read_csv(OUT/'fold0/scored.csv');trials=[];b=predict(d,'full',0)
    for feature in ['full','balanced','local']:
        for weight in [0,.02,.05,.1,.2,.4,.8]:
            p=predict(d,feature,weight);trials.append(dict(feature=feature,weight=weight,stress=float(p.loc[p.variant!='native_all_rows','correct'].mean()),native=float(p.loc[p.variant=='native_all_rows','correct'].mean())))
    trials=pd.DataFrame(trials);trials.to_csv(OUT/'tune_trials.csv',index=False)
    eligible=trials[trials.native>=b.loc[b.variant=='native_all_rows','correct'].mean()]
    best=eligible.sort_values(['stress','weight'],ascending=[False,True]).iloc[0]
    selection=dict(feature=str(best.feature),weight=float(best.weight),selection='fold0 development only; fixed before fold4 labels are read for selection')
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2));print(selection,flush=True)
    if select_only:return
    rows=[];report={'selection':selection,'limits':['Previously reused outer folds; fold4 is confirmation for this new feature, not a pristine competition-wide holdout.','Synthetic stress distribution does not imply hidden-test accuracy.']}
    for fold in [0,4]:
        d=pd.read_csv(OUT/f'fold{fold}/scored.csv');b=predict(d,'full',0);p=predict(d,selection['feature'],selection['weight'])
        for label,frame in [('baseline',b),('sequence',p)]:frame['method']=label;rows.append(frame)
        gain=p.correct.astype(float)-b.correct.astype(float);z=p.copy();z['gain']=gain;ship=z[z.variant!='native_all_rows'].groupby('target').gain.mean().to_numpy();rng=np.random.default_rng(90);ci=np.quantile(rng.choice(ship,size=(5000,len(ship)),replace=True).mean(1),[.025,.975]).tolist()
        report[f'fold{fold}']=dict(baseline=float(b[b.variant!='native_all_rows'].correct.mean()),sequence=float(p[p.variant!='native_all_rows'].correct.mean()),delta=float(ship.mean()),bootstrap95=ci,corrected=int(((~b.correct)&p.correct).sum()),regressed=int((b.correct&(~p.correct)).sum()),native_baseline=float(b[b.variant=='native_all_rows'].correct.mean()),native_sequence=float(p[p.variant=='native_all_rows'].correct.mean()),variants=pd.concat([b,p]).groupby(['variant','method']).correct.mean().unstack().to_dict())
    report['adopt']=bool(selection['weight']>0 and report['fold4']['delta']>=.005 and report['fold4']['bootstrap95'][0]>0 and report['fold4']['native_sequence']>=report['fold4']['native_baseline'])
    pd.concat(rows).to_csv(OUT/'predictions.csv',index=False);(OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int);p.add_argument('--collect',action='store_true');p.add_argument('--select-only',action='store_true');a=p.parse_args()
    if a.collect or a.select_only:collect(a.select_only)
    else:train(a.fold);score(a.fold)
