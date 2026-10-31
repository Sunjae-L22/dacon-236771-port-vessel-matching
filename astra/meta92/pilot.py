"""Astra: distinct-vessel metadata likelihood-ratio pilot, tune0 -> confirm4.
Inputs only original fit rows and the allowed catalog. No test observations.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import sys,json,collections,time,argparse
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'astra'),str(ROOT/'shared/experiments/pinned/build84_astra'),str(ROOT/'astra/sequence90')]
import cv,opus_lib as B,ranker84 as R

def clean(x):return '' if pd.isna(x) else str(x).strip()

class Likelihood:
    def __init__(self,fit,cat,qcols,ccols):
        lookup=cat.set_index('vessel_id');self.joint=collections.Counter();self.q=collections.Counter();self.c=collections.Counter();self.total=0.
        for target,g in fit.groupby('target_vessel_id',sort=False):
            c=tuple(clean(lookup.loc[target,k]) for k in ccols)
            queries=set(tuple(clean(row[k]) for k in qcols) for row in g.to_dict('records'))
            queries={q for q in queries if all(q)}
            if not queries:continue
            for q in queries:
                w=1/len(queries);self.joint[q,c]+=w;self.q[q]+=w;self.c[c]+=w;self.total+=w
    def marginal(self,q):return (self.q[q]+.5)/(self.total+.5*(len(self.q)+1))
    def likelihood(self,q,c,alpha=10.,backoff=None):
        p=self.marginal(q) if backoff is None else backoff
        return (self.joint[q,c]+alpha*p)/(self.c[c]+alpha)
    def ratio(self,q,c):
        if not all(q):return 0.
        return np.log(self.likelihood(q,c)/self.marginal(q))

def score(fold):
    tr,cat=cv.load();folds=cv.folds(tr);fit=tr[folds!=fold];va=tr[folds==fold];assert not set(fit.target_vessel_id)&set(va.target_vessel_id)
    models=[Likelihood(fit,cat,['flag_name'],['flag_name']),Likelihood(fit,cat,['vessel_type'],['vessel_type']),Likelihood(fit,cat,['flag_name','vessel_type'],['flag_name','vessel_type'])]
    va=va[va.vessel_name.map(B.is_hangul)];queries=R.queries(va.drop_duplicates('target_vessel_id'),fit)
    for row in va.to_dict('records'):row['call_sign']='UNSEEN';row['variant']='native_all_rows';queries.append(row)
    d=pd.read_csv(ROOT/f'astra/sequence90/fold{fold}/scored.csv');cache={};ind=[];joint=[]
    cf=cat.flag_name.map(clean).tolist();ct=cat.vessel_type.map(clean).tolist()
    for row in d.to_dict('records'):
        r=queries[row['qid']];assert row['sample_id']==r['sample_id'] and row['variant']==r['variant']
        i=int(row['candidate']);qf=clean(r['flag_name']);qt=clean(r['vessel_type']);key=(qf,qt,cf[i],ct[i])
        if key not in cache:
            fm,tm,jm=models;fq=(qf,);tq=(qt,);fc=(cf[i],);tc=(ct[i],)
            fscore=fm.ratio(fq,fc);tscore=tm.ratio(tq,tc);independent=fscore+tscore
            if qf and qt:
                p=fm.likelihood(fq,fc)*tm.likelihood(tq,tc)
                joint_p=jm.likelihood((qf,qt),(cf[i],ct[i]),alpha=20.,backoff=p)
                hierarchical=np.log(joint_p/jm.marginal((qf,qt)))
            else:hierarchical=independent
            cache[key]=(float(independent),float(hierarchical))
        a,b=cache[key];ind.append(a);joint.append(b)
    d['independent']=ind;d['hierarchical']=joint
    out=OUT/f'fold{fold}';out.mkdir(exist_ok=True);d.to_csv(out/'scored.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(fold=fold,fit_rows=len(fit),fit_vessels=fit.target_vessel_id.nunique(),overlap=0,weighting='Each training vessel contributes total one vote; split equally across distinct observed query labels',independent_alpha=10,joint_alpha=20,query_groups=len(queries),candidate_pairs=len(d),no_test=True),indent=2))

def predict(d,feature,weight):
    rows=[]
    for _,g in d.groupby('qid',sort=False):
        b=g.base.to_numpy();n=g[feature].to_numpy();s=(b-b.mean())/max(b.std(),1e-6)+weight*(n-n.mean())/max(n.std(),1e-6)
        win=g.iloc[np.lexsort((g.candidate.to_numpy(),-s))[0]]
        rows.append(dict(fold=int(win.fold),qid=int(win.qid),sample_id=win.sample_id,target=int(win.target),variant=win.variant,pred=int(win.candidate),correct=bool(win.correct)))
    return pd.DataFrame(rows)

def select():
    d=pd.read_csv(OUT/'fold0/scored.csv');trials=[];b=predict(d,'independent',0)
    for feature in ['independent','hierarchical']:
        for weight in [0,.02,.05,.1,.2,.4]:
            p=predict(d,feature,weight);trials.append(dict(feature=feature,weight=weight,stress=float(p[p.variant!='native_all_rows'].correct.mean()),native=float(p[p.variant=='native_all_rows'].correct.mean())))
    trials=pd.DataFrame(trials);trials.to_csv(OUT/'tune_trials.csv',index=False)
    eligible=trials[trials.native>=b[b.variant=='native_all_rows'].correct.mean()]
    best=eligible.sort_values(['stress','weight'],ascending=[False,True]).iloc[0]
    selection=dict(feature=str(best.feature),weight=float(best.weight),selection='fold0 only, frozen before confirmation')
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2));print(selection,flush=True)

def collect():
    selection=json.loads((OUT/'selection.json').read_text());rows=[];report=dict(selection=selection,limits=['The outer folds have been reused in previous experiments.','Synthetic stress is not an estimate of Public LB.'])
    for fold in [0,4]:
        d=pd.read_csv(OUT/f'fold{fold}/scored.csv');b=predict(d,'independent',0);p=predict(d,selection['feature'],selection['weight'])
        gain=p.correct.astype(float)-b.correct.astype(float);delta=p.copy();delta['gain']=gain
        ship=delta[delta.variant!='native_all_rows'].groupby('target').gain.mean().to_numpy();rng=np.random.default_rng(92);ci=np.quantile(rng.choice(ship,size=(5000,len(ship)),replace=True).mean(1),[.025,.975]).tolist()
        for name,frame in [('baseline',b),('metadata',p)]:frame['method']=name;rows.append(frame)
        report[f'fold{fold}']=dict(baseline=float(b[b.variant!='native_all_rows'].correct.mean()),metadata=float(p[p.variant!='native_all_rows'].correct.mean()),delta=float(ship.mean()),bootstrap95=ci,corrected=int(((~b.correct)&p.correct).sum()),regressed=int((b.correct&(~p.correct)).sum()),native_baseline=float(b[b.variant=='native_all_rows'].correct.mean()),native_metadata=float(p[p.variant=='native_all_rows'].correct.mean()),variants=pd.concat([b,p]).groupby(['variant','method']).correct.mean().unstack().to_dict())
    report['adopt']=bool(selection['weight']>0 and report['fold4']['delta']>=.005 and report['fold4']['bootstrap95'][0]>0 and report['fold4']['native_metadata']>=report['fold4']['native_baseline'])
    pd.concat(rows).to_csv(OUT/'predictions.csv',index=False);(OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int);p.add_argument('--select',action='store_true');p.add_argument('--collect',action='store_true');a=p.parse_args()
    if a.select:select()
    elif a.collect:collect()
    else:score(a.fold)
