"""Astra: fixed 50:50 score average of existing 87 and 88 OOF models.

No weight search. Reports all folds and vessel-bootstrap uncertainty. This
compares paired single-seed models, not the Public incumbent79 ensemble.
"""
from pathlib import Path
import os,sys,json
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'shared/experiments/pinned/build84_astra'))
import astra_lib as A

def main():
    out=ROOT/'astra/fusion91';out.mkdir(exist_ok=True)
    keep=[i for i,n in enumerate(A.FEATURES) if n!='rank_name'];rows=[]
    for f in range(5):
        m87=lgb.Booster(model_file=str(ROOT/f'astra/ranker84_fold{f}/enhanced.txt'))
        m88=lgb.Booster(model_file=str(ROOT/f'astra/rankless87_fold{f}/ranker.txt'))
        with np.load(ROOT/f'astra/neural87_fold{f}/eval_features_enhanced.npz') as archive:
            z={k:archive[k] for k in archive.files}
            a=m87.predict(z['X'],num_threads=1);b=m88.predict(z['X'][:,keep],num_threads=1)
            both=(a+b)*.5;offset=0
            for j,n in enumerate(z['groups']):
                pool=np.arange(offset,offset+n);v=pool[z['allowed'][pool]];v=v if len(v) else pool
                d=dict(fold=f,target=int(z['target'][j]),sample_id=str(z['sample_id'][j]),variant=str(z['variant'][j]))
                for name,s in [('ranked87',a),('rankless88',b),('equal_average',both)]:
                    win=min(v,key=lambda k:(-float(s[k]),int(z['idx'][k])))
                    d[name]=int(z['idx'][win]==z['target'][j])
                rows.append(d);offset+=n
    d=pd.DataFrame(rows);d.to_csv(out/'predictions.csv',index=False)
    s=d[d.variant!='native_all_rows'];cols=['ranked87','rankless88','equal_average']
    vessel=s.groupby(['fold','target'])[cols].mean();delta=(vessel.equal_average-vessel.rankless88).to_numpy()
    rng=np.random.default_rng(91);ci=np.quantile(delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1),[.025,.975]).tolist()
    result=dict(protocol='Fixed 50:50 raw score average, no tuning; exploratory all-fold comparison',stress=s[cols].mean().to_dict(),by_variant=s.groupby('variant')[cols].mean().to_dict('index'),by_fold=s.groupby('fold')[cols].mean().to_dict('index'),native=d[d.variant=='native_all_rows'][cols].mean().to_dict(),delta_vs88=float(delta.mean()),bootstrap_95=ci)
    result['eligible']=bool(result['delta_vs88']>=.005 and ci[0]>0 and result['native']['equal_average']>=result['native']['rankless88'])
    (out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
