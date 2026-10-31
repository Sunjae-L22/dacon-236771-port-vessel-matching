"""Astra fixed exploratory 50:50 coldship/88 score blend; no parameter search."""
from pathlib import Path
import os,sys,json,argparse
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,pandas as pd,lightgbm as lgb
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'shared/experiments/pinned/build84_astra'));import astra_lib as A

def main(fold):
 out=ROOT/f'astra/coldship91/fold{fold}';keep=[i for i,n in enumerate(A.FEATURES) if n!='rank_name']
 models=[lgb.Booster(model_file=str(ROOT/f'astra/rankless87_fold{fold}/ranker.txt')),lgb.Booster(model_file=str(out/'ranker.txt'))];names=[A.FEATURES[i] for i in keep];assert all(m.feature_name()==names for m in models)
 with np.load(ROOT/f'astra/neural87_fold{fold}/eval_features_enhanced.npz') as z:
  X=z['X'][:,keep];idx=z['idx'];allowed=z['allowed'];groups=z['groups'];targets=z['target'];variants=z['variant'];samples=z['sample_id']
 scores=[m.predict(X,num_threads=1) for m in models];scores.append((scores[0]+scores[1])*.5);methods=['rankless','coldship','blend50'];offset=0;rows=[]
 for n,size in enumerate(groups):
  pool=np.arange(offset,offset+size);valid=pool[allowed[pool]];valid=valid if len(valid) else pool
  for method,score in zip(methods,scores):
   win=min(valid,key=lambda j:(-float(score[j]),int(idx[j])))
   rows.append(dict(fold=fold,sample_id=str(samples[n]),target=int(targets[n]),variant=str(variants[n]),method=method,pred=int(idx[win]),correct=bool(idx[win]==targets[n])))
  offset+=size
 d=pd.DataFrame(rows);d.to_csv(out/'blend50_predictions.csv',index=False);v=d.pivot(index=['target','sample_id','variant'],columns='method',values='correct');stress=v[v.index.get_level_values('variant')!='native_all_rows'];native=v[v.index.get_level_values('variant')=='native_all_rows'];orig=v[v.index.get_level_values('variant')=='orig'];delta=(stress.blend50.astype(float)-stress.rankless.astype(float)).groupby('target').mean().to_numpy();rng=np.random.default_rng(91);boots=[rng.choice(delta,len(delta),replace=True).mean() for _ in range(3000)]
 report=dict(fold=fold,rule='Exactly .5*raw_score88 + .5*raw_scorecoldship; fixed before fold4 result; no tuning/search',exploratory=True,stress=stress.mean().to_dict(),native=native.mean().to_dict(),orig=orig.mean().to_dict(),stress_delta_vs88=float(np.mean(delta)),vessel_bootstrap95=np.quantile(boots,[.025,.975]).tolist(),variants=d.groupby(['variant','method']).correct.mean().unstack().to_dict('index'))
 (out/'blend50_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);a=p.parse_args();main(a.fold)
