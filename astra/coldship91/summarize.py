"""Finalize coldship pilot gate and training candidate-absence diagnostics."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
def main(fold):
 out=ROOT/f'astra/coldship91/fold{fold}';m=pd.read_csv(out/'paired_predictions.csv');r=json.loads((out/'report.json').read_text());stress=m[m.variant!='native_all_rows'];native=m[m.variant=='native_all_rows'];orig=m[m.variant=='orig']
 orig_delta=float(orig.correct.mean()-orig.rankless_correct.mean());native_delta=float(native.correct.mean()-native.rankless_correct.mean());delta=float(stress.correct.mean()-stress.rankless_correct.mean())
 r.update(orig_delta=orig_delta,eligible_for_confirmation=bool(delta>=.005 and native_delta>=0 and orig_delta>=0),validation_caveat='Outer fold4 is method confirmation after choosing recipe on fold0; its vessels were used in previous experiments, not a pristine never-used holdout',training_contract='Each training query uses lexicon and flag/type/unit statistics from other two inner vessel partitions, not its own target. All outer heldout ships excluded from every dictionary/statistics fit.')
 dp=(stress.assign(delta=stress.correct.astype(float)-stress.rankless_correct.astype(float)).groupby('target').delta.mean()).to_numpy();rng=np.random.default_rng(91);boots=np.array([rng.choice(dp,len(dp),replace=True).mean() for _ in range(3000)]);r['stress_delta_vessel_bootstrap_95']=np.quantile(boots,[.025,.975]).tolist();r['stress_vessels']=len(dp)
 cold=[];old=[]
 for p in sorted((out/'features').glob('*.npz')):
  with np.load(p) as z:cold.extend(z['present'].tolist())
 paths=list((ROOT/f'astra/ranker84_fold{fold}').glob('enhanced_features_*'));assert len(paths)==1
 for p in sorted(paths[0].glob('*.npz')):
  with np.load(p) as z:
   y=z['y'];g=z['groups'];offset=0
   for n in g:old.append(bool(y[offset:offset+n].sum()));offset+=n
 assert len(old)==len(cold)
 cold=np.array(cold);old=np.array(old);r['missing_positive_training_groups']={'queries':len(cold),'original88':int((~old).sum()),'coldship':int((~cold).sum()),'newly_missing':int((old&~cold).sum()),'newly_retrieved':int((~old&cold).sum()),'note':'All-zero relevance groups have no useful ranking gradient. Positive injection is a possible next experiment, not used here.'}
 (out/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);a=p.parse_args();main(a.fold)
