"""Verify row-local pair predictions without training or threshold changes."""
from pathlib import Path
import os,sys,json,pickle,collections
sys.dont_write_bytecode=True
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import numpy as np
import run as R
HERE=Path(__file__).resolve().parent
out={}
for fitfold,confirmfold in [(2,4),(4,2)]:
 model=R.lgb.Booster(model_file=str(HERE/f'verifier_fitfold{fitfold}.txt'))
 rows=pickle.loads((HERE/f'fold{confirmfold}_features.pkl').read_bytes())
 aa=[];bb=[];sa=[];sb=[]
 for r in rows:
  if not r['hard'] or len(r['ids'])<2:continue
  aa.append(r['F'][1:]);bb.append(np.repeat(r['F'][:1],len(r['ids'])-1,axis=0));sa.extend(r['scores'][1:]);sb.extend([r['scores'][0]]*(len(r['ids'])-1))
 X=R.pair_features(np.concatenate(aa),np.concatenate(bb),np.asarray(sa),np.asarray(sb))
 normal=model.predict(X,num_threads=2);reverse=model.predict(X[::-1],num_threads=2)[::-1]
 rng=np.random.default_rng(112);perm=rng.permutation(len(X));shuffled=model.predict(X[perm],num_threads=2)[np.argsort(perm)]
 take=np.linspace(0,len(X)-1,100,dtype=int);single=np.asarray([model.predict(X[j:j+1],num_threads=2)[0] for j in take])
 errs={'reversal_max_abs':float(np.max(np.abs(normal-reverse))),'shuffle_max_abs':float(np.max(np.abs(normal-shuffled))),'single_100_max_abs':float(np.max(np.abs(normal[take]-single)))}
 assert max(errs.values())==0
 detail={}
 for typ in ['clean','noise']:
  rr=[r for r in rows if r['hard'] and r['variant'].startswith('clean')==(typ=='clean')]
  wrong=[r for r in rr if r['baseline']!=r['target']]
  supported=[r for r in wrong if r['truth_in_top20']]
  detail[typ]={'hard_rows':len(rr),'hard_vessels':len({r['target'] for r in rr}),'baseline_errors':len(wrong),'selection_error_oracle_top20':len(supported),'selection_error_oracle_vessels':len({r['target'] for r in supported})}
 out[f'fit{fitfold}_confirm{confirmfold}']={'pair_rows_audited':len(X),'errors':errs,'protected_unchanged_asserted_in_main':True,'selection_error_oracle':detail}
(HERE/'QA.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
