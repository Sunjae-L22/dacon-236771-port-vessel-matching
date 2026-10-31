"""Collect the two predeclared coldship pilots and fixed exploratory blend."""
from pathlib import Path
import json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];base=ROOT/'astra/coldship91'
d=pd.concat([pd.read_csv(base/f'fold{f}/blend50_predictions.csv') for f in [0,4]],ignore_index=True);p=d.pivot(index=['fold','target','sample_id','variant'],columns='method',values='correct');stress=p[p.index.get_level_values('variant')!='native_all_rows'];native=p[p.index.get_level_values('variant')=='native_all_rows'];orig=p[p.index.get_level_values('variant')=='orig'];o={'folds':{},'pooled':{},'full_training_or_submission_built':False,'limitations':['Two-fold pilot only; fold0 selected the method, fold4 confirms this recipe but was used in earlier experiments.','Pooled bootstrap intervals are descriptive after pilot selection, not an unbiased proof of Public improvement.','The fixed50:50 blend is exploratory, no alpha search.','No test rows or hidden labels used.']}
for f in [0,4]:o['folds'][str(f)]=json.loads((base/f'fold{f}/report.json').read_text());o['folds'][str(f)]['blend50']=json.loads((base/f'fold{f}/blend50_report.json').read_text())
for method in ['coldship','blend50']:
 delta=(stress[method].astype(float)-stress.rankless.astype(float)).groupby(['fold','target']).mean().to_numpy();rng=np.random.default_rng(91);b=[rng.choice(delta,len(delta),replace=True).mean() for _ in range(5000)]
 o['pooled'][method]={'vessels':len(delta),'stress_baseline':float(stress.rankless.mean()),'stress':float(stress[method].mean()),'delta':float(delta.mean()),'vessel_bootstrap95':np.quantile(b,[.025,.975]).tolist(),'native_baseline':float(native.rankless.mean()),'native':float(native[method].mean()),'orig_baseline':float(orig.rankless.mean()),'orig':float(orig[method].mean()),'corrections':int((stress[method]&~stress.rankless).sum()),'regressions':int((~stress[method]&stress.rankless).sum())}
o['progression_gate_plan']=json.loads((base/'progression_gate.json').read_text())
o['progression_gates']={}
for method in ['coldship','blend50']:
 v=o['pooled'][method];trunc=p[p.index.get_level_values('variant')=='trunc_noise'];trunc_delta=float(trunc[method].mean()-trunc.rankless.mean());v['trunc_noise_delta']=trunc_delta
 checks={'pooled_stress_gain_at_least_1pp':bool(v['delta']>=.01),'each_fold_nonnegative':all(bool(stress.xs(f,level='fold')[method].mean()>=stress.xs(f,level='fold').rankless.mean()) for f in [0,4]),'vessel_bootstrap_lower_positive':bool(v['vessel_bootstrap95'][0]>0),'native_no_loss_each_fold':all(bool(native.xs(f,level='fold')[method].mean()>=native.xs(f,level='fold').rankless.mean()) for f in [0,4]),'orig_no_loss_each_fold':all(bool(orig.xs(f,level='fold')[method].mean()>=orig.xs(f,level='fold').rankless.mean()) for f in [0,4]),'pooled_trunc_noise_no_loss':bool(trunc_delta>=0)}
 o['progression_gates'][method]={'checks':checks,'eligible_for_fivefold_validation':all(checks.values()),'failed':[k for k,vv in checks.items() if not vv]}
(base/'two_fold_report.json').write_text(json.dumps(o,indent=2));print(json.dumps({'pooled':o['pooled'],'gates':o['progression_gates']},indent=2))
