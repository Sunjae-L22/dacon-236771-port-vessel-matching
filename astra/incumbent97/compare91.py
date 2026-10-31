"""Compare the heldout79-recipe pair with the heldout91-recipe counterpart, no weight tuning."""
from pathlib import Path
import json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'astra/incumbent97/fold2';report={}
for kind in ['old9','benchmark']:
 b=pd.read_csv(OUT/f'{kind}_2models_predictions.csv');w=pd.read_csv(OUT/f'{kind}_weak91_predictions.csv');m=b.merge(w,on='query_id',suffixes=('_base','_weak'),validate='one_to_one');assert (m.target_vessel_id_base==m.target_vessel_id_weak).all();m['delta']=m.correct_weak.astype(float)-m.correct_base.astype(float)
 if kind=='benchmark':
  meta=pd.read_csv(ROOT/'astra/strategy97/benchmark_fold2.csv')[['query_id','priority','cohort','family','variant','any_effective_change']];m=m.merge(meta,on='query_id',validate='one_to_one')
 else:
  for k in ['family','cohort','variant']:m[k]=m[k+'_base']
  m['priority']='old_synthetic'
 m.to_csv(OUT/f'{kind}_paired91.csv',index=False)
 cols=['cohort','family','variant'];rows=[]
 for keys,g in m.groupby(cols):
  d=dict(zip(cols,keys));d.update(rows=len(g),vessels=g.target_vessel_id_base.nunique(),baseline=float(g.correct_base.mean()),weak91=float(g.correct_weak.mean()),delta=float(g.delta.mean()),baseline_candidate_recall=float(g.recall_base.mean()),weak91_candidate_recall=float(g.recall_weak.mean()),baseline_top20_recall=float(g.top20_recall_base.mean()),weak91_top20_recall=float(g.top20_recall_weak.mean()),corrections=int((g.correct_weak&~g.correct_base).sum()),regressions=int((~g.correct_weak&g.correct_base).sum()));rows.append(d)
 per=pd.DataFrame(rows);per.to_csv(OUT/f'{kind}_condition_comparison.csv',index=False);families=[]
 for keys,g in m.groupby(['cohort','family']):
  delta=g.groupby('target_vessel_id_base').delta.mean().to_numpy();rng=np.random.default_rng(970219);boot=[rng.choice(delta,len(delta),replace=True).mean() for _ in range(3000)];families.append(dict(cohort=keys[0],family=keys[1],rows=len(g),vessels=len(delta),baseline=float(g.correct_base.mean()),weak91=float(g.correct_weak.mean()),vessel_macro_delta=float(delta.mean()),vessel_bootstrap95=np.quantile(boot,[.025,.975]).tolist()))
 report[kind]=dict(per_condition=rows,families=families,condition_wins=int((per.delta>0).sum()),condition_losses=int((per.delta<0).sum()),condition_ties=int((per.delta==0).sum()))
 if kind=='old9':
  x=m[m.family=='old9'];clean=x[~x.variant.isin(['edit2_noise','trunc_noise'])];report[kind]['diagnostic_means']=dict(equal9_baseline=float(x.correct_base.mean()),equal9_weak91=float(x.correct_weak.mean()),other7_baseline=float(clean.correct_base.mean()),other7_weak91=float(clean.correct_weak.mean()))
report['limits']=['Not existing all-data79/91 submissions scored on training rows; these are vessel-heldout retrained counterparts.','Reference has2 augmentation members, production79 has6;157 fit-supported universal entries versus175 production entries.','Per-condition stress scores have no established hidden-test mixture weights; do not select on a single pooled mean.','Compatibility-jamo queries may route to the English model in actual production; the unretrained English route is excluded from primary conclusions.','Baseline pair/91 comparison is diagnostic; Public79 remains the incumbent and91 was rejected on actual Public result.']
(OUT/'paired91_report.json').write_text(json.dumps(report,indent=2));print(json.dumps({'old9':report['old9']['diagnostic_means'],'benchmark_families':report['benchmark']['families'],'benchmark_conditions':{k:report['benchmark'][k] for k in ['condition_wins','condition_losses','condition_ties']}},indent=2))
