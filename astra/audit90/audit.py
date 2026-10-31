"""Astra audit: train-only diagnostics, no new training and no test data."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path('/Users/sunjae/Desktop/항만')
sys.path.insert(0,str(ROOT/'shared/experiments/pinned/build70'))
import cv
tr,cat=cv.load();f=cv.folds(tr);cat=cat.set_index('vessel_id',drop=False)
ko=tr[tr.vessel_name.map(cv.is_hangul)].copy();ko['fold']=f[tr.vessel_name.map(cv.is_hangul)]
pd.set_option('display.max_columns',20)
r=pd.read_csv(ROOT/'astra/ranker84_oof_predictions.csv')
piv=r.pivot(index=['fold','sample_id','target','variant'],columns='method',values='correct')
rl=pd.read_csv(ROOT/'astra/rankless87_oof_predictions.csv')
rl=rl[(rl.variant!='native_all_rows') & (rl.method=='rankless')].set_index(['fold','sample_id','target','variant']).correct.rename('rankless')
d=piv.join(rl);d['oracle']=d[['baseline','enhanced','rankless']].max(axis=1)
out={'ensemble':{'all':d.mean().to_dict(),'variant':d.groupby('variant').mean().to_dict('index'),'baseline_only':int((d.baseline&~d.enhanced&~d.rankless).sum()),'enhanced_only':int((d.enhanced&~d.baseline&~d.rankless).sum()),'rankless_only':int((d.rankless&~d.baseline&~d.enhanced).sum())}}
out['recall_by_variant']=r.groupby(['variant','method'])[['correct','recall']].mean().reset_index().to_dict('records')
w=pd.read_csv(ROOT/'shared/experiments/llm_rerank/weakness_actionability_errors.csv')
out['weakness']={'rows':len(w),'groups':w.groupby(['variant','error_category']).size().reset_index(name='n').to_dict('records'),'gt_evidence':w.groupby('gt_evidence').size().to_dict()}
# Oracle metadata equivalence: known exact catalog GT/type/flag cannot distinguish sister ships with these identical values.
meta=[]
support=[]
for fold in range(5):
 fit=tr[f!=fold];va=ko[ko.fold==fold].drop_duplicates('target_vessel_id');eligible=cat[~cat.vessel_id.isin(set(fit.target_vessel_id))]
 groups=eligible.groupby(['gross_tonnage','vessel_type','flag_name'],dropna=False).size()
 for row in va.to_dict('records'):
  c=cat.loc[row['target_vessel_id']]; key=(c.gross_tonnage,c.vessel_type,c.flag_name)
  meta.append(dict(target=c.vessel_id,fold=fold,exact_catalog_metadata_ties=int(groups.get(key,1))))
 for field in ['flag_name','vessel_type']:
  pairs=fit[['target_vessel_id',field]].drop_duplicates();pairs['catalog_label']=cat.loc[pairs.target_vessel_id,field].to_numpy()
  for row in va.to_dict('records'):
   sub=pairs[pairs[field].map(str)==str(row[field])];truth=cat.loc[row['target_vessel_id'],field]
   support.append(dict(fold=fold,target=row['target_vessel_id'],field=field,query_label=row[field],catalog_label=truth,total=len(sub),positive=int((sub.catalog_label==truth).sum())))
meta=pd.DataFrame(meta);sup=pd.DataFrame(support);meta.to_csv(ROOT/'astra/audit90/metadata_ties.csv',index=False);sup.to_csv(ROOT/'astra/audit90/metadata_support.csv',index=False)
out['metadata_ties']={'ships':len(meta),'unique_exact_gt_type_flag':int((meta.exact_catalog_metadata_ties==1).sum()),'ambiguous':int((meta.exact_catalog_metadata_ties>1).sum()),'mean_equal_tie_probability':float((1/meta.exact_catalog_metadata_ties).mean()),'max_ties':int(meta.exact_catalog_metadata_ties.max()),'note':'Oracle-catalog-label equivalence only, not a bound on hidden-test accuracy. No observed catalog DWT/size in query.'}
out['metadata_support']={field:{'unseen_query':int((g.total==0).sum()),'unseen_correct_mapping':int((g.positive==0).sum()),'support_less_than_5':int((g.positive<5).sum()),'missing_query':int(g.query_label.isna().sum()),'ships':len(g)} for field,g in sup.groupby('field')}
# Synthetic duplication across variants and trivial drop conditions.
unique=ko.drop_duplicates('target_vessel_id');nwords=unique.vessel_name.str.split().str.len()
out['validation']={'ko_rows':len(ko),'ko_ships':len(unique),'one_word_ships':int((nwords==1).sum()),'one_word_fraction':float((nwords==1).mean()),'target_rows_per_ship_median':float(ko.groupby('target_vessel_id').size().median()),'catalog_missing_gt':int(cat.gross_tonnage.isna().sum())}
(ROOT/'astra/audit90/report.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,ensure_ascii=False,indent=2))
