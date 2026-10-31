from pathlib import Path
import os,sys,pickle,json,time
import numpy as np,pandas as pd
ROOT=Path('/Users/sunjae/Desktop/항만');os.environ['KO_DICT']='0';sys.path.insert(0,str(ROOT/'shared/experiments/pinned/build84_astra'))
import cv, opus_lib as B
tr,cat=cv.load(); folds=cv.folds(tr); native=tr[tr.vessel_name.map(B.is_hangul)].drop_duplicates('target_vessel_id'); start=time.time()
with (ROOT/'astra/full84/stats.pkl').open('rb') as fh:full=pickle.load(fh)
inside=B.KA.KoMatcher(cat,full['ko_pairs']); records=[]
for fold in range(5):
 fit=tr[folds!=fold]; gen=B.llm_korean_rows(fit,ROOT/f'astra/ranker84_fold{fold}/llm_fit.csv'); outside=B.KA.KoMatcher(cat,pd.concat([fit,gen],ignore_index=True)); ids=set(tr[folds==fold].target_vessel_id)
 for r in native[native.target_vessel_id.isin(ids)].to_dict('records'):
  i=inside.id_to_idx[r['target_vessel_id']];a=inside.score(r['vessel_name'],i);b=outside.score(r['vessel_name'],i)
  records.append(dict(target=r['target_vessel_id'],fold=fold,inside_translation=inside.translate_ko(r['vessel_name'])['trans'],outside_translation=outside.translate_ko(r['vessel_name'])['trans'],**{f'inside_{k}':v for k,v in a.items()},**{f'outside_{k}':v for k,v in b.items()}))
 d=pd.DataFrame(records);d.to_csv(ROOT/'astra/audit90/feature_shift.csv',index=False)
 print('fold',fold,'elapsed',time.time()-start,flush=True)
out={'ships':len(d),'translation_changes':int((d.inside_translation!=d.outside_translation).sum()),'features':{k:{'inside_mean':float(d['inside_'+k].mean()),'outside_mean':float(d['outside_'+k].mean()),'inside_perfect':int((d['inside_'+k]>=.99999).sum()),'outside_perfect':int((d['outside_'+k]>=.99999).sum())} for k in B.KA.KoMatcher.FEATURE_NAMES}}
(ROOT/'astra/audit90/feature_shift.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
