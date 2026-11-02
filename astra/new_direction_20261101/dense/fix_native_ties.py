from pathlib import Path
import json,sys,time
sys.dont_write_bytecode=True
import numpy as np
from rapidfuzz import process,fuzz
from probe import ROOT,HERE,read,normalize,hangul
sys.path.insert(0,str(ROOT/'shared/src'));from kocat import jamo
cat=read(ROOT/'data/open/vessel_catalog.csv');table={r['vessel_name']:r for r in read(ROOT/'shared/kocat/out/catalog_ko.csv')};ids=[r['vessel_id'] for r in cat];pos={v:i for i,v in enumerate(ids)};owners=[];aliases=[]
for i,r in enumerate(cat):
 a=table.get(r['vessel_name'],{})
 for s in sorted({normalize(a.get(f'ko{j}','')) for j in range(4) if hangul(a.get(f'ko{j}',''))}):aliases.append(s);owners.append(i)
owners=np.array(owners);idx=np.arange(len(cat));ja=[jamo(s) for s in aliases];E=np.load(HERE/'alias_vectors.npy');Q=np.load(HERE/'native_query_vectors.npy');queries=json.loads((HERE/'native_queries.json').read_text());rr=[];t=time.time()
for i,r in enumerate(queries):
 sc=np.full(len(cat),-np.inf,dtype=np.float32);np.maximum.at(sc,owners,E@Q[i]);js=np.full(len(cat),-np.inf,dtype=np.float32);np.maximum.at(js,owners,process.cdist([jamo(r['name'])],ja,scorer=fuzz.ratio,workers=1)[0]);ci=pos[r['vessel_id']];dr=int(np.count_nonzero((sc>sc[ci])|((sc==sc[ci])&(idx<ci))))+1;jr=int(np.count_nonzero((js>js[ci])|((js==js[ci])&(idx<ci))))+1
 rr.append(dict(r,dense_rank=dr,jamo_rank=jr))
r={'tie_policy':'descending vessel max score, then original catalog index; replaces optimistic tied-rank count','rows':len(rr),'seconds':time.time()-t,'recall':{str(k):{'dense':sum(x['dense_rank']<=k for x in rr),'jamo':sum(x['jamo_rank']<=k for x in rr),'union':sum(x['dense_rank']<=k or x['jamo_rank']<=k for x in rr),'dense_unique':sum(x['dense_rank']<=k and x['jamo_rank']>k for x in rr)} for k in (1,10,50,100)}}
(HERE/'ALIAS_RESULTS_TIE_SAFE.json').write_text(json.dumps(r,indent=2));(HERE/'alias_tie_safe_rows.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rr));print(json.dumps(r,indent=2))
# Correct native English-prototype ranks too, without refitting or changing predictions.
E=np.load(HERE/'english_vectors.npy')
for filename,outname in [('native_query_vectors.npy','BASE_RESULTS_TIE_SAFE.json'),('trained_native_vectors.npy','TRAINED_RESULTS_TIE_SAFE.json')]:
 Q=np.load(HERE/filename);folds={}
 for fold in range(5):
  ranks=[]
  for i,x in enumerate(queries):
   if x['fold']!=fold:continue
   sc=E@Q[i];ci=pos[x['vessel_id']];ranks.append(int(np.count_nonzero((sc>sc[ci])|((sc==sc[ci])&(idx<ci))))+1)
  folds[str(fold)]={'rows':len(ranks),'recall':{str(k):sum(z<=k for z in ranks)/len(ranks) for k in (1,10,50,100)}}
 (HERE/outname).write_text(json.dumps({'tie_policy':'catalog index','folds':folds},indent=2))
