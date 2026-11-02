from pathlib import Path
import json,time,sys,os
sys.dont_write_bytecode=True
os.environ.update(TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np,torch
from transformers import AutoTokenizer,AutoModel
from probe import ROOT,HERE,encode,read,normalize,hangul

def main():
 torch.set_num_threads(2);start=time.time();cat=read(ROOT/'data/open/vessel_catalog.csv');table={r['vessel_name']:r for r in read(ROOT/'shared/kocat/out/catalog_ko.csv')};aliases=[];owners=[]
 for r in cat:
  a=table.get(r['vessel_name'],{})
  for s in sorted({normalize(a.get(f'ko{i}','')) for i in range(4) if hangul(a.get(f'ko{i}',''))}):aliases.append(s);owners.append(r['vessel_id'])
 tok=AutoTokenizer.from_pretrained(ROOT/'astra/semantic_prior/weights',local_files_only=True);model=AutoModel.from_pretrained(ROOT/'astra/semantic_prior/weights',local_files_only=True).to('mps').eval()
 f=HERE/'alias_vectors.npy'
 if not f.exists():np.save(f,encode(model,tok,['passage: '+s for s in aliases],'mps'))
 E=np.load(f);rows=json.loads((HERE/'native_queries.json').read_text());Q=np.load(HERE/'native_query_vectors.npy');rr=[];indices={}
 for j,v in enumerate(owners):indices.setdefault(v,[]).append(j)
 base=[json.loads(s) for s in (ROOT/'astra/new_direction_20261101/structural/rows.jsonl').read_text().splitlines()];reports=[]
 from rapidfuzz import fuzz,process
 sys.path.insert(0,str(ROOT/'shared/src'));from kocat import jamo
 ja=[jamo(a) for a in aliases]
 for i,r in enumerate(rows):
  sc=E@Q[i];best=float(sc[indices[r['vessel_id']]].max());above={owners[j] for j in np.flatnonzero(sc>best)};rank=len(above)+1;top=np.argsort(-sc)[:500];unique=list(dict.fromkeys(owners[j] for j in top))[:100]
  hits=process.extract(jamo(r['name']),ja,scorer=fuzz.ratio,limit=500);baseids=list(dict.fromkeys(owners[j] for a,s,j in hits));base_rank=baseids.index(r['vessel_id'])+1 if r['vessel_id'] in baseids else 99999
  rr.append(dict(r,dense_rank=rank,jamo_rank=base_rank,top100=unique))
 report={'method':'frozen E5 query-to-Korean-alias retrieval, no human-query training','rows':len(rr),'seconds':time.time()-start,'recall':{str(k):{'dense':sum(r['dense_rank']<=k for r in rr),'jamo':sum(r['jamo_rank']<=k for r in rr),'union':sum(r['dense_rank']<=k or r['jamo_rank']<=k for r in rr),'dense_unique':sum(r['dense_rank']<=k and r['jamo_rank']>k for r in rr)} for k in (1,10,50,100)}}
 (HERE/'ALIAS_RESULTS.json').write_text(json.dumps(report,indent=2));(HERE/'alias_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
