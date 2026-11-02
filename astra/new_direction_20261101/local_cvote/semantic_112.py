"""Frozen lambda40 semantic change to production112 top100; no labels read."""
from pathlib import Path
import sys,os,json,csv,pickle,subprocess,hashlib
sys.dont_write_bytecode=True
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=HERE/'eval'
sys.path.insert(0,str(ROOT/'astra/research_20261101'));from channel_model import normalize,hangul,read
rows=read(OUT/'test.csv');texts=list(dict.fromkeys(normalize(r['vessel_name']) for r in rows));(OUT/'encode_names.json').write_text(json.dumps(texts,ensure_ascii=False));subprocess.run([sys.executable,str(ROOT/'astra/new_direction_20261101/dense_ranker/embed_worker.py'),str(OUT/'encode_names.json'),str(OUT/'query_vectors.npy')],check=True);V=np.load(OUT/'query_vectors.npy');vec={s:V[i] for i,s in enumerate(texts)}
cat=read(ROOT/'data/open/vessel_catalog.csv');table={r['vessel_name']:r for r in read(ROOT/'shared/kocat/out/catalog_ko.csv')};E=np.load(ROOT/'astra/new_direction_20261101/dense/alias_vectors.npy');owners={};j=0
for r in cat:
 a=table.get(r['vessel_name'],{});ss=sorted({normalize(a.get(f'ko{i}','')) for i in range(4) if hangul(a.get(f'ko{i}',''))});owners[r['vessel_id']]=list(range(j,j+len(ss)));j+=len(ss)
assert len(E)==j
ix={r['vessel_id']:i for i,r in enumerate(cat)};excluded=set(json.loads((ROOT/'astra/arena_20261101/baseline/model/ko/model/train_vessels.json').read_text()));trace={r['sample_id']:r for r in map(json.loads,(OUT/'en_trace.jsonl').read_text().splitlines())};pred={r['sample_id']:r for r in read(OUT/'en_pred.csv')};out=[]
for r in rows:
 sid=r['sample_id'];p=pred[sid];new=p['target_vessel_id'];t=trace.get(sid);record={'sample_id':sid,'original112':new,'semantic112':new,'gap110':float(p['ko_gap']) if p.get('ko_gap') else 99.,'changed':False}
 if t and t['kind']=='ranked' and t['gap110']<4:
  allow=[i for i,v in enumerate(t['candidate_ids']) if v not in excluded];allow=allow or list(range(len(t['candidate_ids'])));order=sorted(allow,key=lambda i:(-t['scores'][i],ix[t['candidate_ids'][i]]))[:100]
  assert t['candidate_ids'][order[0]]==new,'Production112 score reproduction mismatch'
  v=vec[normalize(r['vessel_name'])];sc={i:t['scores'][i]+40*(float((E[owners[t['candidate_ids'][i]]]@v).max()) if owners[t['candidate_ids'][i]] else 0.) for i in order};best=min(order,key=lambda i:(-sc[i],ix[t['candidate_ids'][i]]));new=t['candidate_ids'][best];record.update(semantic112=new,changed=new!=p['target_vessel_id'])
 out.append(record)
(OUT/'semantic112.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in out));print('ROWS',len(out),'CHANGED',sum(r['changed'] for r in out))
