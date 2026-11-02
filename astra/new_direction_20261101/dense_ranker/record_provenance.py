from pathlib import Path
import csv,json,hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'astra/research_20261101'));from channel_model import read,normalize,hangul
cat=read(ROOT/'data/open/vessel_catalog.csv');src=ROOT/'shared/kocat/out/catalog_ko.csv';table={r['vessel_name']:r for r in read(src)};h=hashlib.sha256();n=0
for r in cat:
 a=table.get(r['vessel_name'],{})
 for s in sorted({normalize(a.get(f'ko{i}','')) for i in range(4) if hangul(a.get(f'ko{i}',''))}):h.update((r['vessel_id']+'\0'+s+'\n').encode());n+=1
vectors=ROOT/'astra/new_direction_20261101/dense/alias_vectors.npy';assert np.load(vectors,mmap_mode='r').shape==(n,384)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
report={'catalog_sha256':sha(ROOT/'data/open/vessel_catalog.csv'),'catalog_ko_sha256':sha(src),'alias_order_owner_sha256':h.hexdigest(),'alias_count':n,'vectors_sha256':sha(vectors),'generator_sha256':sha(ROOT/'astra/new_direction_20261101/dense/alias_probe.py'),'query_weights_config_sha256':sha(ROOT/'astra/semantic_prior/weights/config.json'),'note':'provenance recorded after training; exact original generator owner ordering reconstructed, no claim it was a pre-training freeze'}
(HERE/'PROVENANCE.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
