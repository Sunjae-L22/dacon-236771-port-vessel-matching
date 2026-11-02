"""Metadata-corrected synthetic diagnostic. No independent/Public validity claim."""
from pathlib import Path
import csv,json,collections,hashlib
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=HERE/'eval';OUT.mkdir(exist_ok=False)
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
cat={r['vessel_id']:r for r in read(ROOT/'data/open/vessel_catalog.csv')};train=read(ROOT/'data/open/train.csv');maps={k:collections.defaultdict(collections.Counter) for k in ('flag_name','vessel_type')}
for r in train:
 c=cat[r['target_vessel_id']]
 for k in maps:
  if r[k] and c[k]:maps[k][c[k]][r[k]]+=1
rows=read(ROOT/'astra/arena_highgain_20261101/package/data/full_test.csv');truth=read(ROOT/'astra/arena_highgain_20261101/package/data/full_truth.csv');audit=collections.Counter()
for r,t in zip(rows,truth):
 assert r['sample_id']==t['sample_id'];c=cat[t['target_vessel_id']]
 for k in maps:
  counts=maps[k].get(c[k],{});r[k]=sorted(counts,key=lambda x:(-counts[x],x))[0] if counts else ''
  audit[k+'_mapped' if counts else k+'_missing']+=1
for name,rs in [('test.csv',rows),('truth.csv',truth)]:
 with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
report={'rows':len(rows),'ships':len({r['target_vessel_id'] for r in truth}),'mapping':dict(audit),'selection':'unchanged prior fixed hash-selected64 unseen ships; metadata EN->KO modal mapping learned only from official train','independent_holdout':False,'limitations':['Names copied from catalog alias tables used by models; table-seen and style-biased','Metadata synthesized from target catalog plus official train mappings; oracle-clean metadata','This diagnostic can reject an intervention, cannot substantiate expected Public0.98'],'test_sha256':hashlib.sha256((OUT/'test.csv').read_bytes()).hexdigest()}
(OUT/'MANIFEST.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
