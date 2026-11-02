"""Read-only check whether anonymized call_sign is derivable from official row fields.
No external identities, cross-test relations, or hidden answers are used.
"""
from pathlib import Path
import csv,hashlib,json,itertools,collections,time
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent

def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
cat=read(ROOT/'data/open/vessel_catalog.csv');byid={r['vessel_id']:r for r in cat};train=read(ROOT/'data/open/train.csv');seen=set();sample=[]
for r in train:
 if r['target_vessel_id'] not in seen:seen.add(r['target_vessel_id']);sample.append((r,byid[r['target_vessel_id']]))
formats=[]
for field in ('vessel_id','vessel_name'):
 for transform in ('raw','upper','lower','compact','numeric'):
  for prefix in ('','CS','CS_','cs','cs_','call_sign','call_sign:','callsign:','vessel:','vessel_id:','vessel_','ship:','ship_'):
   for suffix in ('',':cs','_cs','_call_sign'):
    formats.append((field,transform,prefix,suffix))
def val(r,f):
 field,tr,pre,suf=f;s=r[field]
 if tr=='upper':s=s.upper()
 if tr=='lower':s=s.lower()
 if tr=='compact':s=''.join(c for c in s.upper() if c.isalnum())
 if tr=='numeric':s=s.split('_')[-1].lstrip('0') if field=='vessel_id' else s
 return (pre+s+suf).encode()
algos=['sha1','sha256','sha512','md5','blake2b','blake2s'];hits=[];t=time.time();tested=0
for f in formats:
 for algo in algos:
  for where in ('prefix','suffix'):
   tested+=1;ok=True
   for r,c in sample[:3]:
    target=r['call_sign'].removeprefix('CS_').lower();d=getattr(hashlib,algo)(val(c,f)).hexdigest();d=d[:len(target)] if where=='prefix' else d[-len(target):]
    if d!=target:ok=False;break
   if ok:hits.append(dict(format=f,algorithm=algo,where=where))
report={'scope':'official train+catalog only; hypothesis test, no identification beyond official target labels','train_vessels':len(sample),'formats_tested':tested,'matching_rules':hits,'seconds':time.time()-t,'conclusion':'No supported identifier decoding rule' if not hits else 'Need heldout validation before use','not_tested':'Unknown salts, unprovided external identifiers, brute-force hidden data, test-row interactions are not attempted'}
(OUT/'RESULTS.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
