import sys,os
sys.dont_write_bytecode=True
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import json,csv,re,unicodedata,hashlib,math,time,fcntl
ROOT=Path('/Users/sunjae/Desktop/항만')
OUT=ROOT/'astra/arena_20261101/run/scratch/a003'
os.environ['TMPDIR']=str(OUT)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def digits(s): return tuple(int(x) for x in re.findall(r'[0-9]+',unicodedata.normalize('NFC',str(s))))
def pos(x):
 try:
  v=float(x); return v if math.isfinite(v) and v>0 else None
 except (ValueError,TypeError): return None
def hangul(s): return any('\uac00'<=c<='\ud7a3' for c in unicodedata.normalize('NFC',str(s)))
catpath=ROOT/'data/open/vessel_catalog.csv'
with catpath.open(encoding='utf-8-sig') as f: cat={r['vessel_id']:r for r in csv.DictReader(f)}
def apply(q,p):
 baseline=p['baseline']; nums=digits(q['vessel_name']); bn=digits(cat[baseline]['vessel_name'])
 if not hangul(q['vessel_name']) or not p['gate110_gap']<4 or not nums or not bn or nums==bn: return baseline,False
 tons=[v for v in (pos(q.get('gross_tonnage')),pos(q.get('international_gross_tonnage'))) if v]
 if not tons: return baseline,False
 hits=[]
 for c in p['candidates'][:10]:
  r=cat[c['id']]; gt=pos(r['gross_tonnage'])
  if digits(r['vessel_name'])==nums and gt and min(abs(t-gt)/gt for t in tons)<=.01: hits.append(c['id'])
 hits=sorted(set(hits))
 return (hits[0],True) if len(hits)==1 else (baseline,False)
def metrics(rs):
 changed=[r for r in rs if r['changed']]
 return dict(n=len(rs),vessels=len(set(r['target'] for r in rs)),base_correct=sum(r['baseline']==r['target'] for r in rs),changed=len(changed),rescue=sum(r['baseline']!=r['target'] and r['pred']==r['target'] for r in rs),harm=sum(r['baseline']==r['target'] and r['pred']!=r['target'] for r in rs),net=sum((r['pred']==r['target'])-(r['baseline']==r['target']) for r in rs),changed_vessels=len(set(r['target'] for r in changed)),gate_outside=sum(r['changed'] and r['gap']>=4 for r in rs))
sources=[catpath,Path(__file__),OUT/'spec.txt',ROOT/'astra/arena_20261101/baseline/script.py',ROOT/'astra/arena_20261101/baseline/model/en/script.py',ROOT/'astra/arena_20261101/baseline/model/en/model/opus_lib.py']
report={'mechanism':'frozen numeric identity veto v1','is_actual_cvote':False,'folds':{}}
lock=open(ROOT/'astra/arena_20261101/CPU.lock','r')
with lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 start=time.perf_counter()
 for fold in (2,4):
  pp=ROOT/f'astra/byt5_base112_20261031/fold{fold}/packets112.jsonl'
  qp=ROOT/f'astra/noise113/fold{fold}/eval/queries.json'
  sources.extend([pp,qp])
  qs={q['query_id']:q for q in json.loads(qp.read_text())}
  packets=[json.loads(l) for l in pp.read_text().splitlines()]
  records=[]; allpairs=[]; broad=[]
  for p in packets:
   q=qs[p['query_id']]
   pred,eligible=apply(q,p)
   allpairs.append((q,p))
   # Evaluation labels are accessed only after decision.
   r={'query_id':p['query_id'],'name':q['vessel_name'],'variant':q['variant'],'target':p['target'],'baseline':p['baseline'],'pred':pred,'eligible':eligible,'changed':pred!=p['baseline'],'gap':p['gate110_gap'],'primary':p['primary']}
   records.append(r)
   if p['primary'] and hangul(q['vessel_name']) and p['gate110_gap']<4 and digits(q['vessel_name']): broad.append(r)
  primary=[r for r in records if r['primary']]
  changed=[r for r in records if r['changed']]
  expected=[apply(q,p)[0] for q,p in allpairs]
  subset_idx=list(range(0,len(allpairs),7))
  contracts={'reverse_mismatch':sum(a!=b for a,b in zip(expected,[apply(q,p)[0] for q,p in allpairs[::-1]][::-1])), 'subset_mismatch':sum(expected[i]!=apply(*allpairs[i])[0] for i in subset_idx),'english_preservation_failures':0,'missing_tonnage_failures':0,'protected_gate_failures':0}
  for q,p in allpairs:
   e=dict(q,vessel_name='TEST 12'); missing=dict(q,gross_tonnage=None,international_gross_tonnage=None); high=dict(p,gate110_gap=4)
   contracts['english_preservation_failures']+=apply(e,p)[0]!=p['baseline']
   contracts['missing_tonnage_failures']+=apply(missing,p)[0]!=p['baseline']
   contracts['protected_gate_failures']+=apply(q,high)[0]!=p['baseline']
  counts={'all':metrics(primary),'clean':metrics([r for r in primary if r['variant'].startswith('clean')]),'noise':metrics([r for r in primary if not r['variant'].startswith('clean')]),'retained_gt':metrics([r for r in primary if r['variant'].endswith('_keep')]),'missing_gt':metrics([r for r in primary if r['variant'].endswith('_missing')])}
  report['folds'][str(fold)]={'packets':len(packets),'primary_counts':counts,'all_including_fewshot_overlap_audit':metrics(records),'contracts':contracts,'broad_numeric_scope':metrics(broad),'changes':changed}
  with (OUT/f'fold{fold}_rows.jsonl').open('w') as f:
   for r in records: f.write(json.dumps(r,ensure_ascii=False)+'\n')
 report['seconds']=time.perf_counter()-start
report['sha256']={str(p):sha(p) for p in sources}
report['public_math']={'baseline':.7*507/532+.3*3506/3568,'fixedE_K514':.7*514/532+.3*3506/3568,'fixedE_K521':.7*521/532+.3*3506/3568,'fixedE_K525':.7*525/532+.3*3506/3568,'perfectEnglish_K507':.7*507/532+.3}
(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='sha256'},ensure_ascii=False,indent=2))
