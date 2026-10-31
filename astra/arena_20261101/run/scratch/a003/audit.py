import os,sys
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import runpy,json,hashlib,zipfile,collections
from pathlib import Path
O=Path('/Users/sunjae/Desktop/항만/astra/arena_20261101/run/scratch/a003')
# Run frozen probe to load its exact function, then independently exercise active branches.
ns=runpy.run_path(str(O/'probe.py'))
f=ns['apply']; cat=ns['cat']; old=cat.copy(); cat.clear()
cat.update({'A':{'vessel_name':'SHIP 1','gross_tonnage':'100'},'B':{'vessel_name':'SHIP 2','gross_tonnage':'100'},'C':{'vessel_name':'OTHER 2','gross_tonnage':'100'}})
q={'vessel_name':'선박 02','gross_tonnage':'100','international_gross_tonnage':None}
p={'baseline':'A','gate110_gap':1,'candidates':[{'id':'A'},{'id':'B'}]}
cases={
 'active_rescue_branch':(q,p,'B'),
 'ambiguous_numbers_and_tonnage':(q,dict(p,candidates=[{'id':'A'},{'id':'B'},{'id':'C'}]),'A'),
 'missing_tonnage':(dict(q,gross_tonnage=None),p,'A'),
 'English':(dict(q,vessel_name='SHIP 02'),p,'A'),
 'protected':(q,dict(p,gate110_gap=4),'A'),
 'numeric_agreement':(dict(q,vessel_name='선박 1'),p,'A'),
 'tonnage_outside':(dict(q,gross_tonnage=102),p,'A'),
 'digit_missing':(dict(q,vessel_name='선박'),p,'A')}
tests={k:{'pred':f(a,b)[0],'expected':expected,'pass':f(a,b)[0]==expected} for k,(a,b,expected) in cases.items()}
# This is deliberately a failure of the heuristic, not an empirical accuracy observation.
tests['wrong_query_digit_counterexample']={'true_id':'A','pred':f(q,p)[0],'harm':f(q,p)[0]!='A','meaning':'query digit typo can falsify deterministic safety claim'}
cat.clear();cat.update(old)
funnel={}
for fold in (2,4):
 ps=[json.loads(l) for l in (ns['ROOT']/f'astra/byt5_base112_20261031/fold{fold}/packets112.jsonl').read_text().splitlines()]
 qs={q['query_id']:q for q in json.loads((ns['ROOT']/f'astra/noise113/fold{fold}/eval/queries.json').read_text())}
 counts=collections.Counter(); errors=[]
 for p in ps:
  if not p['primary']:continue
  q=qs[p['query_id']]; counts['primary']+=1
  if not ns['hangul'](q['vessel_name']) or not p['gate110_gap']<4:continue
  counts['hard_hangul']+=1
  dn=ns['digits'](q['vessel_name'])
  if not dn:continue
  counts['query_has_digits']+=1
  if p['baseline']!=p['target']:errors.append({'query_id':p['query_id'],'target':p['target'],'name':q['vessel_name'],'baseline_name':cat[p['baseline']]['vessel_name'],'truth_name':cat[p['target']]['vessel_name']})
  bn=ns['digits'](cat[p['baseline']]['vessel_name'])
  if not bn or dn==bn:continue
  counts['explicit_digit_conflict']+=1
  if not any(ns['pos'](q.get(k)) for k in ('gross_tonnage','international_gross_tonnage')):continue
  counts['tonnage_present']+=1
  if f(q,p)[1]:counts['unique_candidate']+=1
 funnel[str(fold)]={'counts':dict(counts),'broad_scope_errors':errors}
verified={}
with zipfile.ZipFile('/Users/sunjae/Downloads/C_vote.zip') as z:
 for member in ('script.py','model/en/script.py','model/en/model/opus_lib.py'):
  actual=hashlib.sha256(z.read(member)).hexdigest()
  snapshot=ns['sha'](ns['ROOT']/'astra/arena_20261101/baseline'/member)
  verified[member]={'zip_sha256':actual,'snapshot_sha256':snapshot,'equal':actual==snapshot}
r={'synthetic_only':tests,'label_blind_funnel':funnel,'actual_zip_small_member_verification':verified,'audit_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(O/'audit.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
print(json.dumps(r,ensure_ascii=False,indent=2))
