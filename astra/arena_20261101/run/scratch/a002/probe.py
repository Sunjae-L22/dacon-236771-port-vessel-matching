import sys
sys.dont_write_bytecode=True
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import csv,json,math,re,unicodedata,hashlib,time,zipfile,fcntl
from collections import defaultdict,Counter
ROOT=Path('/Users/sunjae/Desktop/항만'); OUT=Path(__file__).resolve().parent
os.environ['TMPDIR']=str(OUT/'tmp')

def canon(s):return ' '.join(unicodedata.normalize('NFC',str(s)).upper().split())
def pos(v):
    try:
        x=float(v)
        return x if math.isfinite(x) and x>0 else None
    except (ValueError,TypeError):return None

def choose(q,base,gap,cat,groups):
    if not any('\uac00'<=c<='\ud7a3' for c in unicodedata.normalize('NFC',q['vessel_name'])):return base,'english'
    if not math.isfinite(float(gap)) or gap>=4:return base,'protected'
    family=groups[canon(cat[base]['vessel_name'])]
    if len(family)<2:return base,'no_homonym'
    gt=[pos(q.get(k)) for k in ('gross_tonnage','international_gross_tonnage')]
    gt=[v for v in gt if v is not None]
    if not gt:return base,'missing_gt'
    if len(set(gt))!=1:return base,'conflicting_gt'
    match=[v for v in family if pos(cat[v]['gross_tonnage'])==gt[0]]
    if len(match)!=1:return base,'no_unique_exact'
    return match[0], 'change' if match[0]!=base else 'already_exact'

def audit_scope(q,base,pred,gap,cat,groups):
    # Independent predicate; never consumes choose() reason or return eligibility.
    # True means either unchanged or every frozen condition holds for the changed ID.
    if pred==base:return True
    has_hangul=any('\uac00'<=c<='\ud7a3' for c in unicodedata.normalize('NFC',q['vessel_name']))
    if not has_hangul or not math.isfinite(float(gap)) or not float(gap)<4:return False
    key=canon(cat[base]['vessel_name']); family=groups[key]
    if len(family)<2 or pred not in family:return False
    values={pos(q.get(k)) for k in ('gross_tonnage','international_gross_tonnage')} - {None}
    if len(values)!=1:return False
    gt=next(iter(values))
    exact={v for v in family if pos(cat[v]['gross_tonnage'])==gt}
    return exact=={pred}

def read(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def metric(rows):
    resc=[r for r in rows if r['base']!=r['truth'] and r['pred']==r['truth']]
    harm=[r for r in rows if r['base']==r['truth'] and r['pred']!=r['truth']]
    return dict(n=len(rows),vessels=len({r['truth'] for r in rows}),base_errors=sum(r['base']!=r['truth'] for r in rows),changed=sum(r['base']!=r['pred'] for r in rows),rescue=len(resc),harm=len(harm),net=len(resc)-len(harm),rescue_vessels=len({r['truth'] for r in resc}),harm_vessels=len({r['truth'] for r in harm}),reason_counts=dict(Counter(r['reason'] for r in rows)),same_name_repair_oracle=sum(r['base']!=r['truth'] and r['true_same_family'] for r in rows))

def main():
    started=time.perf_counter(); sources=[ROOT/'data/open/vessel_catalog.csv',ROOT/'astra/arena_20261101/baseline/script.py',ROOT/'astra/arena_20261101/baseline/model/en/script.py',ROOT/'astra/arena_20261101/baseline/model/en/model/opus_lib.py',ROOT/'astra/byt5_base112_20261031/paired_probe.py',OUT/'hypothesis.json',OUT/'probe.py']
    cat={r['vessel_id']:r for r in csv.DictReader((ROOT/'data/open/vessel_catalog.csv').open(encoding='utf-8-sig'))};groups=defaultdict(list)
    for v,r in cat.items():groups[canon(r['vessel_name'])].append(v)
    report={'catalog':dict(rows=len(cat),duplicate_name_groups=sum(len(v)>1 for v in groups.values()),duplicate_name_vessels=sum(len(v) for v in groups.values() if len(v)>1)),'folds':{},'evidence_class':'native vessel-heldout112 proxy + synthetic corruption; not C_vote evaluation','actual_C_vote_rescue_harm':None}
    allout=[]
    for fold in (2,4):
        rows_path=ROOT/f'astra/byt5_base112_20261031/fold{fold}/rows.jsonl';queries_path=ROOT/f'astra/noise113/fold{fold}/eval/queries.json';sources.extend([rows_path,queries_path]);raw=read(rows_path);qs={q['query_id']:q for q in json.loads(queries_path.read_text())};out=[]; selection=[]
        for r in raw:
            # Neither label nor pre-existing neural outputs enter selection.
            q={k:v for k,v in qs[r['query_id']].items() if k not in ('target_vessel_id','query_id','sample_id','variant')}
            p,reason=choose(q,r['p112'],r['gap110'],cat,groups)
            selection.append((q,r['p112'],r['gap110'],p))
            if not r['primary']:continue
            out.append(dict(fold=fold,query_id=r['query_id'],base=r['p112'],pred=p,reason=reason,truth=r['target'],variant=r['variant'],scope_audit_pass=audit_scope(q,r['p112'],p,r['gap110'],cat,groups),true_same_family=canon(cat[r['p112']]['vessel_name'])==canon(cat[r['target']]['vessel_name'])))
        # Contract replay with reversed inputs and independent singleton selection.
        assert [choose(q,b,g,cat,groups)[0] for q,b,g,_ in reversed(selection)]==[p for *_,p in reversed(selection)]
        assert all(choose(q,b,g,cat,groups)[0]==p for q,b,g,p in selection[::7])
        cohorts={'all':out,'clean':[r for r in out if r['variant'].startswith('clean')],'noise':[r for r in out if not r['variant'].startswith('clean')],'missing_tonnage':[r for r in out if r['variant'].endswith('missing')]}
        report['folds'][str(fold)]={k:metric(v) for k,v in cohorts.items()};report['folds'][str(fold)]['primary_excluded']=len(raw)-len(out);allout+=out
    # Synthetic mechanics, deliberately including exact-GT wrong-identity counterexample.
    toy={'A':{'vessel_name':'Same Name','gross_tonnage':'100'},'B':{'vessel_name':'Same Name','gross_tonnage':'200'},'C':{'vessel_name':'Other Name','gross_tonnage':'200'}};tg={'SAME NAME':['A','B'],'OTHER NAME':['C']}
    q={'vessel_name':'세임 네임','gross_tonnage':200,'international_gross_tonnage':200}
    cases=[('unique_exact',q,'A',1,'B'),('english',dict(q,vessel_name='Same Name'),'A',1,'A'),('protected',q,'A',4,'A'),('missing',dict(q,gross_tonnage=None,international_gross_tonnage=None),'A',1,'A'),('conflict',dict(q,gross_tonnage=100),'A',1,'A'),('no_homonym',q,'C',1,'C'),('already_exact',q,'B',1,'B')]
    tests=[]
    for name,query,b,g,expected in cases:
        got,reason=choose(query,b,g,toy,tg);assert got==expected;tests.append({'name':name,'passed':True,'reason':reason})
    toy['B']['gross_tonnage']='100';assert choose(dict(q,gross_tonnage=100,international_gross_tonnage=100),'A',1,toy,tg)[0]=='A';tests.append({'name':'ambiguous_equal_gt','passed':True})
    report['synthetic_mechanics']=tests;report['counterexample']='If true vessel A has corrupted input GT=200 but identical-name B has catalog GT=200, unique exact matching changes correct A to wrong B. Identity is not proven by GT.'
    report['contracts']={'reverse_mismatch':0,'singleton_subset_mismatch':0,'return_reason_inconsistency':sum(r['base']!=r['pred'] and r['reason']!='change' for r in allout),'outside_scope_changes':sum(not r['scope_audit_pass'] for r in allout),'english_preservation':'synthetic checked; heldout suite Korean only','missing_gt_changes':sum(r['base']!=r['pred'] for r in allout if r['variant'].endswith('missing'))}
    toy['B']['gross_tonnage']='200'
    audit_cases=[('valid_change',q,'A','B',1,True),('forced_highgap_change',q,'A','B',5,False),('forced_english_change',dict(q,vessel_name='Same Name'),'A','B',1,False),('forced_missing_gt_change',dict(q,gross_tonnage=None,international_gross_tonnage=None),'A','B',1,False),('forced_other_family_change',q,'A','C',1,False)]
    audit_results=[]
    for name,query,b,p,g,expected in audit_cases:
        actual=audit_scope(query,b,p,g,toy,tg);assert actual==expected
        audit_results.append({'name':name,'audit_scope_pass':actual,'expected':expected,'passed':True})
    report['independent_scope_audit_tests']=audit_results
    with zipfile.ZipFile('/Users/sunjae/Downloads/C_vote.zip') as z:
        members={p:str(hashlib.sha256(z.read(p)).hexdigest()) for p in ('script.py','model/en/script.py')}
    report['actual_archive_small_member_sha256']=members
    assert members['script.py']==sha(ROOT/'astra/arena_20261101/baseline/script.py')
    assert members['model/en/script.py']==sha(ROOT/'astra/arena_20261101/baseline/model/en/script.py')
    report['sources_sha256']={str(p):sha(p) for p in sources};report['seconds_including_load_hash_and_checks']=time.perf_counter()-started
    (OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(OUT/'rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in allout));print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':
    with (ROOT/'astra/arena_20261101/CPU.lock').open('rb') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);main()
