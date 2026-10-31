"""One predeclared metadata rule. No model fitting, threshold search or ByT5 use.

Scope: Hangul; original110 gap<4; both gross tonnage fields finite, positive,
and exactly equal; that GT identifies one vessel in the FULL official catalog;
that vessel is not a fit-training target. Override the current prediction.
Labels are accessed only after the complete prediction vector is frozen.
"""
import os, sys
sys.dont_write_bytecode = True
os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import csv, json, math, hashlib, pickle, time, unicodedata, fcntl
from pathlib import Path
from collections import defaultdict

ROOT=Path('/Users/sunjae/Desktop/항만')
OUT=Path(__file__).resolve().parent
os.environ['TMPDIR']=str(OUT/'tmp')
SOURCES={}
def record(p):
    p=Path(p); SOURCES[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest(); return p
def jl(p): return [json.loads(s) for s in record(p).read_text().splitlines() if s.strip()]
def positive(x):
    try: v=float(x)
    except (TypeError,ValueError): return None
    return v if math.isfinite(v) and v>0 else None
def hangul(s): return any('\uac00'<=c<='\ud7a3' for c in unicodedata.normalize('NFC',str(s)))

def predict(q, base, gap, gt_index, excluded):
    a,b=positive(q.get('gross_tonnage')),positive(q.get('international_gross_tonnage'))
    if not hangul(q.get('vessel_name','')) or not math.isfinite(gap) or gap>=4: return base,False
    if a is None or a!=b: return base,False
    candidates=gt_index.get(a,())
    if len(candidates)!=1 or candidates[0] in excluded: return base,False
    return candidates[0],True

def audit_change(q, base, pred, original_gap, fixed_gt_index, fit_targets):
    """Reconstruct frozen scope without reading predict()'s eligibility flag."""
    if pred == base:
        return True
    if not hangul(q.get('vessel_name','')) or not math.isfinite(original_gap) or original_gap >= 4:
        return False
    first = positive(q.get('gross_tonnage'))
    second = positive(q.get('international_gross_tonnage'))
    if first is None or second is None or first != second:
        return False
    ids = fixed_gt_index.get(first,())
    return len(ids) == 1 and ids[0] == pred and pred not in fit_targets

def main():
    start=time.monotonic()
    cat=list(csv.DictReader(record(ROOT/'data/open/vessel_catalog.csv').open(encoding='utf-8-sig')))
    gt=defaultdict(list); names=defaultdict(list)
    for c in cat:
        v=positive(c['gross_tonnage'])
        if v is not None: gt[v].append(c['vessel_id'])
        names[unicodedata.normalize('NFC',c['vessel_name']).upper().strip()].append(c['vessel_id'])
    homonyms={i for ids in names.values() if len(ids)>1 for i in ids}
    report={'rule':__doc__,'evaluation':'112 held-out recipe proxy, not actual C_vote', 'catalog_rows':len(cat),
            'catalog_unique_gt_values':sum(len(v)==1 for v in gt.values()),'folds':{}}
    for fold in (2,4):
        rows=jl(ROOT/f'astra/byt5_base112_20261031/fold{fold}/rows.jsonl')
        qs=json.loads(record(ROOT/f'astra/noise113/fold{fold}/eval/queries.json').read_text())
        stats=pickle.loads(record(ROOT/f'astra/byt5_reuse_20261031/package/baseline/fold{fold}/stats.pkl').read_bytes())
        excluded=set(stats['cs_map'].values())
        assert len(rows)==len(qs)
        inputs=[]
        for q,r in zip(qs,rows):
            assert q['query_id']==r['query_id']
            inputs.append((q,r['p112'],float(r['gap110'])))
        t=time.monotonic()
        frozen=[predict(q,b,g,gt,excluded) for q,b,g in inputs]
        inference_seconds=time.monotonic()-t
        assert frozen==list(reversed([predict(q,b,g,gt,excluded) for q,b,g in reversed(inputs)]))
        assert frozen[::7]==[predict(q,b,g,gt,excluded) for q,b,g in inputs[::7]]
        english_mutation_fail=0; missing_mutation_fail=0
        for (q,b,g),p in zip(inputs,frozen):
            e=dict(q,vessel_name='SYNTHETIC ENGLISH NAME')
            m=dict(q,gross_tonnage=None,international_gross_tonnage=None)
            english_mutation_fail+=predict(e,b,g,gt,excluded)!=(b,False)
            missing_mutation_fail+=predict(m,b,g,gt,excluded)!=(b,False)
        # Evaluation begins here: no truth enters predict(), construction or thresholds.
        evaluated=[]
        for q,r,(pred,eligible) in zip(qs,rows,frozen):
            assert r['target']==q['target_vessel_id']
            if not r['primary']: continue
            assert r['target'] not in excluded, 'fit target leakage in held-out primary set'
            evaluated.append(dict(query_id=r['query_id'],variant=r['variant'],target=r['target'],base=r['p112'],
                pred=pred,eligible=eligible,changed=pred!=r['p112'],rescue=r['p112']!=r['target'] and pred==r['target'],
                harm=r['p112']==r['target'] and pred!=r['target'],homonym=r['target'] in homonyms,
                scope_audit_pass=audit_change(q,r['p112'],pred,float(r['gap110']),gt,excluded),
                missing=positive(q.get('gross_tonnage')) is None or positive(q.get('international_gross_tonnage')) is None,
                gate110_gap=r['gap110'],target_was112_rescue=r['p110']!=r['target'] and r['p112']==r['target']))
        def metrics(rs):
            rescue=sum(x['rescue'] for x in rs); harm=sum(x['harm'] for x in rs)
            return dict(n=len(rs),vessels=len({x['target'] for x in rs}),base_correct=sum(x['base']==x['target'] for x in rs),
                eligible=sum(x['eligible'] for x in rs),eligible_base_errors=sum(x['eligible'] and x['base']!=x['target'] for x in rs),
                changed=sum(x['changed'] for x in rs),rescue=rescue,harm=harm,net=rescue-harm,
                rescue_vessels=len({x['target'] for x in rs if x['rescue']}),harm_vessels=len({x['target'] for x in rs if x['harm']}),
                outside_scope=sum(x['changed'] and not x['scope_audit_pass'] for x in rs),
                returned_eligibility_inconsistency=sum(x['changed'] and not x['eligible'] for x in rs),
                harmed112_rescues=sum(x['harm'] and x['target_was112_rescue'] for x in rs))
        cohorts={'all':evaluated,'clean':[x for x in evaluated if x['variant'].startswith('clean')],
            'synthetic_noise':[x for x in evaluated if not x['variant'].startswith('clean')],
            'missing_tonnage':[x for x in evaluated if x['missing']], 'homonym_targets':[x for x in evaluated if x['homonym']]}
        report['folds'][str(fold)]={'cohorts':{k:metrics(v) for k,v in cohorts.items()},
            'seconds_existing_rows':inference_seconds,'all_input_rows':len(inputs),'reverse_mismatch':0,'subset_mismatch':0,
            'synthetic_english_preservation_mismatch':english_mutation_fail,'missing_tonnage_preservation_mismatch':missing_mutation_fail}
        (OUT/f'fold{fold}_paired.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in evaluated))
    # Mechanics-only counterexamples: ambiguity and missing information must abstain.
    q={'vessel_name':'한글','gross_tonnage':100,'international_gross_tonnage':100}
    assert predict(q,'base',1,{100:['a','b']},set())==('base',False)
    assert predict(q,'base',1,{100:['a']},{'a'})==('base',False)
    assert predict(q,'base',4,{100:['a']},set())==('base',False)
    assert predict(dict(q,international_gross_tonnage=101),'base',1,{100:['a']},set())==('base',False)
    audit_cases=[
        ('valid_change',q,'a',1,{100:['a']},set(),True),
        ('forced_gap5_change',q,'a',5,{100:['a']},set(),False),
        ('forced_english_change',dict(q,vessel_name='ENGLISH'),'a',1,{100:['a']},set(),False),
        ('forced_missing_gt_change',dict(q,gross_tonnage=None),'a',1,{100:['a']},set(),False),
        ('forced_conflicting_gt_change',dict(q,international_gross_tonnage=101),'a',1,{100:['a']},set(),False),
        ('forced_duplicate_gt_change',q,'a',1,{100:['a','b']},set(),False),
        ('forced_fit_target_change',q,'a',1,{100:['a']},{'a'},False),
        ('forced_wrong_candidate_change',q,'b',1,{100:['a']},set(),False),
    ]
    audit_results=[]
    for name,row,pred,gap,index,fit,expected in audit_cases:
        got=audit_change(row,'base',pred,gap,index,fit)
        assert got==expected,name
        audit_results.append(dict(name=name,scope_pass=got,expected=expected,passed=True))
    report['independent_scope_audit_controls']=audit_results
    report['mechanics_ambiguity_exclusion_gate_conflicting_gt_pass']=True
    report['seconds_total']=time.monotonic()-start
    record(__file__)
    report['source_sha256']=SOURCES
    (OUT/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    with (ROOT/'astra/arena_20261101/CPU.lock').open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX); main()
