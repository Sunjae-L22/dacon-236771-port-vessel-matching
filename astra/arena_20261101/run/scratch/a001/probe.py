import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import sys
sys.dont_write_bytecode=True
import csv,json,math,re,time,hashlib,fcntl,zipfile
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path('/Users/sunjae/Desktop/항만')
HERE=Path(__file__).resolve().parent
os.environ['TMPDIR']=str(HERE)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def num(x):
    try:
        v=float(x)
        return v if math.isfinite(v) and v>0 else None
    except (ValueError,TypeError): return None

def choose(row,base,gap,top5,gt_by_id,gt_ids):
    if not re.search('[가-힣]',row.get('vessel_name','')) or not math.isfinite(gap) or gap>=4: return base
    gt,igt=num(row.get('gross_tonnage')),num(row.get('international_gross_tonnage'))
    if gt is None or igt!=gt: return base
    matches=gt_ids.get(gt,[])
    if len(matches)!=1 or matches[0] not in top5: return base
    bgt=gt_by_id.get(base)
    if bgt is None or abs(bgt-gt)/gt<=.1: return base
    return matches[0]

def metrics(rs):
    changed=[r for r in rs if r['base']!=r['new']]
    resc=[r for r in changed if r['new']==r['truth']]
    harm=[r for r in changed if r['base']==r['truth']]
    return dict(rows=len(rs),vessels=len({r['truth'] for r in rs}),base_correct=sum(r['base']==r['truth'] for r in rs),changed=len(changed),rescue=len(resc),harm=len(harm),net=len(resc)-len(harm),rescue_vessels=len({r['truth'] for r in resc}),harm_vessels=len({r['truth'] for r in harm}))

def main():
    start=time.perf_counter()
    paths=[ROOT/'data/open/vessel_catalog.csv',HERE/'hypothesis.json',HERE/'probe.py']
    with paths[0].open(encoding='utf-8-sig',newline='') as f: cat=list(csv.DictReader(f))
    gt_by_id={r['vessel_id']:num(r['gross_tonnage']) for r in cat}
    gt_ids=defaultdict(list)
    for vid,gt in gt_by_id.items():
        if gt is not None:gt_ids[gt].append(vid)
    result={'candidate':'unique_exact_gross_tonnage_veto_v1','catalog_rows':len(cat),'catalog_unique_gt_vessels':sum(len(v)==1 for v in gt_ids.values()),'folds':{}}
    for fold in (2,4):
        qpath=ROOT/f'astra/noise113/fold{fold}/eval/queries.json'
        ppath=ROOT/f'astra/byt5_base112_20261031/fold{fold}/packets112.jsonl'
        paths.extend([qpath,ppath])
        qs={q['query_id']:q for q in json.loads(qpath.read_text())}
        ps=[json.loads(s) for s in ppath.read_text().splitlines()]
        rs=[]
        for p in ps:
            if not p['primary']:continue
            q=qs[p['query_id']]
            top5=[c['id'] for c in p['candidates'][:5]]
            gap=float(p['gate110_gap'])
            base=p['baseline']
            new=choose(q,base,gap,top5,gt_by_id,gt_ids)
            gt,igt=num(q.get('gross_tonnage')),num(q.get('international_gross_tonnage'))
            scope=bool(re.search('[가-힣]',q['vessel_name'])) and gap<4 and gt is not None and igt==gt
            rs.append(dict(query_id=p['query_id'],variant=p['variant'],base=base,new=new,truth=p['target'],scope=scope,gt=gt,igt=igt,globally_unique=gt is not None and len(gt_ids.get(gt,[]))==1,truth_top5=p['target'] in top5,truth_exact_gt=gt is not None and gt_by_id.get(p['target'])==gt))
        changed=[r for r in rs if r['base']!=r['new']]
        # Repeat actual function in reverse/subset order; outputs must depend only on one row.
        original={r['query_id']:r['new'] for r in rs}
        def rerun(seq):
            return {p['query_id']:choose(qs[p['query_id']],p['baseline'],float(p['gate110_gap']),[c['id'] for c in p['candidates'][:5]],gt_by_id,gt_ids) for p in seq if p['primary']}
        reverse=rerun(reversed(ps)); subset=rerun(ps[::7])
        result['folds'][str(fold)]={'all':metrics(rs),'clean':metrics([r for r in rs if r['variant'].startswith('clean')]),'noise':metrics([r for r in rs if not r['variant'].startswith('clean')]),'missing':metrics([r for r in rs if r['gt'] is None]),'scope_outside_changes':sum(not r['scope'] for r in changed),'scope_rows':sum(r['scope'] for r in rs),'scope_baseline_errors':sum(r['scope'] and r['base']!=r['truth'] for r in rs),'scope_errors_with_true_exact_gt':sum(r['scope'] and r['base']!=r['truth'] and r['truth_exact_gt'] for r in rs),'scope_errors_with_unique_true_gt':sum(r['scope'] and r['base']!=r['truth'] and r['truth_exact_gt'] and r['globally_unique'] for r in rs),'scope_errors_unique_true_gt_top5':sum(r['scope'] and r['base']!=r['truth'] and r['truth_exact_gt'] and r['globally_unique'] and r['truth_top5'] for r in rs),'reverse_mismatches':sum(original[k]!=v for k,v in reverse.items()),'subset_mismatches':sum(original[k]!=v for k,v in subset.items()),'changed_rows':changed}
        with (HERE/f'fold{fold}_evaluated.jsonl').open('w') as out:
            for r in rs:out.write(json.dumps(r,ensure_ascii=False)+'\n')
    # Mechanical positive/negative controls, including colliding catalog GT and missing metadata.
    row={'vessel_name':'가나다','gross_tonnage':100,'international_gross_tonnage':100}
    args=('A',1,['A','B'],{'A':200.,'B':100.},{100.:['B'],200.:['A']})
    checks={'positive_rescue':choose(row,*args)=='B','english_preserved':choose(dict(row,vessel_name='ABC'),*args)=='A','easy_preserved':choose(row,'A',4,*args[2:])=='A','missing_preserved':choose(dict(row,gross_tonnage=None),*args)=='A','disagreeing_fields_preserved':choose(dict(row,international_gross_tonnage=99),*args)=='A','homonym_gt_abstains':choose(row,'A',1,['A','B'],args[3],{100.:['B','C']})=='A'}
    assert all(checks.values()), checks
    result['mechanics_controls']=checks
    baseline=ROOT/'astra/arena_20261101/baseline/script.py'
    en=ROOT/'astra/arena_20261101/baseline/model/en/script.py'
    paths.extend([baseline,en])
    with zipfile.ZipFile('/Users/sunjae/Downloads/C_vote.zip') as z:
        for path,member in [(baseline,'script.py'),(en,'model/en/script.py')]:
            assert z.read(member)==path.read_bytes()
    result['actual_zip_small_source_match']=True
    result['input_sha256']={str(p):sha(p) for p in paths}
    result['seconds']=time.perf_counter()-start
    (HERE/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    # Existing shared lock opened without changing its contents; this serializes CPU probes.
    with open(ROOT/'astra/arena_20261101/CPU.lock','r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main()
