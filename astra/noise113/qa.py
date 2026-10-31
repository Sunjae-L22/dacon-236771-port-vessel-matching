"""Astra: repository local_check plus output-vs110 row trace and adversarial single-row checks."""
from pathlib import Path
import os,sys,json,re,subprocess,tempfile,zipfile,shutil,argparse,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent

def main(rule,variant=None):
    tag=f'{variant}_{rule}' if variant else rule
    manifest=json.loads((OUT/f'build_{tag}.json').read_text());dest=OUT/f'qa_{tag}';dest.mkdir(exist_ok=True)
    trace=dest/'trace.jsonl';assert not trace.exists(),'Do not silently append to a prior QA run'
    env=dict(os.environ,PYTHONHASHSEED='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',ASTRA_TRACE=str(trace),LOCAL_CHECK_SAVE=str(dest/'pred.csv'))
    env.pop('KO_DICT',None)
    p=subprocess.run([sys.executable,str(ROOT/'shared/src/local_check.py'),manifest['path']],cwd='/tmp',env=env,capture_output=True,text=True)
    (dest/'local_check.log').write_text(p.stdout+p.stderr)
    assert p.returncode==0,p.stderr[-3000:]
    assert 'subset 600 rows, mismatches 0' in p.stdout,p.stdout
    a=pd.read_csv(dest/'pred.csv').set_index('sample_id').target_vessel_id
    b=pd.read_csv(OUT/'pred_110.csv').set_index('sample_id').target_vessel_id
    assert a.index.is_unique and set(a.index)==set(b.index) and len(a)==3000
    a=a.reindex(b.index);changed=a.ne(b)
    logs=pd.read_json(trace,lines=True);assert len(logs)==3600
    full=logs.iloc[:3000].set_index('sample_id').loc[b.index]
    assert full.index.is_unique
    assert full.baseline.eq(b).all() and full.selected.eq(a).all()
    assert not (changed & ~full.eligible).any()
    ko=full.route.eq('ko');easy=ko&~full.hard
    assert not changed[~ko].any() and not changed[easy].any()
    missing=ko & full.tonnage_missing.fillna(False).astype(bool) if 'tonnage_missing' in full else pd.Series(False,index=full.index)
    if rule=='noise_with_tonnage':assert not changed[missing].any()
    assert logs.groupby('sample_id').selected.nunique().max()==1
    # Separate-process runs on carefully interleaved scopes, plus four singleton runs.
    sys.path.insert(0,str(ROOT/'shared/src'))
    from local_check import corrupted_test
    t=corrupted_test().set_index('sample_id')
    buckets=[list(full.index[ko&full.eligible&changed][:4]),list(full.index[easy][:4]),list(full.index[~ko][:4])]
    if rule=='noise_with_tonnage':buckets.append(list(full.index[missing][:4]))
    interleaved=[bucket[j] for j in range(4) for bucket in buckets if len(bucket)>j]
    assert interleaved
    orders=[interleaved,interleaved[::-1]]+[[x] for x in interleaved[:4]]
    env2=dict(env);env2.pop('ASTRA_TRACE');env2.pop('LOCAL_CHECK_SAVE')
    with tempfile.TemporaryDirectory() as tmp:
        work=Path(tmp)
        with zipfile.ZipFile(manifest['path']) as z:z.extractall(work)
        (work/'data').mkdir(exist_ok=True);shutil.copy(ROOT/'data/open/vessel_catalog.csv',work/'data/vessel_catalog.csv')
        for order in orders:
            q=t.loc[order].reset_index().drop(columns=['target_vessel_id'])
            q.to_csv(work/'data/test.csv',index=False,encoding='utf-8-sig')
            run=subprocess.run([sys.executable,'script.py'],cwd=work,env=env2,capture_output=True,text=True)
            assert run.returncode==0,run.stderr[-1500:]
            out=pd.read_csv(work/'output/submission.csv').set_index('sample_id').target_vessel_id
            assert out.eq(a.loc[out.index]).all()
    result=dict(rule=rule,model_variant=variant or 'three_extra_copies',sha256=hashlib.sha256(Path(manifest['path']).read_bytes()).hexdigest(),local_check='PASS',
        seconds=float(re.search(r'time ([\d.]+)s',p.stdout)[1]),rows=3000,subset_rows=600,subset_mismatches=0,
        korean_rows=int(ko.sum()),english_rows=int((~ko).sum()),korean_changed=int(changed[ko].sum()),english_changed=int(changed[~ko].sum()),
        easy_rows=int(easy.sum()),easy_changed=int(changed[easy].sum()),tonnage_missing_rows=int(missing.sum()),tonnage_missing_changed=int(changed[missing].sum()),outside_scope_changed=int((changed&~full.eligible).sum()),
        full_trace_baseline_equal110=True,full_trace_selected_equal_candidate=True,
        adversarial_orders=len(orders),singletons=4,adversarial_mismatches=0)
    (dest/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('rule');p.add_argument('--variant',choices=['light']);a=p.parse_args();main(a.rule,a.variant)
