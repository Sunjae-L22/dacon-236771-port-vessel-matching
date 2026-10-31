"""Run official local_check, original110 comparison, and independent extra processes."""
import os,sys,json,subprocess,zipfile,time,tempfile,re,shutil
from pathlib import Path
import pandas as pd

def command(args,cwd,env,log):
    started=time.time()
    p=subprocess.run(args,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    # Publish after completion; workspace snapshots may replace an open log inode.
    Path(log).write_text(p.stdout)
    if p.returncode:raise RuntimeError(f'command failed; see {log}')
    return time.time()-started

def qa(base,candidate,out):
    here=Path(__file__).resolve().parent;out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    env={k:v for k,v in os.environ.items() if k!='KO_DICT'}
    env.update(PYTHONHASHSEED='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',ASTRA_TRACE=str(out/'trace'),LOCAL_CHECK_SAVE=str(out/'candidate3000.csv'))
    command([sys.executable,'-u',str(here/'local_check.py'),str(candidate)],tempfile.gettempdir(),env,out/'local_check.log')
    log=(out/'local_check.log').read_text();m=re.search(r'rows 3000\s+time ([0-9.]+)s',log)
    assert m and 'subset 600 rows, mismatches 0' in log,log[-2000:]
    trace=pd.DataFrame(json.loads((out/'trace.3000.json').read_text())).set_index('sample_id')
    pred=pd.read_csv(out/'candidate3000.csv').set_index('sample_id');pred.index=pred.index.astype(str)
    assert len(trace)==3000 and (trace.loc[pred.index].pred==pred.target_vessel_id).all()
    changed=trace.pred!=trace.baseline
    report=dict(rows=3000,seconds=float(m[1]),subset600_mismatches=0,changed_ko=int((changed&trace.is_ko).sum()),changed_en=int((changed&~trace.is_ko).sum()),outside_scope=int((changed&~trace.eligible).sum()),easy_changed=int((changed&(trace.gap.fillna(99)>=4)).sum()))
    assert report['changed_en']==report['outside_scope']==report['easy_changed']==0
    # Use original local_check corruption (frozen input), identical ordering.
    t=pd.read_csv(here/'data/local_check3000_with_truth.csv').sample(frac=1,random_state=1).drop(columns='target_vessel_id')
    t.sample_id=t.sample_id.astype(str)
    with tempfile.TemporaryDirectory(prefix='astra_qa_') as d:
        work=Path(d);(work/'data').mkdir();shutil.copy(here/'data/open/vessel_catalog.csv',work/'data')
        with zipfile.ZipFile(base) as z:z.extractall(work)
        t.to_csv(work/'data/test.csv',index=False,encoding='utf-8-sig')
        baseenv={k:v for k,v in env.items() if k not in ['ASTRA_TRACE','LOCAL_CHECK_SAVE']}
        command([sys.executable,'script.py'],work,baseenv,out/'baseline110.log')
        bp=pd.read_csv(work/'output/submission.csv').set_index('sample_id');bp.index=bp.index.astype(str)
        report['trace_baseline_mismatches']=int((bp.target_vessel_id!=trace.loc[bp.index].baseline).sum());assert report['trace_baseline_mismatches']==0
        with zipfile.ZipFile(candidate) as z:z.extractall(work)
        # Up to4 cases including changes, eligible, easy and English when available.
        picks=[]
        for flag in [changed,trace.eligible,trace.is_ko&~trace.eligible,~trace.is_ko]:
            ids=trace.index[flag].tolist()
            if ids:picks.append(next((i for i in ids if i not in picks),ids[0]))
        picks=list(dict.fromkeys(picks));small=t.set_index('sample_id').loc[picks].reset_index()
        cases=[('forward',small),('reverse',small.iloc[::-1])]+[(f'single{i}',small.iloc[[i]]) for i in range(min(2,len(small)))]
        checks=[]
        for label,sub in cases:
            sub.to_csv(work/'data/test.csv',index=False,encoding='utf-8-sig')
            command([sys.executable,'script.py'],work,baseenv,out/f'{label}.log')
            part=pd.read_csv(work/'output/submission.csv').set_index('sample_id');part.index=part.index.astype(str)
            mism=int((part.target_vessel_id!=pred.loc[part.index].target_vessel_id).sum());assert mism==0
            checks.append(dict(case=label,rows=len(part),mismatches=mism))
        report['extra_process_checks']=checks
    import torch
    report.update(status='QA_PASSED_UNREVIEWED',gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',l4_runtime_verified=False,warning='G4 timing is not an L4 timing proof; local_check accuracy on excluded train targets is not validation accuracy.')
    (out/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));return report

if __name__=='__main__':qa(*sys.argv[1:])
