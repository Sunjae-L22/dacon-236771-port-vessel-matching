"""Run actual C_vote router with observation-only child hooks; fail closed for measurement."""
from pathlib import Path
import sys,os,json,shutil,time,importlib.util,hashlib
sys.dont_write_bytecode=True

def main(payload,test,catalog,dest,trace=True):
    import pandas as pd
    payload,test,catalog,dest=map(lambda p:Path(p).resolve(),[payload,test,catalog,dest]);dest.mkdir(parents=True,exist_ok=False)
    os.environ.update(PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1',ASTRA_TRACE_DIR=str(dest/'internal'))
    (dest/'tmp').mkdir();os.environ['TMPDIR']=str(dest/'tmp')
    import tempfile
    tempfile.tempdir=str(dest/'tmp')
    s=importlib.util.spec_from_file_location('router',payload/'script.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    if trace:
        helper=str(Path(__file__).resolve().parent)
        m.RUN=f"import sys;sys.path.insert(0,{helper!r});from trace_hooks import run_child;run_child(*sys.argv[1:5])"
    finish=m.finish;original_start=m.start;starts=[];handles=[]
    def start(*args):
        job=original_start(*args)
        if job:
            starts.append(dict(pkg=job[0],pid=job[1].pid,rows=job[3],start=job[4]));handles.append(job[1])
        return job
    def checked_finish(job,*args):
        result=finish(job,*args)
        if job is not None:
            pkg,p,dst,n,t=job
            if p.returncode or len(result)!=n:raise RuntimeError(f'{pkg}: baseline voter failed/incomplete; do not accept fallback')
            raw=pd.read_csv(dst,dtype=str,keep_default_na=False);inp=pd.read_csv(Path(dst).with_name(pkg+'_test.csv'),dtype=str,keep_default_na=False)
            if raw.sample_id.duplicated().any() or set(raw.sample_id)!=set(inp.sample_id) or raw.target_vessel_id.eq('').any():raise RuntimeError('Invalid component output')
            shutil.copy(dst,dest/f'{pkg}_pred.csv')
        return result
    m.start=start;m.finish=checked_finish;t=time.perf_counter()
    try:m.run(str(test),str(catalog),str(dest/'pred.csv'))
    finally:
        for proc in handles:
            if proc.poll() is None:proc.kill()
        for proc in handles:proc.wait()
    secs=time.perf_counter()-t
    inp=pd.read_csv(test,dtype=str,keep_default_na=False);pred=pd.read_csv(dest/'pred.csv',dtype=str,keep_default_na=False)
    if pred.sample_id.tolist()!=inp.sample_id.tolist():raise RuntimeError('Output order mismatch')
    (dest/'run_meta.json').write_text(json.dumps(dict(rows=len(inp),seconds_end_to_end=secs,trace_enabled=trace,all_voters_succeeded=True,processes=starts,test_sha256=hashlib.sha256(test.read_bytes()).hexdigest(),catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),router_sha256=hashlib.sha256((payload/'script.py').read_bytes()).hexdigest()),indent=2))
    return secs

if __name__=='__main__':main(*sys.argv[1:5],trace='--plain' not in sys.argv[5:])
