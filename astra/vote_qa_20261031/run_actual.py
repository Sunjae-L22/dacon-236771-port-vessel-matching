"""Astra independent rehearsal. All writes stay below astra/. No submission."""
import os,sys,json,time,shutil,zipfile,subprocess,hashlib,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
ENV={k:v for k,v in os.environ.items() if k!='KO_DICT'}
ENV.update(PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1')

def invoke(cmd,cwd,log):
    t=time.monotonic();p=subprocess.run(cmd,cwd=cwd,env=ENV,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    log.write_text(p.stdout);assert p.returncode==0,(p.returncode,str(log),p.stdout[-2000:])
    return time.monotonic()-t

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
    import pandas as pd
    sys.path.insert(0,str(ROOT/'shared/src')); from local_check import corrupted_test
    inp=HERE/'inputs'; t=corrupted_test().sample(frac=1,random_state=1).drop(columns='target_vessel_id')
    t.to_csv(inp/'full.csv',index=False,encoding='utf-8-sig')
    t.sample(frac=.2,random_state=5).to_csv(inp/'subset.csv',index=False,encoding='utf-8-sig')
    catalog=ROOT/'data/open/vessel_catalog.csv'
    files=[ROOT/p for p in ['submissions/93_submit.zip','external/hsw_orig/1031_08_hyb_w.zip','submissions/112_submit.zip','submissions/110_submit.zip','shared/experiments/build_vote.py','shared/experiments/build_fuse.py']]
    (HERE/'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
    out=ROOT/'astra/builds/QA_ONLY_vote93_hsw112_20261031.zip'
    invoke([sys.executable,str(HERE/'sources/build_vote.py'),str(out),*[str(p) for p in files[:3]],'4','ko'],HERE,HERE/'build.log')
    payload=HERE/'payload';payload.mkdir()
    with zipfile.ZipFile(out) as z:z.extractall(payload)
    reports={}
    for tag in ['full','subset']:
        dest=HERE/tag;dest.mkdir()
        secs=invoke([sys.executable,str(HERE/'capture_router.py'),str(payload/'script.py'),str(inp/f'{tag}.csv'),str(catalog),str(dest)],HERE,dest/'run.log')
        a=pd.read_csv(inp/f'{tag}.csv',dtype=str);b=pd.read_csv(dest/'pred.csv',dtype=str)
        assert list(b.columns)==['sample_id','target_vessel_id'] and b.sample_id.tolist()==a.sample_id.tolist()
        assert b.target_vessel_id.notna().all() and b.sample_id.is_unique
        reports[tag]={'rows':len(b),'seconds':secs,'input_order_preserved':True}
        (HERE/'actual_progress.json').write_text(json.dumps(reports,indent=2))
        print(tag,reports[tag],flush=True)
    a=pd.read_csv(HERE/'full/pred.csv',dtype=str).set_index('sample_id');b=pd.read_csv(HERE/'subset/pred.csv',dtype=str).set_index('sample_id')
    reports['subset_mismatches']=int((a.loc[b.index].target_vessel_id!=b.target_vessel_id).sum())
    assert reports['subset_mismatches']==0
    # Independent original110 with gap instrumentation: all3000 includes branch transitions.
    base=HERE/'base110';base.mkdir()
    with zipfile.ZipFile(files[3]) as z:z.extractall(base)
    bf=load('audit_fuse',HERE/'sources/build_fuse.py');(base/'script.py').write_bytes(bf.gap_script((base/'script.py').read_bytes()))
    run="import importlib.util,sys;s=importlib.util.spec_from_file_location('entry',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.run(sys.argv[2],sys.argv[3],sys.argv[4])"
    reports['base110_seconds']=invoke([sys.executable,'-c',run,str(base/'script.py'),str(inp/'full.csv'),str(catalog),str(HERE/'base110.csv')],base,HERE/'base110.log')
    en=pd.read_csv(HERE/'full/en_pred.csv',dtype={'sample_id':str}).set_index('sample_id');old=pd.read_csv(HERE/'base110.csv',dtype={'sample_id':str}).set_index('sample_id')
    import numpy as np
    x=en.ko_gap.to_numpy();y=old.loc[en.index].ko_gap.to_numpy();assert np.allclose(x,y,rtol=0,atol=1e-12,equal_nan=True)
    reports['gap112_vs_original110']={'mismatches':0,'rows':len(en),'max_abs_finite_diff':float(np.nanmax(np.abs(x-y)))}
    (HERE/'ACTUAL_REPORT.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2),flush=True)

if __name__=='__main__':main()
