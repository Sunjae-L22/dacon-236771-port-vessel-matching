"""Read-only Erde source; local run adapter and Korean-only QA, never a ZIP."""
import sys,os,json,time,hashlib,shutil,subprocess,difflib
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
SRC=ROOT/'external/erde/submissions/submit_v42b_fuzzy500';PKG=HERE/'package'
ENV=dict(os.environ,PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1');ENV.pop('KO_DICT',None)
RUN="import sys,importlib.util;s=importlib.util.spec_from_file_location('entry',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.run(sys.argv[2],sys.argv[3],sys.argv[4])"
def invoke(script,inp,out,log):
    t=time.monotonic();p=subprocess.run([sys.executable,'-c',RUN,str(script),str(inp),str(ROOT/'data/open/vessel_catalog.csv'),str(out)],cwd=PKG,env=ENV,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    log.write_text(p.stdout);assert p.returncode==0,(p.returncode,p.stdout[-2500:]);return time.monotonic()-t
def main():
    import pandas as pd
    shutil.copytree(SRC,PKG,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy(ROOT/'data/open/train.csv',PKG/'model/train.csv')
    before=(SRC/'script.py').read_text();after=before
    edits=[('def main():','def run(test_path="data/test.csv", catalog_path="data/vessel_catalog.csv", output_path="output/submission.csv"):'),('pd.read_csv(find("test.csv"),','pd.read_csv(test_path,'),('pd.read_csv(find("vessel_catalog.csv"),','pd.read_csv(catalog_path,'),('    out_dir = Path.cwd() / "output"','    output_path = Path(output_path)\n    out_dir = output_path.parent'),('        out_dir / "submission.csv",','        output_path,'),('    main()','    run()')]
    for old,new in edits:
        assert after.count(old)==1,old;after=after.replace(old,new)
    (PKG/'script.py').write_text(after)
    (HERE/'adapter.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='Erde original script.py',tofile='astra adapter script.py')))
    manifest={str(p.relative_to(SRC)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SRC.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    manifest['added_train.csv']=hashlib.sha256((ROOT/'data/open/train.csv').read_bytes()).hexdigest()
    (HERE/'source_hashes.json').write_text(json.dumps(manifest,indent=2))
    t=pd.read_csv(HERE/'korean.csv',dtype=str,keep_default_na=False)
    # 291 rows crosses original CH256 boundary; reverse and subset place them in new blocks.
    tests={'full':t,'reverse':t.iloc[::-1],'subset':t.sample(n=min(64,len(t)),random_state=912)}
    report={'reference':'local corrupted train rows; NOT accuracy or Public prediction','adapter_only':True,'zip_created':False,'runs':{}}
    for tag,df in tests.items():
        inp=HERE/f'{tag}_input.csv';df.to_csv(inp,index=False,encoding='utf-8-sig')
        sec=invoke(PKG/'script.py',inp,HERE/f'{tag}_pred.csv',HERE/f'{tag}.log')
        pred=pd.read_csv(HERE/f'{tag}_pred.csv',dtype=str)
        assert pred.sample_id.tolist()==df.sample_id.tolist() and pred.target_vessel_id.notna().all()
        report['runs'][tag]={'rows':len(df),'seconds':sec,'input_order_preserved':True}
        if tag!='full':
            full=pd.read_csv(HERE/'full_pred.csv',dtype=str).set_index('sample_id');p=pred.set_index('sample_id')
            mism=int((full.loc[p.index].target_vessel_id!=p.target_vessel_id).sum());report['runs'][tag]['mismatches']=mism;assert mism==0
        (HERE/'RUNTIME_INDEPENDENCE.json').write_text(json.dumps(report,indent=2));print(tag,report['runs'][tag],flush=True)
    # Source main vs adapter on same subset, isolated cwd and explicit data files.
    (PKG/'original_entry.py').write_text(before);data=PKG/'data';data.mkdir()
    shutil.copy(HERE/'subset_input.csv',data/'test.csv');shutil.copy(ROOT/'data/open/vessel_catalog.csv',data/'vessel_catalog.csv')
    t0=time.monotonic();p=subprocess.run([sys.executable,'original_entry.py'],cwd=PKG,env=ENV,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (HERE/'original_main.log').write_text(p.stdout);assert p.returncode==0,p.stdout[-2000:]
    orig=pd.read_csv(PKG/'output/submission.csv',dtype=str);ada=pd.read_csv(HERE/'subset_pred.csv',dtype=str)
    assert orig.equals(ada);report['adapter_vs_original_main']={'rows':len(orig),'mismatches':0,'seconds':time.monotonic()-t0}
    (HERE/'RUNTIME_INDEPENDENCE.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
