from pathlib import Path
import sys,subprocess,os,json,fcntl
import pandas as pd
sys.dont_write_bytecode=True
r=Path(__file__).resolve().parent;d=r/'cpu_component_qa';base=r.parent/'arena_20261101/baseline';x=pd.read_csv(d/'test.csv',dtype=str,keep_default_na=False);x.iloc[::-1].to_csv(d/'reverse.csv',index=False);x.iloc[:1].to_csv(d/'singleton.csv',index=False)
env=dict(os.environ,TMPDIR=str(r/'tmp'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');lock=(r/'CPU.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
report={}
for tag in ['en','hs']:
 for mode in ['reverse','singleton']:
  out=d/f'{tag}_{mode}.csv'
  p=subprocess.run([sys.executable,str(d/f'{tag}_plain_launch.py'),str(r/'package'),str(base/f'model/{tag}/script.py'),str(d/f'{mode}.csv'),str(r/'package/data/vessel_catalog.csv'),str(out)],cwd=base/f'model/{tag}',env=env,capture_output=True,text=True)
  (d/f'{tag}_{mode}.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
  a=pd.read_csv(d/f'{tag}_plain.csv',dtype=str).set_index('sample_id');b=pd.read_csv(out,dtype=str).set_index('sample_id');assert a.loc[b.index].equals(b)
  report[tag+'_'+mode]={'rows':len(b),'mismatch':0};print(tag,mode,'PASS',flush=True)
(d/'ORDER_RESULT.json').write_text(json.dumps(report,indent=2))
