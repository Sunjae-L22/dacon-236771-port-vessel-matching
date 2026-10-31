from pathlib import Path
import subprocess,sys,os,json,fcntl
sys.dont_write_bytecode=True
r=Path(__file__).resolve().parent;base=r.parent/'arena_20261101/baseline';d=r/'cpu_component_qa';env=dict(os.environ,ASTRA_TRACE_DIR=str(d/'traces'),TMPDIR=str(r/'tmp'))
lock=(r/'CPU.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
for tag in ['en','hs']:
 for mode in ['plain','traced']:
  code="import sys;sys.path.insert(0,sys.argv[1]);from trace_hooks import run_child;run_child(*sys.argv[2:])" if mode=='traced' else "import sys,inspect;sys.path.insert(0,sys.argv[1]);from trace_hooks import load_entry;m=load_entry(sys.argv[2]);k='output_path' if 'output_path' in inspect.signature(m.run).parameters else 'out';m.run(test_path=sys.argv[3],catalog_path=sys.argv[4],**{k:sys.argv[5]})"
  helper=d/f'{tag}_{mode}_launch.py';helper.write_text(code)
  p=subprocess.run([sys.executable,str(helper),str(r/'package'),str(base/f'model/{tag}/script.py'),str(d/'test.csv'),str(r/'package/data/vessel_catalog.csv'),str(d/f'{tag}_{mode}.csv')],cwd=base/f'model/{tag}',env=env,capture_output=True,text=True)
  (d/f'{tag}_{mode}.log').write_text(p.stdout+p.stderr);print(tag,mode,p.returncode,flush=True)
  if p.returncode:print(p.stderr[-2000:],flush=True);raise SystemExit(p.returncode)
import pandas as pd
report={}
for tag in ['en','hs']:
 a=pd.read_csv(d/f'{tag}_plain.csv');b=pd.read_csv(d/f'{tag}_traced.csv');assert a.equals(b)
 tr=[json.loads(s) for s in (d/f'traces/{tag}_candidates.jsonl').read_text().splitlines()];pred=b.set_index('sample_id').target_vessel_id.to_dict();assert all(pred[t['sample_id']]==t['top20_ids'][0] for t in tr)
 report[tag]={'rows':len(a),'plain_traced_mismatch':0,'traced_rows':len(tr),'trace_top1_matches_output':True}
 if tag=='en':
  gaps=a.set_index('sample_id').ko_gap.to_dict();assert all(abs(float(gaps[t['sample_id']])-t['gap110'])<1e-9 for t in tr)
  report[tag].update(original_110_gap_trace_mismatch=0,kc_scored_trace_rows=sum(t.get('score_family')=='kc112' for t in tr))
(d/'RESULT.json').write_text(json.dumps(report,indent=2));print(report)
