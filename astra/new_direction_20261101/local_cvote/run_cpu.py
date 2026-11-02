from pathlib import Path
import sys,os,inspect,time,json
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'astra/arena_highgain_20261101/package'))
from trace_hooks import load_entry,attach_112_trace,attach_hsw_trace
pkg=sys.argv[1];script=ROOT/f'astra/arena_20261101/baseline/model/{pkg}/script.py';dest=HERE/'eval';t=time.time();os.chdir(script.parent);m=load_entry(script)
restore=attach_112_trace(script,dest/'en_trace.jsonl') if pkg=='en' else attach_hsw_trace(dest/'hs_trace.jsonl')
k='output_path' if 'output_path' in inspect.signature(m.run).parameters else 'out'
try:m.run(test_path=str(dest/'test.csv'),catalog_path=str(ROOT/'data/open/vessel_catalog.csv'),**{k:str(dest/f'{pkg}_pred.csv')})
finally:restore()
(dest/f'{pkg}_timing.json').write_text(json.dumps({'seconds':time.time()-t,'rows':400}));print('FINISHED',pkg,time.time()-t,flush=True)
