from pathlib import Path
import sys,os,importlib.util,inspect,time,json
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
cohort=sys.argv[2] if len(sys.argv)>2 else 'cohort';dest=sys.argv[3] if len(sys.argv)>3 else 'local'
pkg=sys.argv[1];payload=ROOT/'astra/arena_20261101/baseline/model'/pkg
sys.path.insert(0,str(payload));s=importlib.util.spec_from_file_location('entry',payload/'script.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
k='output_path' if 'output_path' in inspect.signature(m.run).parameters else 'out'
kwargs={k:str(HERE/f'{dest}/{pkg}_pred.csv')}
if pkg=='hs':kwargs['model_path']=str(payload/'model/model.pkl')
t=time.monotonic();m.run(test_path=str(HERE/f'{cohort}/test.csv'),catalog_path=str(ROOT/'data/open/vessel_catalog.csv'),**kwargs)
(HERE/f'{dest}/{pkg}_runtime.json').write_text(json.dumps({'seconds':time.monotonic()-t,'rows':sum(1 for _ in open(HERE/f'{cohort}/test.csv'))-1,'platform':'MacM4 CPU, not L4'},indent=2))
