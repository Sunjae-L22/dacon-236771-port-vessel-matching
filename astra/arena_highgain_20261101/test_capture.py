"""Exercise real capture wrapper + unchanged router, and observation-only score hooks."""
from pathlib import Path
import sys,types,json,subprocess,os,shutil,importlib.util
import pandas as pd,numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;D=HERE/'capture_qa';D.mkdir(exist_ok=True)
sys.path.insert(0,str(HERE/'package'))
import trace_hooks

class Views:
    def __init__(self):
        self.top_k=20;self.weight=.25;self.r=types.SimpleNamespace(ids=np.array(['A','B','C']))
        self.baseline=types.SimpleNamespace(score_groups=lambda blocks:[np.array(b,dtype=float) for b in blocks]);self.views=[]
    def score_records(self,rows,signals,blocks):return self.baseline.score_groups(blocks)
fake=types.ModuleType('src.encoder_views');fake.EncoderViewsEnsemble=Views
sys.modules['src']=types.ModuleType('src');sys.modules['src.encoder_views']=fake
v=Views();rows=[{'sample_id':'x','vessel_name':'테스트'},{'sample_id':'y','vessel_name':'배'}];signals=[{'cand':np.array([0,1,2]),'llm_beams':['TEST']}]*2;blocks=[[.4,.35,.25],[.8,.15,.05]]
expected=v.score_records(rows,signals,blocks)
trace_hooks.attach_neural_trace(D/'trace.jsonl')
got=v.score_records(rows,signals,blocks)
assert all(np.array_equal(a,b) for a,b in zip(expected,got))
reverse=v.score_records(rows[::-1],signals[::-1],blocks[::-1]);assert all(np.array_equal(a,b) for a,b in zip(got,reverse[::-1]))
for i in range(2):assert np.array_equal(v.score_records(rows[i:i+1],signals[i:i+1],blocks[i:i+1])[0],got[i])
trace=[json.loads(x) for x in (D/'trace.jsonl').read_text().splitlines()];assert trace[0]['top20_ids']==['A','B','C'] and trace[0]['base_prob']==blocks[0]
assert trace_hooks.digest_row(dict(rows[0],target_vessel_id='A'))==trace_hooks.digest_row(dict(rows[0],target_vessel_id='B'))
pkg=D/'payload';pkg.mkdir(exist_ok=True);shutil.copy(HERE.parent/'arena_20261101/baseline/script.py',pkg/'script.py')
stub='''from pathlib import Path
import pandas as pd,os
def run(test_path,catalog_path,output_path):
 tag=Path(__file__).parent.name
 if os.getenv('FAIL_KO')=='1' and tag=='ko':raise SystemExit(7)
 x=pd.read_csv(test_path,dtype=str,keep_default_na=False)
 o=pd.DataFrame({'sample_id':x.sample_id,'target_vessel_id':x[tag]})
 if tag=='en':o['ko_gap']=x.gap
 o.iloc[::-1].to_csv(output_path,index=False)
'''
en_stub='''from pathlib import Path
import pandas as pd,types,numpy as np
def run(test_path,catalog_path,output_path):
 x=pd.read_csv(test_path,dtype=str,keep_default_na=False);preds=[]
 ix=types.SimpleNamespace(ids=['A','B','C'])
 for row in x.to_dict('records'):
  cs=None
  if cs is not None:
   preds.append(ix.ids[cs])
   continue
  idx=[0,1,2];score=[0.,0.,0.];best=ix.ids.index(row['en']);score[best]=1.
  allowed=[best]+[i for i in idx if i!=best];row_gap=float(row['gap']);use_kc=row_gap<4
  preds.append(ix.ids[best])
 o=pd.DataFrame({'sample_id':x.sample_id,'target_vessel_id':preds,'ko_gap':x.gap})
 o.iloc[::-1].to_csv(output_path,index=False)
'''
for tag in ['en','ko','hs']:
 p=pkg/'model'/tag;p.mkdir(parents=True,exist_ok=True);(p/'script.py').write_text(en_stub if tag=='en' else stub)
 if tag=='hs':
  (p/'pipeline.py').write_text('TON_GATE=None\ndef score_candidates(*a,**kw):return [],[]\ndef allowed(*a,**kw):return []\n')
 if tag=='ko':
  (p/'model').mkdir(exist_ok=True);(p/'model/train_vessels.json').write_text('[]')
  s=p/'model/src';s.mkdir(parents=True,exist_ok=True);(s/'__init__.py').write_text('');(s/'encoder_views.py').write_text('class EncoderViewsEnsemble:\n def score_records(self,rows,signals,blocks):return blocks\n')
input_rows=[['0','ENGLISH','0','C','A','A'],['1','한글','4','C','A','A'],['2','한글','1','C','A','A'],['3','한글','1','C','A','B'],['4','한글','1','C','A','C']]
df=pd.DataFrame(input_rows,columns=['sample_id','vessel_name','gap','en','ko','hs']);df.to_csv(D/'test.csv',index=False)
pd.DataFrame({'vessel_id':['A','B','C']}).to_csv(D/'catalog.csv',index=False)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0')
for tag,flags in [('plain',['--plain']),('observed',[])]:
 out=D/tag
 if out.exists():shutil.rmtree(out)
 p=subprocess.run([sys.executable,str(HERE/'package/capture_baseline.py'),str(pkg),str(D/'test.csv'),str(D/'catalog.csv'),str(out),*flags],env=env,capture_output=True,text=True)
 (D/f'{tag}.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
a=pd.read_csv(D/'plain/pred.csv',dtype=str);b=pd.read_csv(D/'observed/pred.csv',dtype=str);assert a.equals(b) and b.target_vessel_id.tolist()==['C','C','A','A','C']
env['FAIL_KO']='1';out=D/'failure'
if out.exists():shutil.rmtree(out)
p=subprocess.run([sys.executable,str(HERE/'package/capture_baseline.py'),str(pkg),str(D/'test.csv'),str(D/'catalog.csv'),str(out)],env=env,capture_output=True,text=True)
(D/'failure.log').write_text(p.stdout+p.stderr);assert p.returncode!=0 and not (out/'pred.csv').exists()
result=dict(scope='CPU mocks; no GPU model execution or accuracy claim',trace_hook_scores_unchanged=True,reverse_singleton_equal=True,labels_do_not_change_fingerprint=True,actual_router_plain_vs_observed_mismatch=0,router_cases=5,failed_voter_aborts_measurement=True)
(HERE/'CAPTURE_SMOKE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
