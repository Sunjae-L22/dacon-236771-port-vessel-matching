from pathlib import Path
import sys,csv,json
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'package'))
from evaluate import evaluate
D=R/'evaluator_qa';data=D/'data';base=D/'baseline';out=D/'out'
for p in (data,base,base/'internal'):p.mkdir(parents=True,exist_ok=True)
def write(p,rr):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
q=[dict(sample_id=str(i),vessel_name='한글') for i in range(6)];q[-1]['vessel_name']='ENGLISH'
truth=[dict(sample_id=str(i),target_vessel_id='T',variant='clean_nogt' if i==0 else 'delete_nogt',source='table') for i in range(6)]
parts={x:[] for x in ('en','ko','hs')};b=[];c=[];tr={x:[] for x in ('v20','en','hs')}
for i in range(6):
 sid=str(i);e='T' if i==0 else 'B';k='T' if i in (0,4) else 'B';h=e;gap=4 if i==0 else 1
 for tag,pred in [('en',e),('ko',k),('hs',h)]:
  rr=dict(sample_id=sid,target_vessel_id=pred)
  if tag=='en':rr['ko_gap']=gap
  if tag=='en' or i<5:parts[tag].append(rr)
 b.append(dict(sample_id=sid,target_vessel_id=e));c.append(dict(sample_id=sid,target_vessel_id='T' if i==3 else e))
 pool=['B'] if i in (1,2) else ['B','T'];top=['B'] if i==3 else pool
 if i<5:
  tr['v20'].append(dict(sample_id=sid,candidate_ids=pool,top20_ids=top))
  tr['en'].append(dict(sample_id=sid,candidate_ids=['B','T'] if i==2 else ['B'],gap110=gap))
  tr['hs'].append(dict(sample_id=sid,candidate_ids=['B']))
write(data/'pilot_test.csv',q);write(data/'pilot_truth.csv',truth);write(base/'pred.csv',b);write(D/'candidate.csv',c)
for tag,rr in parts.items():write(base/f'{tag}_pred.csv',rr)
for tag,rr in tr.items():(base/f'internal/{tag}_candidates.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rr))
r=evaluate(data,base,out,D/'candidate.csv');assert r['cohorts']['ALL']['rescue']==1 and r['cohorts']['ALL']['harm']==0
expected={'baseline_correct':1,'absent_all_traced_pools':1,'absent_v20_present_other':1,'v20_below20':1,'vote_overrode_correct_v20':1,'protected_error':1};assert r['baseline_error_categories']==expected,r
c[0]['target_vessel_id']='B';write(D/'candidate.csv',c)
try:evaluate(data,base,D/'bad',D/'candidate.csv');raise AssertionError('protected edit accepted')
except ValueError as e:assert 'Protected row' in str(e)
print('PASS error decomposition, paired rescue/harm, route reconstruction, protected row rejection')
