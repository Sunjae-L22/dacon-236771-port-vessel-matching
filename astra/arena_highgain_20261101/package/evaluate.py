"""Offline paired diagnosis. Truth enters here, never inference."""
from pathlib import Path
import csv,json,collections,unicodedata

def read(path):
 with Path(path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def index(rows):
 rows=list(rows)
 out={r['sample_id']:r for r in rows}
 if len(out)!=len(rows):raise ValueError('Duplicate sample_id')
 return out
def jl(path):return index(json.loads(x) for x in Path(path).read_text().splitlines()) if Path(path).exists() else {}
def hangul(s):return any('가'<=c<='힣' for c in unicodedata.normalize('NFC',s))
def vote(en,ko,hs,gap,is_ko):
 if not is_ko or gap>=4:return en
 if hs in (en,ko):return hs
 if ko==en:return ko
 return ko

def evaluate(data,baseline,out,experiment=None,split='pilot'):
 data,baseline,out=map(Path,[data,baseline,out]);out.mkdir(parents=True,exist_ok=True)
 queries=index(read(data/f'{split}_test.csv'));truth=index(read(data/f'{split}_truth.csv'));base=index(read(baseline/'pred.csv'))
 parts={tag:index(read(baseline/f'{tag}_pred.csv')) for tag in ('en','ko','hs')}
 trace={tag:jl(baseline/f'internal/{tag}_candidates.jsonl') for tag in ('en','hs','v20')}
 candidate=index(read(experiment)) if experiment else base
 if set(queries)!=set(base) or set(base)!=set(truth) or set(base)!=set(candidate):raise ValueError('Row IDs mismatch')
 rows=[];counts=collections.Counter();cohorts={}
 for sid,q in queries.items():
  y=truth[sid]['target_vessel_id'];b=base[sid]['target_vessel_id'];c=candidate[sid]['target_vessel_id'];ko=hangul(q['vessel_name']);e=parts['en'][sid];gap=float(e['ko_gap']) if ko else 99
  a=parts['ko'][sid]['target_vessel_id'] if ko else e['target_vessel_id'];h=parts['hs'][sid]['target_vessel_id'] if ko else e['target_vessel_id']
  if vote(e['target_vessel_id'],a,h,gap,ko)!=b:raise ValueError(f'Router reconstruction mismatch: {sid}')
  if c!=b and (not ko or gap>=4):raise ValueError(f'Protected row changed: {sid}')
  t=trace['v20'].get(sid,{});pool=t.get('candidate_ids',[]);top=t.get('top20_ids',[])
  et=trace['en'].get(sid)
  if ko and (et is None or abs(float(et['gap110'])-gap)>1e-9):raise ValueError(f'110 gap trace mismatch: {sid}')
  union=set(pool)
  for tag in ('en','hs'):union.update(trace[tag].get(sid,{}).get('candidate_ids',[]))
  if b==y:category='baseline_correct'
  elif not ko or gap>=4:category='protected_error'
  elif not t:category='trace_missing'
  elif y not in union:category='absent_all_traced_pools'
  elif y not in pool:category='absent_v20_present_other'
  elif y not in top:category='v20_below20'
  elif a!=y:category='v20_top20_misranked'
  else:category='vote_overrode_correct_v20'
  r=dict(sample_id=sid,vessel_name=q['vessel_name'],truth=y,baseline=b,experiment=c,en=e['target_vessel_id'],v20=a,hs=h,gap110=gap,is_ko=ko,changed=c!=b,rescue=b!=y and c==y,harm=b==y and c!=y,variant=truth[sid]['variant'],source=truth[sid]['source'],category=category,truth_in_v20_pool=y in pool,truth_in_v20_top20=y in top,truth_in_union=y in union)
  rows.append(r);counts[category]+=1
 for key in ['ALL','KOREAN','ENGLISH']+sorted({r['variant'] for r in rows})+sorted({r['source'] for r in rows}):
  rr=[r for r in rows if key=='ALL' or key=='KOREAN' and r['is_ko'] or key=='ENGLISH' and not r['is_ko'] or key in (r['variant'],r['source'])]
  cohorts[key]={k:sum(int(r[k]) for r in rr) for k in ('changed','rescue','harm')};cohorts[key].update(rows=len(rr),baseline_correct=sum(r['baseline']==r['truth'] for r in rr),experiment_correct=sum(r['experiment']==r['truth'] for r in rr))
 clean=[r for r in rows if r['variant'].startswith('clean')];wrong=[r for r in rows if r['is_ko'] and r['baseline']!=r['truth'] and r['gap110']<4]
 report=dict(evidence_class='SYNTHETIC_TABLE_SEEN_STYLE_BIASED',actual_C_vote_weights_and_router=True,independent_holdout=False,submission_eligible=False,public_gain_estimate=None,cohorts=cohorts,baseline_error_categories=dict(counts),clean_harm=sum(r['harm'] for r in clean),hard_wrong_truth_in_union=sum(r['truth_in_union'] for r in wrong),hard_wrong_rows=len(wrong),english_protected_changes=0,route_reconstruction_mismatch=0,note='Coverage is an oracle ceiling on these synthetic rows, not achievable gain or Public accuracy.')
 with (out/'paired_rows.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));return report
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--baseline',required=True);p.add_argument('--out',required=True);p.add_argument('--experiment');p.add_argument('--split',default='pilot',choices=['pilot','full']);a=p.parse_args();print(json.dumps(evaluate(**vars(a)),ensure_ascii=False,indent=2))
