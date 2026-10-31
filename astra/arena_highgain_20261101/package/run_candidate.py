"""Join actual successful C_vote traces, score one fixed rescue, publish only COMPLETE."""
from pathlib import Path
import argparse,hashlib,json,sys,subprocess,csv,math
sys.dont_write_bytecode=True
import neural_rescue as nr
HERE=Path(__file__).resolve().parent

def read(path):
 with Path(path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def index(records):
 records=list(records);out={r['sample_id']:r for r in records}
 if len(out)!=len(records):raise ValueError('Duplicate sample_id')
 return out
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def packets_from_baseline(payload,test,catalog,baseline):
 payload,test,catalog,baseline=map(Path,[payload,test,catalog,baseline])
 meta=json.loads((baseline/'run_meta.json').read_text())
 if meta.get('all_voters_succeeded') is not True or meta.get('trace_enabled') is not True:raise ValueError('Unsuccessful/untraced baseline')
 for key,path in [('test_sha256',test),('catalog_sha256',catalog),('router_sha256',payload/'script.py')]:
  if meta.get(key)!=sha(path):raise ValueError(f'Stale baseline: {key}')
 from audit_baseline import audit
 audit(payload,test,baseline,catalog)
 rows=index(read(test));base=index(read(baseline/'pred.csv'));parts={t:index(read(baseline/f'{t}_pred.csv')) for t in ('en','ko','hs')}
 import pandas as pd
 parsed=index(pd.read_csv(test,encoding='utf-8-sig',dtype={k:str for k in ('sample_id','vessel_name','call_sign','vessel_type','flag_name')}).to_dict('records'))
 traces=index(json.loads(s) for s in (baseline/'internal/v20_candidates.jsonl').read_text().splitlines())
 partial=index(json.loads(s) for s in (baseline/'internal/v20_packets_partial.jsonl').read_text().splitlines())
 expected={sid for sid,row in rows.items() if nr.hangul(row)}
 if set(partial)!=expected:raise ValueError('Missing/extra v20 packet')
 result=[]
 for sid in rows:
  if sid not in expected:continue
  p=dict(partial[sid]);e=parts['en'][sid];k=parts['ko'][sid];h=parts['hs'][sid]
  def scalar(v):
   if hasattr(v,'item'):v=v.item()
   return None if isinstance(v,float) and not math.isfinite(v) else v
  expected_row={key:scalar(parsed[sid][key]) for key in nr.ROW_KEYS if key in parsed[sid]}
  if p['row']!=expected_row:raise ValueError('Captured model row differs from actual input')
  t=traces[sid]
  for pk,tk in [('ids','candidate_ids'),('base','base_prob'),('post','final_scores'),('beams','beams')]:
   if p[pk]!=t[tk]:raise ValueError('The two independent observation records disagree: '+pk)
  if p['v20']!=k['target_vessel_id']:raise ValueError('Captured v20 winner disagrees with actual output')
  p.update(en=e['target_vessel_id'],hs=h['target_vessel_id'],gap=float(e['ko_gap']),components_ok=True)
  nr.validate(p)
  if nr.run_one(p,{i:{} for i in set(p['ids'])|{p['en'],p['hs']}},None,None,enabled=False)['candidate']!=base[sid]['target_vessel_id']:raise ValueError('Disabled packet replay changed C_vote')
  result.append(p)
 return result,rows,base

def assemble(packets,rows,base,results_path,status_path,catalog,out):
 status=json.loads(Path(status_path).read_text());pp=Path(out)/'packets.jsonl'
 if status.get('state')!='COMPLETE':raise ValueError('Incomplete GPU experiment')
 for key,path in [('packets_sha256',pp),('results_sha256',results_path),('catalog_sha256',catalog)]:
  if status.get(key)!=sha(path):raise ValueError('Stale/mismatched GPU result hash: '+key)
 rr=index(json.loads(s) for s in Path(results_path).read_text().splitlines())
 if set(rr)!={p['sample_id'] for p in packets} or status.get('rows')!=len(packets):raise ValueError('GPU result row mismatch')
 for p in packets:nr.verify_result(p,rr[p['sample_id']])
 final=[];counts={'korean':0,'english':0}
 for sid,row in rows.items():
  value=rr[sid]['candidate'] if sid in rr else base[sid]['target_vessel_id']
  if value!=base[sid]['target_vessel_id']:counts['korean' if nr.hangul(row) else 'english']+=1
  final.append(dict(sample_id=sid,target_vessel_id=value))
 if counts['english']:raise ValueError('English changed')
 path=Path(out)/'pred.csv'
 with path.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=['sample_id','target_vessel_id']);w.writeheader();w.writerows(final)
 summary={'state':'COMPLETE','candidate_status':'HOLD','changed_rows':counts,'baseline_rows':len(base),'scored_korean_rows':len(packets),'all_changed_rows_have_verified_packet_and_original_router':True,'protected_english_changes':0,'public_gain_estimate':None,'GPU_status':status}
 (Path(out)/'SUMMARY.json').write_text(json.dumps(summary,indent=2));return summary

def main():
 ap=argparse.ArgumentParser()
 for name in ('payload','test','catalog','baseline','out'):ap.add_argument('--'+name,type=Path,required=True)
 a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 packets,rows,base=packets_from_baseline(a.payload,a.test,a.catalog,a.baseline)
 pp=a.out/'packets.jsonl';pp.write_text(''.join(json.dumps(p,ensure_ascii=False,allow_nan=False)+'\n' for p in packets))
 output=a.out/'rescue.jsonl'
 subprocess.run([sys.executable,str(HERE/'run_gpu.py'),'--model-root',str(a.payload/'model/ko/model'),'--packets',str(pp),'--catalog',str(a.catalog),'--output',str(output),'--budget-seconds','600'],check=True)
 print(json.dumps(assemble(packets,rows,base,output,output.with_suffix('.status.json'),a.catalog,a.out),indent=2))
if __name__=='__main__':main()
