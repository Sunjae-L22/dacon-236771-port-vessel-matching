"""Frozen CPU diagnostic. Never writes a submission package or uses labels for decisions."""
from pathlib import Path
import argparse,json,time,pickle,math,csv,random,sys,collections
sys.dont_write_bytecode=True
from channel_model import Channel,read,decide,hangul
HERE=Path(__file__).resolve().parent

def indexed(path,expected):
 rows=read(path);ids=[r['sample_id'] for r in rows]
 if len(ids)!=len(set(ids)) or set(ids)!=set(expected) or any(not r['target_vessel_id'] for r in rows):raise ValueError(f'Invalid prediction IDs: {path}')
 return {r['sample_id']:r for r in rows}

def write(path,rows):
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(payload,test,catalog,baseline,out):
 start=time.monotonic();payload,test,catalog,baseline,out=map(Path,(payload,test,catalog,baseline,out));out.mkdir(exist_ok=False)
 rows=read(test);ids=[r['sample_id'] for r in rows];kids=[r['sample_id'] for r in rows if hangul(r['vessel_name'])]
 if len(ids)!=len(set(ids)) or any('target_vessel_id' in r for r in rows):raise ValueError('Invalid test or truth leakage')
 original=indexed(baseline/'pred.csv',ids);en=indexed(baseline/'en_pred.csv',ids);hs=indexed(baseline/'hs_pred.csv',kids);ko=indexed(baseline/'ko_pred.csv',kids)
 if [r['sample_id'] for r in read(baseline/'pred.csv')]!=ids:raise ValueError('Original order differs')
 cat=read(catalog);valid={r['vessel_id'] for r in cat}
 for d in (original,en,hs,ko):
  if any(r['target_vessel_id'] not in valid for r in d.values()):raise ValueError('Invalid vessel ID')
 stats=pickle.loads((payload/'model/en/model/stats.pkl').read_bytes());excluded=set(stats['cs_map'].values())
 model=Channel(cat,read(payload/'model/en/model/catalog_ko.csv'),json.loads((HERE/'params.json').read_text()),excluded)
 traces=[];pred=[];route_mismatch=0;t=time.monotonic()
 for r in rows:
  sid=r['sample_id'];old=original[sid]['target_vessel_id'];e=en[sid]['target_vessel_id'];gap=en[sid]['ko_gap'];h=hs.get(sid,{}).get('target_vessel_id');k=ko.get(sid,{}).get('target_vessel_id');isk=hangul(r['vessel_name'])
  if isk and not math.isfinite(float(gap)):raise ValueError('Missing original110 gap')
  route=e if not isk or float(gap)>=4 else h if h==e or h==k else k
  route_mismatch+=route!=old
  tr=model.query(r['vessel_name']) if isk and float(gap)<4 else {};new,reason=decide(r,old,gap,tr)
  if new not in valid:raise ValueError('Invalid candidate vessel')
  if new!=old and (not isk or float(gap)>=4):raise ValueError('Changed protected row')
  pred.append({'sample_id':sid,'target_vessel_id':new});traces.append(dict(sample_id=sid,baseline=old,candidate=new,p112=e,hsw=h,v20=k,gap110=gap,reason=reason,trace=tr))
 if route_mismatch:raise ValueError(f'Original C_vote router mismatch: {route_mismatch}')
 predict_seconds=time.monotonic()-t
 write(out/'pred.csv',pred);(out/'trace.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in traces))
 # Exercise actual scoring, order and repeated queries with frozen per-row original outputs.
 expected={r['sample_id']:r['target_vessel_id'] for r in pred};checks=(rows*math.ceil(600/len(rows)))[:600];random.Random(110112).shuffle(checks);mismatch=0;t=time.monotonic()
 for r in checks:
  sid=r['sample_id'];gap=en[sid]['ko_gap'];tr=model.query(r['vessel_name']) if hangul(r['vessel_name']) and float(gap)<4 else {};new,_=decide(r,original[sid]['target_vessel_id'],gap,tr);mismatch+=new!=expected[sid]
 if mismatch:raise ValueError('CPU overlay row invariance failed')
 report={'rows':len(rows),'changed_korean':sum(r['baseline']!=r['candidate'] and hangul(q['vessel_name']) for r,q in zip(traces,rows)),'changed_english':sum(r['baseline']!=r['candidate'] and not hangul(q['vessel_name']) for r,q in zip(traces,rows)),'router_reconstruction_mismatches':route_mismatch,'protected_row_changes':0,'cpu_overlay_repeated_shuffled_rows':len(checks),'cpu_overlay_mismatches':mismatch,'cpu_overlay_replay_seconds':time.monotonic()-t,'cpu_query_seconds':predict_seconds,'cpu_total_seconds':time.monotonic()-start,'train_vessels_excluded':len(excluded),'score_mass_share_is_calibrated_probability':False,'production_local_check_3000_600':False,'submission_eligible':False}
 (out/'CPU_QA.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main(*sys.argv[1:6])
