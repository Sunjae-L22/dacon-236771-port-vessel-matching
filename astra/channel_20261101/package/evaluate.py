from pathlib import Path
import json,collections,random,math
from channel_model import read

def evaluate(data,baseline,experiment,out):
 data,baseline,experiment,out=map(Path,(data,baseline,experiment,out));out.mkdir(exist_ok=False)
 truth={r['sample_id']:r for r in read(data/'truth.csv')};rows=[dict(r,**truth[r['sample_id']]) for r in map(json.loads,(experiment/'trace.jsonl').read_text().splitlines())];report={'evidence':'TRAIN_TOKEN_COMPOSITION_SYNTHETIC; actual unchanged C_vote baseline; NOT Public prediction','cohorts':{}}
 for condition in sorted({r['variant'] for r in rows})+['korean_all','all']:
  xs=[r for r in rows if condition=='all' or condition=='korean_all' and r['variant']!='english_control' or r['variant']==condition];resc=[r for r in xs if r['baseline']!=r['target_vessel_id'] and r['candidate']==r['target_vessel_id']];harm=[r for r in xs if r['baseline']==r['target_vessel_id'] and r['candidate']!=r['target_vessel_id']]
  report['cohorts'][condition]={'rows':len(xs),'base_correct':sum(r['baseline']==r['target_vessel_id'] for r in xs),'new_correct':sum(r['candidate']==r['target_vessel_id'] for r in xs),'rescue':len(resc),'harm':len(harm),'net':len(resc)-len(harm),'changed':sum(r['baseline']!=r['candidate'] for r in xs),'rescue_vessels':len({r['target_vessel_id'] for r in resc}),'harm_vessels':len({r['target_vessel_id'] for r in harm}),'rescue_112_hsw_same_wrong':sum(r['p112']==r['hsw'] for r in resc),'rescue_all_three_same_wrong':sum(r['p112']==r['hsw']==r['v20'] for r in resc),'harm_112_correct':sum(r['p112']==r['target_vessel_id'] for r in harm),'harm_hsw_correct':sum(r['hsw']==r['target_vessel_id'] for r in harm)}
 report['clean_harm']=sum(r['variant'].startswith('clean') and r['baseline']==r['target_vessel_id'] and r['candidate']!=r['target_vessel_id'] for r in rows)
 # Resample ships, not six correlated rows as independent observations.
 byship=collections.defaultdict(list)
 for r in rows:
  if r['variant']!='english_control':byship[r['target_vessel_id']].append(r)
 shipvals=[sum((r['candidate']==r['target_vessel_id'])-(r['baseline']==r['target_vessel_id']) for r in rs) for rs in byship.values()];rng=random.Random(112);boot=sorted(sum(rng.choice(shipvals) for _ in shipvals) for _ in range(4000));report['ship_bootstrap_net_rows_95pct']=[boot[100],boot[3899]];report['bootstrap_scope']='Uncertainty only within this synthetic cohort; not Public extrapolation.'
 report['offline_screen_pass']=report['clean_harm']==0 and report['cohorts']['korean_all']['net']>0;report['submission_eligible']=False
 (out/'SUMMARY.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(out/'paired_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));return report
