"""Post-freeze CPU execution and explicit conditional score scenarios, never Public forecasts."""
from pathlib import Path
import json,time,sys,math,collections,random
sys.dont_write_bytecode=True
from channel_model import Channel,read,decide,hangul
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=.9618922587

def known(r):return r['variant']=='english_control' or float(r['gap110'])>=4 or r['p112']==r['hsw']
def active(r):return r['reason']=='nondeletion_channel_dominant'
def summarize(rr):
 result={}
 for condition in sorted({r['variant'] for r in rr})+['all']:
  xs=[r for r in rr if condition=='all' or r['variant']==condition];kk=[r for r in xs if known(r)];uu=[r for r in xs if not known(r) and active(r)];kg=sum(int(r['candidate']==r['target_vessel_id'])-int(r['p112']==r['target_vessel_id']) for r in kk);ug=sum(r['candidate']==r['target_vessel_id'] for r in uu);ub=len(uu)-ug
  result[condition]={'rows':len(xs),'p112_rescue':sum(r['rescue'] for r in xs),'p112_harm':sum(r['harm'] for r in xs),'known_cvote_rows':len(kk),'known_cvote_rescue':sum(r['rescue'] for r in kk),'known_cvote_harm':sum(r['harm'] for r in kk),'unknown_active_new_correct':ug,'unknown_active_new_wrong':ub,'cvote_synthetic_net_lower':kg-ub,'cvote_synthetic_net_upper':kg+ug,'changed_vs112':sum(r['candidate']!=r['p112'] for r in xs)}
 return result

def main():
 out=HERE/'cpu_direct';out.mkdir(exist_ok=False)
 rr=[]
 for split in ('validation','replication'):
  rows=[json.loads(s) for s in (HERE/f'local_{split}/rows_v3.jsonl').read_text().splitlines()]
  rr.extend(dict(r,sample_id=split+':'+r['sample_id'],split=split) for r in rows)
 s=summarize(rr);params=json.loads((HERE/'translit/NORMALIZED_FIT.json').read_text());op=params['operation_probabilities'];steps=params['step_probabilities']
 # Explicit illustrative assumptions; not a prediction interval. Other untested channels set delta0.
 weights={'delete':steps['1']*op['delete'],'dup':steps['1']*op['dup'],'swap':steps['1']*op['swap'],'double':steps['2']*op['delete']*op['swap']}
 weighted={bound:sum(weights[c]*s[c]['cvote_synthetic_net_'+bound]/s[c]['rows'] for c in weights) for bound in ('lower','upper')}
 projected={k:BASE+.7*v for k,v in weighted.items()}
 scenario={'label':'ASSUMPTION-BASED SENSITIVITY ONLY; NOT calibrated expected score or Public confidence interval','assumptions':['English fitted event proportions transfer unchanged to Korean syllables.','Synthetic composed-name cohort per-condition gains transfer to hidden Public.','Untested event sequences and unknown corruption types contribute exactly zero gain/harm.','Only tested delete-then-swap is transferred; reverse order not counted.','English stays3506/3568 and base Public is0.9618922587.'],'event_weights':weights,'public_score_sensitivity_range':projected,'implied_korean_net_rows':{k:532*v for k,v in weighted.items()},'not_identifiable':'Actual v20 predictions, Korean/Public corruption mix and unseen-name representativeness are missing; no point forecast justified.'}
 report={'scope':'V3 frozen before two prospective composition cohorts; all CPU runs executed locally','rows':len(rr),'ships':len({r['target_vessel_id'] for r in rr}),'metrics':s,'score_scenario':scenario,'full_cvote_gpu_executed':False,'production_local_check_3000_600':False}
 (out/'RESULTS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
 # Benchmark added module only, using per-row frozen112 as reference where v20 is unknown.
 t=time.monotonic();exclude={r['target_vessel_id'] for r in read(ROOT/'data/open/train.csv')};m=Channel.load(ROOT/'data/open/vessel_catalog.csv',ROOT/'shared/kocat/out/catalog_ko.csv',HERE/'translit/NORMALIZED_FIT.json',exclude);init=time.monotonic()-t
 seq=(rr*math.ceil(3000/len(rr)))[:3000];expected={r['sample_id']:r['candidate'] for r in rr};mismatch=0;t=time.monotonic()
 for r in seq:
  tr=m.query(r['vessel_name']) if hangul(r['vessel_name']) and float(r['gap110'])<4 else {};p,_=decide(r,r['p112'],r['gap110'],tr);mismatch+=p!=expected[r['sample_id']]
 elapsed=time.monotonic()-t;checks=list(seq[:600]);random.Random(777).shuffle(checks);t=time.monotonic();mi=0
 for r in checks:
  tr=m.query(r['vessel_name']) if hangul(r['vessel_name']) and float(r['gap110'])<4 else {};p,_=decide(r,r['p112'],r['gap110'],tr);mi+=p!=expected[r['sample_id']]
 qa={'platform':'local MacM4 CPU; NOT L4','scope':'added CPU module only; cached112 reference; not full C_vote local_check','initialization_seconds':init,'rows_3000':len(seq),'seconds_3000':elapsed,'mismatch_3000':mismatch,'shuffled_duplicate_rows_600':600,'seconds_600':time.monotonic()-t,'mismatch_600':mi,'query_cache_used':False}
 if mismatch or mi:raise ValueError('Frozen inference mismatch')
 (out/'CPU_MODULE_QA.json').write_text(json.dumps(qa,indent=2));print(json.dumps(qa,indent=2),flush=True)
if __name__=='__main__':main()
