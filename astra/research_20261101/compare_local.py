from pathlib import Path
import json,time,csv,sys,collections
sys.dont_write_bytecode=True
from channel_model import Channel,read,decide,hangul
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def main():
 cohort=sys.argv[1] if len(sys.argv)>1 else 'cohort';dest=sys.argv[2] if len(sys.argv)>2 else 'local'
 t=time.monotonic();rows=read(HERE/f'{cohort}/test.csv');truth={r['sample_id']:r for r in read(HERE/f'{cohort}/truth.csv')};en={r['sample_id']:r for r in read(HERE/f'{dest}/en_pred.csv')};hs={r['sample_id']:r for r in read(HERE/f'{dest}/hs_pred.csv')};exclude={r['target_vessel_id'] for r in read(ROOT/'data/open/train.csv')}
 model=Channel.load(ROOT/'data/open/vessel_catalog.csv',ROOT/'shared/kocat/out/catalog_ko.csv',HERE/'translit/NORMALIZED_FIT.json',exclude);rr=[]
 for r in rows:
  sid=r['sample_id'];old=en[sid]['target_vessel_id'];gt=truth[sid]['target_vessel_id'];gap=en[sid]['ko_gap'];tr=model.query(r['vessel_name']) if hangul(r['vessel_name']) else {};new,reason=decide(r,old,gap,tr)
  rr.append(dict(**r,**{k:v for k,v in truth[sid].items() if k!='sample_id'},p112=old,hsw=hs[sid]['target_vessel_id'],candidate=new,gap110=gap,reason=reason,trace=tr,rescue=old!=gt and new==gt,harm=old==gt and new!=gt))
 report={'split':cohort,'scope':'Actual original 112 and HSW CPU; NOT full C_vote; V3 frozen .99 plus deletion-only veto','seconds_including_channel_init':time.monotonic()-t,'excluded_train_vessels':len(exclude),'cohorts':{}}
 for v in sorted({r['variant'] for r in rr})+['all']:
  xs=[r for r in rr if v=='all' or r['variant']==v];changes=[r for r in xs if r['p112']!=r['candidate']];report['cohorts'][v]={'rows':len(xs),'base_correct':sum(r['p112']==r['target_vessel_id'] for r in xs),'new_correct':sum(r['candidate']==r['target_vessel_id'] for r in xs),'changed':len(changes),'rescue':sum(r['rescue'] for r in xs),'harm':sum(r['harm'] for r in xs),'rescue_hsw_also_wrong':sum(r['rescue'] and r['hsw']!=r['target_vessel_id'] for r in xs),'rescue_112_equals_hsw':sum(r['rescue'] and r['hsw']==r['p112'] for r in xs)}
 (HERE/f'{dest}/COMPARE_V3.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(HERE/f'{dest}/rows_v3.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr));print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
