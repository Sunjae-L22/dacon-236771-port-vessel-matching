from pathlib import Path
import csv,json,collections,math,numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent

def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def pos(x):
 try:v=float(x);return v if v>0 and math.isfinite(v) else float('nan')
 except:return float('nan')
cat=read(ROOT/'data/open/vessel_catalog.csv');byid={r['vessel_id']:r for r in cat};train=read(ROOT/'data/open/train.csv');r=[];hist=collections.Counter();pairs=collections.Counter();catmatch=collections.Counter();bytype=collections.defaultdict(list)
for x in train:
 c=byid[x['target_vessel_id']];g=pos(c['gross_tonnage']);a=pos(x['gross_tonnage']);b=pos(x['international_gross_tonnage']);h=any('가'<=z<='힣' for z in x['vessel_name']);cl='ko' if h else 'en'
 if math.isfinite(a) and math.isfinite(b):hist[cl+':same_two_gt']+=a==b
 hist[cl+':rows']+=1
 if math.isfinite(g):
  vals=[v/g for v in (a,b) if math.isfinite(v)]
  if vals:
   best=min(vals,key=lambda z:abs(z-1));hist[cl+':has_tonnage']+=1;hist[cl+':exact_any']+=best==1;hist[cl+':within_10pct']+=abs(best-1)<.1;pairs[tuple(round(v,3) for v in vals)]+=1;r.append({'sample_id':x['sample_id'],'vessel_id':x['target_vessel_id'],'is_ko':h,'r1':a/g,'r2':b/g,'best':best});bytype[cl].append(best)
 for f in ('gross_tonnage','deadweight_tonnage','size'):
  val=pos(c[f]);catmatch[f]+=math.isfinite(val) and (val==a or val==b)
report={'hist':dict(hist),'catalog_exact_matches':dict(catmatch),'common_joint_ratios':pairs.most_common(30),'best_ratio_quantiles':{k:np.quantile(v,[0,.01,.1,.25,.5,.75,.9,.99,1]).tolist() for k,v in bytype.items()}}
(OUT/'AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
