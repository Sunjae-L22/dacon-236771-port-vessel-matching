from pathlib import Path
import json,collections,hashlib
HERE=Path(__file__).resolve().parent;out={}
for fold in (2,4):
 d=HERE/f'dense_ranker/fold{fold}';res=json.loads((d/'RESULTS.json').read_text());assert res['old_proxy_reproduction_mismatch']==0;assert res['matched62_vs_old_proxy_prediction_difference']==0
 rows=[json.loads(s) for s in (d/'rows.jsonl').read_text().splitlines()];by=collections.defaultdict(list)
 for r in rows:by[r['target']].append(int(r['p64']==r['target'])-int(r['p62']==r['target']))
 group={k:sum(v) for k,v in by.items()};res['vessel_cluster_summary']={'total':len(group),'net_positive':sum(v>0 for v in group.values()),'net_negative':sum(v<0 for v in group.values()),'net_zero':sum(v==0 for v in group.values())}
 res['screen_pass']=res['metrics']['p62']['clean']['harm']==0 and res['metrics']['p62']['noise']['net']>0
 out[str(fold)]=res
report={'decision':'REJECT matched64-feature candidate: fold2 noisy net0 fails predeclared positive-gain screen','folds':out,'caveats':['single seed and feature_fraction0.8; feature-sampling changes alongside added columns','reused developer folds, not untouched test set','surrogate62 baseline, not production112/C_vote','multiple synthetic rows per vessel are correlated']};(HERE/'dense_ranker/DECISION.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v['vessel_cluster_summary'] for k,v in out.items()},indent=2))
