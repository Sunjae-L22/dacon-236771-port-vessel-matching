from pathlib import Path
import csv,json
HERE=Path(__file__).resolve().parent;D=HERE/'eval'
def read(name):
 with (D/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def index(name):
 rs=read(name);assert len({r['sample_id'] for r in rs})==len(rs);return {r['sample_id']:r for r in rs}
e,h,v,c,t,q=index('en_pred.csv'),index('hs_pred.csv'),index('v20_pred.csv'),index('v20_checks_pred.csv'),index('truth.csv'),index('test.csv');ss=[json.loads(s) for s in (D/'semantic112.jsonl').read_text().splitlines()];changed=[r for r in ss if r['changed']];assert set(v)=={r['sample_id'] for r in changed};assert set(c)==set(v)
assert all(v[s]['target_vessel_id']==c[s]['target_vessel_id'] for s in v),'Row order independence failed'
def vote(a,b,ours):return b if b in (a,ours) else a
rows=[]
for r in changed:
 s=r['sample_id'];assert float(e[s]['ko_gap'])<4 and any('가'<=z<='힣' for z in q[s]['vessel_name']);a=v[s]['target_vessel_id'];b=h[s]['target_vessel_id'];old=vote(a,b,e[s]['target_vessel_id']);new=r['semantic112'];secondary=vote(a,b,new)
 rows.append({'sample_id':s,'name':q[s]['vessel_name'],'truth':t[s]['target_vessel_id'],'v20':a,'HSW':b,'112':e[s]['target_vessel_id'],'C_vote_MPS_reference':old,'semantic_challenge':new,'replace112_in_vote':secondary,'variant':t[s]['variant']})
result={'rows_in_fixed_diagnostic':len(ss),'all_proposed_change_rows_evaluated':len(rows),'unchanged_rows_by_construction':len(ss)-len(rows),'whole_C_vote_accuracy_computed':False,'order_replay_rows':len(c),'order_replay_mismatch':0,'comparison_scope':'Original C_vote source/weights with runtime device MPS. Not proven CUDA identity. Table-seen synthetic diagnostic, not Public validation.','strategies':{}}
for k in ('semantic_challenge','replace112_in_vote'):
 result['strategies'][k]={'changed_vs_C_vote':sum(r[k]!=r['C_vote_MPS_reference'] for r in rows),'rescue':sum(r['C_vote_MPS_reference']!=r['truth'] and r[k]==r['truth'] for r in rows),'harm':sum(r['C_vote_MPS_reference']==r['truth'] and r[k]!=r['truth'] for r in rows)}
 z=result['strategies'][k];z['net']=z['rescue']-z['harm']
(D/'CVOTE_COMPARISON_ROWS.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));(D/'CVOTE_COMPARISON.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
