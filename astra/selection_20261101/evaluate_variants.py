from pathlib import Path
import os,json,csv
ROOT=Path(__file__).resolve().parents[2];HERE=Path(os.environ.get('ASTRA_WORK_DIR',str(Path(__file__).resolve().parent)));D=Path(os.environ.get('ASTRA_EVAL_DIR',str(ROOT/'astra/new_direction_20261101/local_cvote/eval')))
def read(p):
 with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def idx(p):
 rows=read(p);assert len(rows)==len({r['sample_id'] for r in rows});return {r['sample_id']:r for r in rows}
def route(a,b,e):return b if b in (a,e) else a
rows=read(D/'test.csv');t=idx(D/'truth.csv');e=idx(D/'en_pred.csv');h=idx(D/'hs_pred.csv');v=idx(HERE/'v20_hard_pred.csv');dec={r['sample_id']:r for r in map(json.loads,(HERE/'decomposition.jsonl').read_text().splitlines())};alts={k:idx(HERE/(k+'_v20_pred.csv')) for k in ('mask','context') if (HERE/(k+'_v20_pred.csv')).exists()};out=[]
for row in rows:
 s=row['sample_id'];ko=any('가'<=c<='힣' for c in row['vessel_name']);hard=ko and float(e[s]['ko_gap'])<4;base=e[s]['target_vessel_id'];r={'sample_id':s,'truth':t[s]['target_vessel_id'],'variant':t[s]['variant'],'korean':ko,'hard':hard}
 if hard:
  base=route(v[s]['target_vessel_id'],h[s]['target_vessel_id'],base);views=json.loads((HERE/'capture'/(s+'.json')).read_text())['views'];tops=[z['ids'][max(range(len(z['scores'])),key=lambda i:z['scores'][i])] for z in views];a=v[s]['target_vessel_id'];a_veto=dec[s]['base_post_KO'] if len(tops)==2 and len(set(tops))==2 and a!=dec[s]['base_post_KO'] else a
  r['neural_conflict_veto']=route(a_veto,h[s]['target_vessel_id'],e[s]['target_vessel_id']);r['neural_views_disagree']=len(tops)==2 and len(set(tops))==2
  for k,data in alts.items():r[k]=route(data[s]['target_vessel_id'],h[s]['target_vessel_id'],e[s]['target_vessel_id'])
 else:
  r['neural_conflict_veto']=base
  for k in alts:r[k]=base
 r['C_vote']=base;out.append(r)
report={'rows':len(out),'scope':'Original C_vote MPS reference on synthetic diagnostic, no Public estimate','original_C_vote_correct':sum(r['C_vote']==r['truth'] for r in out),'strategies':{}}
for k in list(alts)+['neural_conflict_veto']:
 report['strategies'][k]={}
 for cond in ('all','Kclean','Knoise','English','protected_K'):
  xs=[r for r in out if cond=='all' or cond=='Kclean' and r['korean'] and r['variant'].startswith('clean') or cond=='Knoise' and r['korean'] and not r['variant'].startswith('clean') or cond=='English' and not r['korean'] or cond=='protected_K' and r['korean'] and not r['hard']]
  good=[r for r in xs if r['C_vote']!=r['truth'] and r[k]==r['truth']];bad=[r for r in xs if r['C_vote']==r['truth'] and r[k]!=r['truth']];report['strategies'][k][cond]={'rows':len(xs),'changed':sum(r[k]!=r['C_vote'] for r in xs),'rescue':len(good),'harm':len(bad),'net':len(good)-len(bad),'rescue_ships':len({r['truth'] for r in good}),'harm_ships':len({r['truth'] for r in bad})}
 for cond in ('English','protected_K'):assert report['strategies'][k][cond]['changed']==0
(HERE/'COMPARISON_ROWS.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in out));(HERE/'COMPARISON.json').write_text(json.dumps(report,indent=2));
with (HERE/'cvote_reference_pred.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['sample_id','target_vessel_id']);w.writeheader();w.writerows({'sample_id':r['sample_id'],'target_vessel_id':r['C_vote']} for r in out)
print(json.dumps(report,indent=2))
