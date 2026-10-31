from pathlib import Path
import sys,json,csv,hashlib,tempfile,copy
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'package'))
import neural_rescue as nr
from run_candidate import packets_from_baseline,assemble
D=R/'bridge_qa';D.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir=D) as tmp:
 d=Path(tmp);payload=d/'payload';b=d/'baseline';out=d/'out'
 for f in [payload/'model/ko/model',b/'internal',out]:f.mkdir(parents=True)
 (payload/'script.py').write_text('# fixed router fixture');(payload/'model/ko/model/train_vessels.json').write_text('[]')
 def csvwrite(p,rr):
  with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 ids=[f'C{i:03}' for i in range(100)];q=[dict(sample_id='S',vessel_name='스타 오션'),dict(sample_id='E',vessel_name='ENGLISH')];csvwrite(d/'test.csv',q);csvwrite(d/'catalog.csv',[dict(vessel_id=i,vessel_name=i) for i in ids])
 p=dict(sample_id='S',row=q[0],ids=ids,base=[.9]+[.1/99]*99,post=[1.]+[0.]*99,signals={k:[1. if i=='C064' else 0. for i in ids] for k in nr.SIGNALS},beams=['STAR OCEAN'],excluded=[],v20='C000')
 (b/'internal/v20_packets_partial.jsonl').write_text(json.dumps(p)+'\n')
 t=dict(sample_id='S',candidate_ids=ids,base_prob=p['base'],final_scores=p['post'],top20_ids=ids[:20],beams=p['beams'],views=[])
 (b/'internal/v20_candidates.jsonl').write_text(json.dumps(t)+'\n')
 for tag,win in [('en','C001'),('hs','C002')]:
  tr=dict(sample_id='S',candidate_ids=[win,'C000'],scores=[2.,1.],top20_ids=[win,'C000'])
  if tag=='en':tr['gap110']=.3
  (b/f'internal/{tag}_candidates.jsonl').write_text(json.dumps(tr)+'\n')
 csvwrite(b/'en_pred.csv',[dict(sample_id='S',target_vessel_id='C001',ko_gap=.3),dict(sample_id='E',target_vessel_id='C003',ko_gap='')]);csvwrite(b/'hs_pred.csv',[dict(sample_id='S',target_vessel_id='C002')]);csvwrite(b/'ko_pred.csv',[dict(sample_id='S',target_vessel_id='C000')]);csvwrite(b/'pred.csv',[dict(sample_id='S',target_vessel_id='C000'),dict(sample_id='E',target_vessel_id='C003')])
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 meta=dict(all_voters_succeeded=True,trace_enabled=True,test_sha256=sha(d/'test.csv'),catalog_sha256=sha(d/'catalog.csv'),router_sha256=sha(payload/'script.py'));(b/'run_meta.json').write_text(json.dumps(meta))
 packets,rows,base=packets_from_baseline(payload,d/'test.csv',d/'catalog.csv',b);assert len(packets)==1
 class Scorer:
  def score(self,row,candidates,beams):return [9. if c['vessel_name']=='C064' else 0. for c in candidates]
 cat={i:dict(vessel_name=i) for i in ids};results=[nr.run_one(packets[0],cat,Scorer(),Scorer())];assert results[0]['candidate']=='C064'
 (out/'packets.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in packets));(out/'rescue.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in results))
 status=dict(state='COMPLETE',rows=1,packets_sha256=sha(out/'packets.jsonl'),results_sha256=sha(out/'rescue.jsonl'),catalog_sha256=sha(d/'catalog.csv'));(out/'rescue.status.json').write_text(json.dumps(status))
 summary=assemble(packets,rows,base,out/'rescue.jsonl',out/'rescue.status.json',d/'catalog.csv',out);assert summary['changed_rows']=={'korean':1,'english':0}
 status['state']='ABORT';(out/'rescue.status.json').write_text(json.dumps(status))
 try:assemble(packets,rows,base,out/'rescue.jsonl',out/'rescue.status.json',d/'catalog.csv',out);raise AssertionError('ABORT accepted')
 except ValueError:pass
 bad=copy.deepcopy(p);bad['row']['vessel_name']='다른 이름';(b/'internal/v20_packets_partial.jsonl').write_text(json.dumps(bad)+'\n')
 try:packets_from_baseline(payload,d/'test.csv',d/'catalog.csv',b);raise AssertionError('Stale model row accepted')
 except ValueError as e:assert 'model row' in str(e)
print('PASS captured-packet join, actual source parser alignment, rank65 rescue through frozen router, English unchanged, ABORT/stale packet rejected (CPU doubles)')
