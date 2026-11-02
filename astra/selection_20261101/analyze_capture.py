import os
from pathlib import Path
import csv,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(os.environ.get("ASTRA_WORK_DIR",str(Path(__file__).resolve().parent)));D=Path(os.environ.get('ASTRA_EVAL_DIR',str(ROOT/'astra/new_direction_20261101/local_cvote/eval')))
def read(p):
 with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def route(a,b,ours):return b if b in (a,ours) else a

def prepare():
 sys.path.insert(0,str(ROOT/'astra/arena_highgain_20261101/package'));from trace_hooks import load_entry
 m=load_entry(ROOT/'astra/new_direction_20261101/local_cvote/payload_mps/model/ko/script.py')
 import pandas as pd
 # Need only table name order and the exact original name-similarity function.
 cat=pd.read_csv(ROOT/'data/open/vessel_catalog.csv');table=json.loads((ROOT/'astra/arena_20261101/baseline/model/ko/model/catalog_ko.json').read_text());ko_names=[m._ko_norm(table.get(str(n),'')) for n in cat.vessel_name]
 excluded=set(json.loads((ROOT/'astra/arena_20261101/baseline/model/ko/model/train_vessels.json').read_text()));truth={r['sample_id']:r for r in read(D/'truth.csv')};en={r['sample_id']:r for r in read(D/'en_pred.csv')};hs={r['sample_id']:r for r in read(D/'hs_pred.csv')};vp={r['sample_id']:r for r in read(HERE/'v20_hard_pred.csv')};rr=[];maxdiff=0.
 out=HERE/'mask';out.mkdir(exist_ok=True)
 for row in read(HERE/'hard_test.csv'):
  sid=row['sample_id'];z=np.load(HERE/'capture'/(sid+'.npz'));meta=json.loads((HERE/'capture'/(sid+'.json')).read_text());ids=z['candidate_ids'];idx=z['candidate_index'];raw=z['raw'];w=np.array([a['weight'] for a in meta['members']]);bad=np.isin(ids,list(excluded));valid=~bad if bad.any() and not bad.all() else np.ones(len(ids),bool)
  rawexp=np.exp(raw-raw.max(axis=1,keepdims=True));prob=rawexp/rawexp.sum(axis=1,keepdims=True);mass=prob[:,valid].sum(axis=1)
  old=(w[:,None]*prob).sum(axis=0);old[~valid]=0.;old/=old.sum();new=(w[:,None]*np.where(valid,prob/mass[:,None],0.)).sum(axis=0);new/=new.sum()
  ks=m.ko_similarity(row['vessel_name'],idx,ko_names)
  def prior(b):
   zz=np.full(len(b),-np.inf);zz[valid]=np.log(b[valid])+m.KO_W*ks[valid];zz-=zz[valid].max();p=np.exp(zz);return p/p.sum()
  oldko=prior(old);newko=prior(new);diff=float(np.max(np.abs(oldko-z['base'])));maxdiff=max(maxdiff,diff);assert diff<1e-10,(sid,diff)
  def pick(p):return str(ids[int(np.argmax(np.where(valid,p,-np.inf)))])
  assert pick(z['final'])==vp[sid]['target_vessel_id'];target=truth[sid]['target_vessel_id'];ti=np.flatnonzero(ids==target)
  oldorder=np.argsort(-oldko,kind='stable')[:20];neworder=np.argsort(-newko,kind='stable')[:20]
  def active(p,order):
   bp=p[order]/p[order].sum();return len(order)>=2 and float(bp[0]-bp[1])<=1/3+1e-6
  a=vp[sid]['target_vessel_id'];b=hs[sid]['target_vessel_id'];ours=en[sid]['target_vessel_id'];cv=route(a,b,ours)
  r={'sample_id':sid,'variant':truth[sid]['variant'],'truth':target,'v20':a,'HSW':b,'112':ours,'C_vote':cv,'candidate_contains_truth':len(ti)>0,'base_pre_KO':pick(old),'base_post_KO':pick(oldko),'counter_pre_KO':pick(new),'counter_post_KO':pick(newko),'old_neural_active':active(oldko,oldorder),'new_neural_active':active(newko,neworder),'top20_changed':not np.array_equal(oldorder,neworder),'target_in_old_top20':bool(len(ti) and ti[0] in oldorder),'target_in_new_top20':bool(len(ti) and ti[0] in neworder),'retained_mass':mass.tolist(),'effective_weights_old':(w*mass/(w*mass).sum()).tolist(),'effective_weights_new':(w/w.sum()).tolist(),'baseline_replay_maxdiff':diff}
  np.savez_compressed(out/(sid+'.npz'),base=newko);rr.append(r)
 (HERE/'decomposition.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr));summary={'hard_rows':len(rr),'original_base_replay_maxdiff':maxdiff,'C_vote_errors':sum(r['C_vote']!=r['truth'] for r in rr),'C_vote_errors_with_truth_in_candidates':sum(r['C_vote']!=r['truth'] and r['candidate_contains_truth'] for r in rr),'C_vote_errors_with_truth_in_v20_top20':sum(r['C_vote']!=r['truth'] and r['target_in_old_top20'] for r in rr),'base_preKO_correct':sum(r['base_pre_KO']==r['truth'] for r in rr),'base_postKO_correct':sum(r['base_post_KO']==r['truth'] for r in rr),'v20_correct':sum(r['v20']==r['truth'] for r in rr),'mask_top20_changed':sum(r['top20_changed'] for r in rr),'mask_postKO_winner_changed':sum(r['base_post_KO']!=r['counter_post_KO'] for r in rr),'not_public':True}
 (HERE/'DECOMPOSITION.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':prepare()
