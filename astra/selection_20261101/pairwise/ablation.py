"""Preregistered causal ablation; only no_score_hardest1 is promotion-eligible."""
from pathlib import Path
import os,sys,json,pickle,collections,time
sys.dont_write_bytecode=True
import run as R
import numpy as np,lightgbm as lgb
HERE=Path(__file__).resolve().parent;OUT=HERE/'ablation';OUT.mkdir(exist_ok=True)
FOLDS={f:pickle.loads((HERE/f'fold{f}_features.pkl').read_bytes()) for f in (2,4)}
PRIMARY='no_score_hardest1'

def features(A,B,sa,sb,score):
 x=R.pair_features(A,B,sa,sb)
 return x if score else x[:,:186]

def fit(rows,cell,fold,score,hardest):
 R.checktime();xs=[];ys=[];ships=[];groups=0
 for r in rows:
  if not r['hard'] or not r['truth_in_top20']:continue
  ti=r['ids'].index(r['target']);neg=[j for j in range(len(r['ids'])) if j!=ti]
  if hardest:neg=neg[:1]
  if not neg:continue
  groups+=1;A=np.repeat(r['F'][ti:ti+1],len(neg),axis=0);B=r['F'][neg];sa=np.full(len(neg),r['scores'][ti]);sb=r['scores'][neg]
  xs.extend([features(A,B,sa,sb,score),features(B,A,sb,sa,score)]);ys.extend([np.ones(len(neg)),np.zeros(len(neg))]);ships.extend([r['target']]*(2*len(neg)))
 X=np.concatenate(xs);y=np.concatenate(ys);counts=collections.Counter(ships);w=np.asarray([len(ships)/(len(counts)*counts[s]) for s in ships],dtype=np.float32)
 assert X.shape[1]==(187 if score else 186) and np.isfinite(X).all();assert abs(w.mean()-1)<1e-5
 totals=collections.defaultdict(float)
 for v,z in zip(ships,w):totals[v]+=float(z)
 assert max(totals.values())-min(totals.values())<1e-2
 params={'objective':'binary','num_leaves':15,'max_depth':4,'min_data_in_leaf':30,'learning_rate':.03,'seed':112,'num_threads':2,'verbosity':-1,'deterministic':True,'force_col_wise':True,'feature_pre_filter':False}
 names=[p+n for p in ('A_','B_','D_') for n in R.P.B.FEATURES+['kc_sim','kc_margin']]+(['kc_score_difference'] if score else [])
 def callback(env):R.checktime()
 callback.order=0;m=lgb.train(params,lgb.Dataset(X,y,weight=w,feature_name=names),num_boost_round=250,callbacks=[callback]);dest=OUT/cell;dest.mkdir(exist_ok=True);m.save_model(str(dest/f'fit{fold}.txt'))
 meta={'cell':cell,'fit_fold':fold,'groups':groups,'pairs':len(y),'vessels':len(counts),'feature_width':X.shape[1],'trees':m.num_trees(),'fit_vessel_ids':sorted(counts),'minimum_vessel_weight':min(totals.values()),'maximum_vessel_weight':max(totals.values()),'params':params}
 (dest/f'fit{fold}_TRAIN.json').write_text(json.dumps(meta,indent=2)+'\n');return m,set(counts)

def evaluate(model,rows,cell,fitfold,confirmfold,score,fitships):
 assert not fitships & {r['target'] for r in rows};out=[]
 for r in rows:
  R.checktime();pred=r['baseline'];prob=None;rank=None
  if r['hard'] and len(r['ids'])>1:
   A=r['F'][1:];B=np.repeat(r['F'][:1],len(A),axis=0);sa=r['scores'][1:];sb=np.full(len(A),r['scores'][0]);ab=model.predict(features(A,B,sa,sb,score),num_threads=2);ba=model.predict(features(B,A,sb,sa,score),num_threads=2);p=.5*(ab+1-ba);j=int(np.argmax(p));prob=float(p[j]);rank=j+2
   if prob>=.95:pred=r['ids'][j+1]
  x={k:v for k,v in r.items() if k not in ('F','scores')};x.update(prediction=pred,challenger_probability=prob,challenger_original_rank=rank);out.append(x)
 report={'cell':cell,'fit_fold':fitfold,'confirm_fold':confirmfold,'primary':cell==PRIMARY,'fixed_threshold':.95,'metrics':{}}
 for typ in ('clean','noise','all'):
  rr=[r for r in out if typ=='all' or r['variant'].startswith('clean')==(typ=='clean')];res=[r for r in rr if r['baseline']!=r['target'] and r['prediction']==r['target']];harm=[r for r in rr if r['baseline']==r['target'] and r['prediction']!=r['target']]
  report['metrics'][typ]={'rows':len(rr),'vessels':len({r['target'] for r in rr}),'hard_rows':sum(r['hard'] for r in rr),'changed':sum(r['prediction']!=r['baseline'] for r in rr),'rescue':len(res),'harm':len(harm),'net':len(res)-len(harm),'rescue_vessels':len({r['target'] for r in res}),'harm_vessels':len({r['target'] for r in harm}),'truth_missing_top20':sum(r['hard'] and not r['truth_in_top20'] for r in rr)}
 report['protected_changes']=sum(not r['hard'] and r['prediction']!=r['baseline'] for r in out);assert report['protected_changes']==0
 report['screen_pass']=report['metrics']['clean']['harm']==0 and report['metrics']['noise']['net']>0
 tag=f'fit{fitfold}_confirm{confirmfold}';dest=OUT/cell;(dest/f'{tag}_ROWS.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out));(dest/f'{tag}_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);return report

def main():
 start=time.monotonic();R.checktime();result={'score_all19':json.loads((HERE/'RESULTS.json').read_text())['directions']}
 for cell,score,hardest in [('score_hardest1',True,True),('no_score_all19',False,False),('no_score_hardest1',False,True)]:
  result[cell]=[]
  for a,b in [(2,4),(4,2)]:
   model,ships=fit(FOLDS[a],cell,a,score,hardest);result[cell].append(evaluate(model,FOLDS[b],cell,a,b,score,ships))
 primary_pass=all(r['screen_pass'] for r in result[PRIMARY]);summary={'primary':PRIMARY,'primary_both_pass':primary_pass,'cells':result,'no_best_of_grid_selection':True,'seconds':time.monotonic()-start,'fullfit_prepared':False}
 if primary_pass:
  model,ships=fit(FOLDS[2]+FOLDS[4],PRIMARY,'2plus4',False,True);summary['fullfit_prepared']=True;summary['fullfit_vessels']=len(ships)
 (OUT/'RESULTS.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'primary_pass':primary_pass,'fullfit_prepared':summary['fullfit_prepared'],'seconds':summary['seconds']}),flush=True)

if __name__=='__main__':main()
