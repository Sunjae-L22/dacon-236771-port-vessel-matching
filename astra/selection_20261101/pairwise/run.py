from pathlib import Path
import os,sys,json,time,datetime,pickle,collections,gc,hashlib
sys.dont_write_bytecode=True
os.environ.update(KO_DICT='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')
import numpy as np
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'astra/byt5_reuse_20261031/package'))
import baseline_phase as P
sys.path.insert(0,str(ROOT/'astra/byt5_base112_20261031/snapshot'))
from kocat import KoCat,kc_features
CUTOFF=datetime.datetime(2026,10,31,22,tzinfo=datetime.timezone.utc).timestamp()
START=time.monotonic()

def checktime():
 if time.time()>=CUTOFF:raise RuntimeError('07:00 cutoff reached')
 if time.monotonic()-START>45*60:raise RuntimeError('45 minute experiment watchdog')

def prepare(fold):
 start=time.monotonic();queries=json.loads((ROOT/f'astra/noise113/fold{fold}/eval/queries.json').read_text());packets=[json.loads(s) for s in (ROOT/f'astra/byt5_base112_20261031/fold{fold}/packets112.jsonl').read_text().splitlines()]
 assert len(queries)==len(packets)
 cat,ix,fx,base110=P.context(fold);assert not P.B.USE_ALIAS
 excluded=set(ix.cs_map.values());ids=cat.vessel_id.astype(str).tolist();kc=KoCat(cat,ROOT/'shared/kocat/out/catalog_ko.csv');model=lgb.Booster(model_file=str(ROOT/f'astra/byt5_base112_20261031/snapshot/fold{fold}_kc.txt'))
 assert model.feature_name()==P.B.FEATURES+['kc_sim','kc_margin']
 records=[];cache={};hard_count=0;max_score_diff=0.;top100_mismatch=0;baseline_mismatch=0;target_fit_overlap=0
 for q,p in zip(queries,packets):
  checktime();assert q['query_id']==p['query_id'];assert q['target_vessel_id']==p['target']
  if not p['primary']:continue
  row={k:q.get(k) for k in ('sample_id','vessel_name','call_sign','gross_tonnage','international_gross_tonnage','flag_name','vessel_type','arrival_year')}
  ko=P.B.is_hangul(str(q['vessel_name']));r={k:p[k] for k in ('query_id','name','variant','target','baseline')};r.update(fold=fold,gap110=float(p['gate110_gap']),korean=ko,hard=bool(ko and p['gate110_gap']<4))
  if r['hard']:
   hard_count+=1;assert row['call_sign'] not in ix.cs_map
   target_fit_overlap+=ix.id2i[p['target']] in excluded
   key=tuple(str(row.get(k)) for k in ('vessel_name','gross_tonnage','international_gross_tonnage','flag_name','vessel_type','arrival_year'))
   if key not in cache:
    cand=fx.add(P.B.query_candidates(ix,row),row)
    for rank,(ci,_) in enumerate(kc.search(row['vessel_name'],50)):
     e=cand.setdefault(int(ci),[999,999,999]);e[0]=min(e[0],rank)
    idx,F=P.B.query_features(ix,row,cand);F62=np.hstack([F,kc_features(kc,row['vessel_name'],idx)]).astype(np.float32);score=model.predict(F62,num_threads=2)
    allowed=[j for j,ci in enumerate(idx) if ci not in excluded] or list(range(len(idx)))
    order=sorted(allowed,key=lambda j:(-float(score[j]),idx[j]));top=order[:20]
    cache[key]={'ids':[ids[idx[j]] for j in top],'F':F62[top].copy(),'scores':score[top].copy(),'top100_ids':[ids[idx[j]] for j in order[:100]],'top100_scores':[float(score[j]) for j in order[:100]],'top100_indices':[int(idx[j]) for j in order[:100]]}
   payload=cache[key];expected=p['candidates'][:100]
   same=payload['top100_ids']==[c['id'] for c in expected] and payload['top100_indices']==[int(c['index']) for c in expected];top100_mismatch+=not same;assert same,('candidate mismatch',fold,p['query_id'])
   delta=max(abs(a-float(b['score'])) for a,b in zip(payload['top100_scores'],expected));max_score_diff=max(max_score_diff,delta);assert delta<1e-6,('score mismatch',fold,p['query_id'],delta)
   baseline_mismatch+=payload['ids'][0]!=p['baseline'];assert payload['ids'][0]==p['baseline']
   r.update(ids=payload['ids'],F=payload['F'],scores=payload['scores'],truth_in_top20=p['target'] in payload['ids'])
  records.append(r)
  if len(records)%200==0:print(json.dumps({'prepare_fold':fold,'primary':len(records),'hard':hard_count,'seconds':time.monotonic()-start}),flush=True)
 assert target_fit_overlap==0
 report={'fold':fold,'primary_rows':len(records),'hard_rows':hard_count,'hard_vessels':len({r['target'] for r in records if r['hard']}),'unique_hard_metadata_queries':len(cache),'truth_absent_top20':sum(r['hard'] and not r['truth_in_top20'] for r in records),'top100_mismatches':top100_mismatch,'baseline_mismatches':baseline_mismatch,'max_single_model_score_difference':max_score_diff,'target_in_base_fit_count':target_fit_overlap,'feature_names':model.feature_name(),'seconds':time.monotonic()-start}
 with (HERE/f'fold{fold}_features.pkl').open('wb') as f:pickle.dump(records,f)
 (HERE/f'fold{fold}_PREPARE.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
 del ix,kc,cache,base110,model;gc.collect();return records

def pair_features(A,B,sA,sB):
 A=np.asarray(A,dtype=np.float32);B=np.asarray(B,dtype=np.float32)
 return np.concatenate([A,B,A-B,np.asarray(sA-sB,dtype=np.float32)[...,None]],axis=-1)

def fit(train,fold):
 xx=[];yy=[];ships=[];groups=0
 for r in train:
  if not r['hard'] or not r['truth_in_top20']:continue
  ti=r['ids'].index(r['target']);neg=[j for j in range(len(r['ids'])) if j!=ti]
  if not neg:continue
  groups+=1;A=np.repeat(r['F'][ti:ti+1],len(neg),axis=0);B=r['F'][neg];sa=np.full(len(neg),r['scores'][ti]);sb=r['scores'][neg]
  xx.extend([pair_features(A,B,sa,sb),pair_features(B,A,sb,sa)]);yy.extend([np.ones(len(neg)),np.zeros(len(neg))]);ships.extend([r['target']]*(2*len(neg)))
 X=np.concatenate(xx);y=np.concatenate(yy);counts=collections.Counter(ships);w=np.asarray([len(ships)/(len(counts)*counts[v]) for v in ships],dtype=np.float32)
 assert X.shape[1]==187 and np.isfinite(X).all();assert abs(w.mean()-1)<1e-5
 totals=collections.defaultdict(float)
 for v,z in zip(ships,w):totals[v]+=float(z)
 assert max(totals.values())-min(totals.values())<1e-2
 params={'objective':'binary','num_leaves':15,'max_depth':4,'min_data_in_leaf':30,'learning_rate':.03,'seed':112,'num_threads':2,'verbosity':-1,'deterministic':True,'force_col_wise':True,'feature_pre_filter':False}
 names=[prefix+name for prefix in ('A_','B_','D_') for name in P.B.FEATURES+['kc_sim','kc_margin']]+['kc_score_difference']
 def callback(env):checktime()
 callback.order=0
 m=lgb.train(params,lgb.Dataset(X,y,weight=w,feature_name=names),num_boost_round=250,callbacks=[callback]);m.save_model(str(HERE/f'verifier_fitfold{fold}.txt'))
 report={'fold':fold,'groups':groups,'pairs_with_reverse':len(y),'vessels':len(counts),'feature_width':X.shape[1],'min_vessel_weight':min(totals.values()),'max_vessel_weight':max(totals.values()),'weight_mean':float(w.mean()),'params':params,'trees':m.num_trees(),'fit_vessel_ids':sorted(counts)}
 (HERE/f'fold{fold}_TRAIN.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='fit_vessel_ids'}),flush=True);return m,set(counts)

def evaluate(model,records,fitfold,confirmfold,fitships):
 assert not fitships & {r['target'] for r in records}
 out=[];symmetry_max=0.
 for r in records:
  checktime();pred=r['baseline'];prob=None;rank=None
  if r['hard'] and len(r['ids'])>1:
   A=r['F'][1:];B=np.repeat(r['F'][:1],len(A),axis=0);sa=r['scores'][1:];sb=np.full(len(A),r['scores'][0]);ab=pair_features(A,B,sa,sb);ba=pair_features(B,A,sb,sa)
   pAB=model.predict(ab,num_threads=2);pBA=model.predict(ba,num_threads=2);sym=.5*(pAB+1-pBA);rev=.5*(pBA+1-pAB);symmetry_max=max(symmetry_max,float(np.max(np.abs(sym+rev-1))))
   j=int(np.argmax(sym));prob=float(sym[j]);rank=j+2
   if prob>=.95:pred=r['ids'][j+1]
  x={k:v for k,v in r.items() if k not in ('F','scores')};x.update(prediction=pred,challenger_probability=prob,challenger_original_rank=rank);out.append(x)
 report={'fit_fold':fitfold,'confirm_fold':confirmfold,'fixed_threshold':.95,'fit_confirmation_vessel_overlap':0,'symmetry_error_max':symmetry_max,'metrics':{},'scope':'cross-fold pair verifier on single-KC-model OOF112; not C_vote, not strict nested CV, not Public estimate'}
 for condition in ('clean','noise','all'):
  rr=[r for r in out if condition=='all' or r['variant'].startswith('clean')==(condition=='clean')]
  rescue=[r for r in rr if r['baseline']!=r['target'] and r['prediction']==r['target']];harm=[r for r in rr if r['baseline']==r['target'] and r['prediction']!=r['target']]
  report['metrics'][condition]={'rows':len(rr),'vessels':len({r['target'] for r in rr}),'hard_rows':sum(r['hard'] for r in rr),'base_correct':sum(r['baseline']==r['target'] for r in rr),'new_correct':sum(r['prediction']==r['target'] for r in rr),'changed':sum(r['prediction']!=r['baseline'] for r in rr),'rescue':len(rescue),'harm':len(harm),'net':len(rescue)-len(harm),'rescue_vessels':len({r['target'] for r in rescue}),'harm_vessels':len({r['target'] for r in harm}),'hard_truth_missing_top20':sum(r['hard'] and not r['truth_in_top20'] for r in rr)}
 report['protected_changes']=sum(not r['hard'] and r['prediction']!=r['baseline'] for r in out);assert report['protected_changes']==0
 report['screen_pass']=report['metrics']['clean']['harm']==0 and report['metrics']['noise']['net']>0
 tag=f'fit{fitfold}_confirm{confirmfold}';(HERE/(tag+'_rows.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out));(HERE/(tag+'_RESULTS.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True);return report

def main():
 checktime();f2=prepare(2);f4=prepare(4);m2,ids2=fit(f2,2);r24=evaluate(m2,f4,2,4,ids2);m4,ids4=fit(f4,4);r42=evaluate(m4,f2,4,2,ids4)
 summary={'directions':[r24,r42],'both_pass':r24['screen_pass'] and r42['screen_pass'],'seconds':time.monotonic()-START,'production_prepared':False,'public_submissions':0,'threshold_search':False}
 (HERE/'RESULTS.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'final_both_pass':summary['both_pass'],'seconds':summary['seconds']}),flush=True)

if __name__=='__main__':main()
