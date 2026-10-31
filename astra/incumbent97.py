"""Astra: fold-safe matched-recipe reference for the actual79 Korean ensemble."""
from pathlib import Path
import os,sys,json,time,hashlib,zipfile,re,argparse,fcntl,gc,ast,unicodedata
os.environ['KO_DICT']='1';os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,pandas as pd,lightgbm as lgb
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'astra/incumbent97';SRC=BASE/'src';SRC.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(ROOT/'submissions/79_submit.zip') as z:
 for n in ['opus_lib.py','ko_translit_argon.py']:
  p=SRC/n;b=z.read('model/'+n)
  if p.exists():assert p.read_bytes()==b
  else:p.write_bytes(b)
p=SRC/'cv.py';b=(ROOT/'shared/experiments/pinned/build70/cv.py').read_bytes()
if p.exists():assert p.read_bytes()==b
else:p.write_bytes(b)
sys.path.insert(0,str(SRC));import cv,opus_lib as B
UNIVERSAL=list(B.KA._UNIVERSAL_KO_SUBWORDS)
PARAMS=dict(objective='lambdarank',metric='ndcg',eval_at=[1],learning_rate=.05,num_leaves=63,min_data_in_leaf=20,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,lambdarank_truncation_level=10,verbose=-1,num_threads=2)
SEEDS=[0,5000,1000,2000,6000,7000]
MODEL_FIELDS=['sample_id','vessel_name','call_sign','gross_tonnage','international_gross_tonnage','flag_name','vessel_type','arrival_year']

def dump_json(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str));temp.replace(p)

def lockpath(p):
 p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX);return f

def context(fold):
 tr,cat=cv.load();f=cv.folds(tr);fit=tr[f!=fold].copy();va=tr[f==fold].copy();out=BASE/f'fold{fold}';out.mkdir(exist_ok=True)
 assert not set(fit.target_vessel_id)&set(va.target_vessel_id)
 support=json.loads((out/'dictionary_support.json').read_text());B.KA._UNIVERSAL_KO_SUBWORDS=[(r['ko'],r['en']) for r in support['retained']]
 ix=B.Index(cat,fit);llm=B.llm_korean_rows(fit,out/'llm_fit.csv');ix.ko=B.KA.KoMatcher(cat,pd.concat([fit,llm],ignore_index=True))
 return tr,cat,fit,va,ix,llm,out

def prepare(fold):
 tr,cat=cv.load();f=cv.folds(tr);fit=tr[f!=fold].copy();va=tr[f==fold].copy();out=BASE/f'fold{fold}';out.mkdir(exist_ok=True);lock=lockpath(out/'prepare.lock')
 if (out/'plan.json').exists():print('PREPARED_ALREADY',fold,flush=True);return
 ids=set(fit.target_vessel_id);held=set(va.target_vessel_id);catalog=cat.set_index('vessel_id');raw=pd.read_csv(ROOT/'shared/experiments/pinned/build70/llm_korean_scored.csv');raw=raw[raw.vessel_id.isin(ids)].copy()
 pairs=[]
 for r in fit[fit.vessel_name.map(B.is_hangul)].to_dict('records'):pairs.append((r['vessel_name'],r['target_vessel_id'],'native_train'))
 for r in raw.to_dict('records'):pairs.append((str(r['ko']),r['vessel_id'],'train_only_llm'))
 compact=lambda s:re.sub(r'[^a-z0-9]','',str(s).lower());pairs=[(re.sub(r'\s+','',str(k)),v,source,compact(catalog.loc[v,'vessel_name'])) for k,v,source in pairs]
 retained=[];removed=[]
 for ko,en in UNIVERSAL:
  witnesses=[dict(vessel_id=v,source=source,ko_name=k,catalog_name=str(catalog.loc[v,'vessel_name'])) for k,v,source,eng in pairs if ko in k and compact(en) in eng]
  if witnesses:
   assert all(w['vessel_id'] in ids and w['vessel_id'] not in held for w in witnesses)
   retained.append(dict(ko=ko,en=en,witnesses=witnesses[:5],witness_count=len(witnesses)))
  else:removed.append(dict(ko=ko,en=en))
 dump_json(out/'dictionary_support.json',dict(original_count=len(UNIVERSAL),retained=retained,removed=removed,rule='Existing79 loanword entry retained only if outer-fit native or permitted train-only LLM Korean/catalog target pair contains both ko and en substrings; raw global LLM score never used',heldout_vessels=len(held)))
 B.KA._UNIVERSAL_KO_SUBWORDS=[(x['ko'],x['en']) for x in retained];ix=B.Index(cat,fit)
 raw['score']=[ix.ko.score(n,ix.id2i[v])['ko_combined_score'] for n,v in zip(raw.ko,raw.vessel_id)]
 raw.to_csv(out/'llm_fit.csv',index=False);llm=B.llm_korean_rows(fit,out/'llm_fit.csv');ko=fit[fit.vessel_name.map(B.is_hangul)]
 blocks=out/'blocks';blocks.mkdir(exist_ok=True)
 fixed={'real':fit,'nogt':ko.assign(gross_tonnage=np.nan,international_gross_tonnage=np.nan),'llm':llm}
 for name,df in fixed.items():dump_json(blocks/f'{name}.json',df.to_dict('records'))
 plans={}
 for seed in SEEDS:
  augmented=B.augment_korean(fit,seed);noisy=augmented.iloc[len(fit)+len(ko):].copy();corrupt=B.augment_corrupt(fit,seed);rename=B.augment_rename(fit,seed)
  for name,df in [('noisy',noisy),('corrupt',corrupt),('rename',rename)]:dump_json(blocks/f'seed{seed}_{name}.json',df.to_dict('records'))
  order=['real','nogt',f'seed{seed}_noisy',f'seed{seed}_corrupt',f'seed{seed}_rename','llm'];rows=sum((json.loads((blocks/f'{name}.json').read_text()) for name in order),[])
  reference=pd.concat([augmented,corrupt,rename,llm],ignore_index=True).to_dict('records');sig=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest();assert sig(rows)==sig(reference)
  plans[str(seed)]=dict(order=order,queries=len(rows),queries_sha256=sig(rows))
 # Reuse the exact old stress function without executing ranker84 module import side effects.
 tree=ast.parse((ROOT/'astra/ranker84.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='queries');ns={'B':B,'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'old_queries','exec'),ns)
 va_all=va[va.vessel_name.map(B.is_hangul)].copy();rows=ns['queries'](va_all.drop_duplicates('target_vessel_id'),fit)
 for r in va_all.to_dict('records'):r['call_sign']='UNSEEN';r['variant']='native_all_rows';rows.append(r)
 for r in rows:r['query_id']=r['variant']+'::'+r['sample_id'];r['family']='old9' if r['variant']!='native_all_rows' else 'native';r['source_language']='ko'
 assert len({r['query_id'] for r in rows})==len(rows);dump_json(out/'old9_queries.json',rows)
 provenance={'79_zip_sha256':hashlib.sha256((ROOT/'submissions/79_submit.zip').read_bytes()).hexdigest(),'sources':{n:hashlib.sha256((SRC/n).read_bytes()).hexdigest() for n in ['opus_lib.py','ko_translit_argon.py','cv.py']}}
 dump_json(out/'plan.json',dict(fold=fold,fit_rows=len(fit),fit_vessels=len(ids),validation_ko_vessels=va_all.target_vessel_id.nunique(),validation_ko_rows=len(va_all),llm_rows=len(llm),dictionary_retained=len(retained),dictionary_original=len(UNIVERSAL),training_seeds=SEEDS,lgb_params=PARAMS,trees=600,features=B.FEATURES,seeds=plans,source=provenance,mismatches_to_production79=['Vessel-disjoint outer-fold training instead of all train','Universal loanwords filtered to documented outer-fit data support; not identical production lexicon','v1 LLM selection rescored using only outer-fit data','num_threads=2 for resource cap (production training used8); LightGBM seed remains0'],not_actual79_heldout_predictions=True))
 print('PREPARED',fold,'fit',len(fit),'llm',len(llm),'dictionary',len(retained),'/',len(UNIVERSAL),'queries',plans['0']['queries'],flush=True)

def cache_block(ix,out,name,start):
 rows=json.loads((out/'blocks'/f'{name}.json').read_text());cache=out/'features'/name;cache.mkdir(parents=True,exist_ok=True);paths=[]
 for lo in range(0,len(rows),500):
  path=cache/f'{lo:06d}.npz';paths.append(path)
  if path.exists():continue
  lock=lockpath(path.with_suffix('.lock'))
  if path.exists():lock.close();continue
  X=[];y=[];groups=[];flat=[];targets=[]
  for row in rows[lo:lo+500]:
   idx,F=B.query_features(ix,row,B.query_candidates(ix,row));t=ix.id2i[row['target_vessel_id']];X.append(F);y.append(np.array([i==t for i in idx],np.int8));groups.append(len(idx));flat.extend(idx);targets.append(t)
  temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,X=np.concatenate(X),y=np.concatenate(y),groups=np.asarray(groups,np.int32),idx=np.asarray(flat,np.int32),target=np.asarray(targets,np.int32),query_index=np.arange(lo,lo+len(groups),dtype=np.int32));temp.replace(path);lock.close();del X,y;gc.collect()
  print('BLOCK',name,lo+len(groups),len(rows),'elapsed',round(time.time()-start),flush=True)
 return paths

def get_eval_rows(out,kind):
 if kind=='old9':return json.loads((out/'old9_queries.json').read_text())
 p=ROOT/('astra/strategy97/llm_holdout_fold2.csv' if kind=='llm_holdout' else 'astra/strategy97/benchmark_fold2.csv')
 if not p.exists():return None
 return pd.read_csv(p).to_dict('records')

def eval_cache(ix,out,kind,rows):
 sig=hashlib.sha256(json.dumps(rows,sort_keys=True,default=str).encode()).hexdigest();p=out/f'eval_{kind}_{sig[:12]}.npz';lock=lockpath(out/f'eval_{kind}.lock')
 if p.exists():return p
 X=[];idxs=[];groups=[];targets=[];allowed=[];fit_targets=set(ix.cs_map.values())
 for row in rows:
  q={k:row.get(k) for k in MODEL_FIELDS};idx,F=B.query_features(ix,q,B.query_candidates(ix,q));X.append(F);idxs.extend(idx);groups.append(len(idx));targets.append(ix.id2i[row['target_vessel_id']]);allowed.extend(i not in fit_targets for i in idx)
 temp=p.with_suffix('.tmp.npz');np.savez_compressed(temp,X=np.concatenate(X),idx=np.asarray(idxs,np.int32),groups=np.asarray(groups,np.int32),target=np.asarray(targets,np.int32),allowed=np.asarray(allowed,bool));temp.replace(p);dump_json(out/f'eval_{kind}_queries.json',rows);lock.close();print('EVAL_CACHE',kind,len(rows),'candidates',len(idxs),flush=True);return p

def predict_rows(scores,z,rows,ids,method):
 idx=z['idx'];groups=z['groups'];targets=z['target'];allowed=z['allowed'];offset=0;pred=[]
 for j,n in enumerate(groups):
  pool=np.arange(offset,offset+n);valid=pool[allowed[pool]];valid=valid if len(valid) else pool;ranked=sorted(valid,key=lambda i:(-float(scores[i]),int(idx[i])));win=ranked[0];target_rank=next((k+1 for k,i in enumerate(ranked) if idx[i]==targets[j]),None);r=rows[j]
  pred.append(dict(query_id=r['query_id'],sample_id=r['sample_id'],target_vessel_id=ids[int(targets[j])],variant=r.get('variant',''),family=r.get('family',''),source_language=r.get('source_language',''),cohort=r.get('cohort','old9'),method=method,pred=ids[int(idx[win])],correct=bool(idx[win]==targets[j]),recall=bool(targets[j] in idx[pool]),top20_recall=bool(target_rank is not None and target_rank<=20),top100_recall=bool(target_rank is not None and target_rank<=100),target_rank=target_rank));offset+=n
 return pd.DataFrame(pred)

def train(fold,seed):
 start=time.time();tr,cat,fit,va,ix,llm,out=context(fold);seedout=out/f'seed{seed}';seedout.mkdir(exist_ok=True);runlock=lockpath(seedout/'run.lock');plan=json.loads((out/'plan.json').read_text());order=plan['seeds'][str(seed)]['order']
 modelpath=seedout/'ranker.txt'
 if not modelpath.exists():
  generation=(['llm']+[n for n in order if n not in ['llm','real','nogt']]+['real','nogt']) if seed==5000 else order
  for name in generation:cache_block(ix,out,name,start)
  paths=sum(([out/'features'/name/f'{lo:06d}.npz' for lo in range(0,len(json.loads((out/'blocks'/f'{name}.json').read_text())),500)] for name in order),[]);groups=[];ys=[];count=0
  for path in paths:
   with np.load(path) as z:groups.extend(z['groups'].tolist());ys.append(z['y']);count+=len(z['y'])
  X=np.lib.format.open_memmap(seedout/'matrix.npy',mode='w+',dtype=np.float32,shape=(count,len(B.FEATURES)));offset=0
  for path in paths:
   with np.load(path) as z:n=len(z['y']);X[offset:offset+n]=z['X'];offset+=n
  X.flush();y=np.concatenate(ys);assert sum(groups)==len(X)==len(y);print('TRAIN',seed,'queries',len(groups),'candidates',len(y),'elapsed',round(time.time()-start),flush=True)
  m=lgb.train(PARAMS,lgb.Dataset(X,y,group=groups,feature_name=B.FEATURES),600);temp=modelpath.with_suffix('.tmp');m.save_model(str(temp));temp.replace(modelpath);del X,y,ys;gc.collect();print('TRAINED',seed,round(time.time()-start),flush=True)
 else:m=lgb.Booster(model_file=str(modelpath))
 for kind in ['old9','benchmark','llm_holdout']:
  rows=get_eval_rows(out,kind)
  if rows is None:continue
  assert set(r['target_vessel_id'] for r in rows)<=set(va.target_vessel_id)
  cache=eval_cache(ix,out,kind,rows)
  with np.load(cache) as z:
   scores=m.predict(z['X'],num_threads=1);temp=seedout/f'{kind}_scores.tmp.npy';np.save(temp,scores);temp.replace(seedout/f'{kind}_scores.npy');d=predict_rows(scores,z,rows,ix.ids,f'aug{seed}');d.to_csv(seedout/f'{kind}_predictions.csv',index=False)
  print('EVAL',seed,kind,d.groupby(['source_language','cohort','family','variant']).correct.mean().to_dict(),flush=True)
 dump_json(seedout/'manifest.json',dict(fold=fold,augmentation_seed=seed,lgb_seed=0,seconds=time.time()-start,model_sha256=hashlib.sha256(modelpath.read_bytes()).hexdigest(),not_actual_full79_model=True))
 aggregate(fold)

def aggregate(fold):
 out=BASE/f'fold{fold}';lock=lockpath(out/'aggregate.lock');cat=pd.read_csv(ROOT/'data/open/vessel_catalog.csv');ids=cat.vessel_id.astype(str).to_numpy();reports={}
 for kind in ['old9','benchmark','llm_holdout']:
  paths=[p for p in out.glob(f'eval_{kind}_*.npz') if '.tmp.' not in p.name]
  if not paths:continue
  assert len(paths)==1;rows=json.loads((out/f'eval_{kind}_queries.json').read_text());members={s:np.load(out/f'seed{s}/{kind}_scores.npy') for s in SEEDS if (out/f'seed{s}/{kind}_scores.npy').exists()};reports[kind]={}
  for count in [1,2,3,6]:
   subset=SEEDS[:count]
   if not all(s in members for s in subset):continue
   score=sum(members[s] for s in subset)/count
   with np.load(paths[0]) as z:d=predict_rows(score,z,rows,ids,f'incumbent_{count}models')
   d.to_csv(out/f'{kind}_{count}models_predictions.csv',index=False);groups=d.groupby(['source_language','cohort','family','variant'])[['correct','recall','top20_recall','top100_recall']].mean();reports[kind][str(count)]=dict(augmentation_seeds=subset,per_condition=groups.reset_index().to_dict('records'))
   if kind=='old9':reports[kind][str(count)].update(stress_accuracy=float(d[d.family=='old9'].correct.mean()),native_accuracy=float(d[d.family=='native'].correct.mean()))
 dump_json(out/'ensemble_report.json',reports);print('ENSEMBLE_REPORT',json.dumps(reports.get('old9',{})),flush=True)

def weak91(fold,kinds=None):
 tr,cat,fit,va,ix,llm,out=context(fold);sys.path.insert(0,str(ROOT/'shared/experiments/pinned/build84_astra'));import astra_lib as A
 saved=B.KA._UNIVERSAL_KO_SUBWORDS
 try:
  B.KA._UNIVERSAL_KO_SUBWORDS=[];weakix=A.Index(cat,fit);gen=B.llm_korean_rows(fit,ROOT/f'astra/ranker84_fold{fold}/llm_fit.csv');weakix.ko=B.KA.KoMatcher(cat,pd.concat([fit,gen],ignore_index=True))
 finally:B.KA._UNIVERSAL_KO_SUBWORDS=saved
 m=lgb.Booster(model_file=str(ROOT/f'astra/rankless87_fold{fold}/ranker.txt'));keep=[A.FEATURES.index(n) for n in m.feature_name()];fit_ids=set(fit.target_vessel_id)
 for kind in (kinds or ['old9','benchmark','llm_holdout']):
  rows=get_eval_rows(out,kind)
  if rows is None:continue
  rows=[r for r in rows if r.get('source_language','ko')=='ko']
  if kind=='old9':
   with np.load(ROOT/f'astra/neural87_fold{fold}/eval_features_enhanced.npz') as z:
    assert z['sample_id'].tolist()==[r['sample_id'] for r in rows] and z['variant'].tolist()==[r['variant'] for r in rows]
    scores=m.predict(z['X'][:,keep],num_threads=1);d=predict_rows(scores,z,rows,weakix.ids,'weak91_recipe_fold2')
   d.to_csv(out/f'{kind}_weak91_predictions.csv',index=False);print('WEAK91',kind,d.groupby(['source_language','cohort','family','variant']).correct.mean().to_dict(),flush=True);continue
  result=[]
  for j,r in enumerate(rows):
   q={k:r.get(k) for k in MODEL_FIELDS};q['vessel_name']=unicodedata.normalize('NFC',str(q.get('vessel_name') or ''));idx,F=A.query_features(weakix,q,A.query_candidates(weakix,q));s=m.predict(F[:,keep],num_threads=1);order=sorted(range(len(idx)),key=lambda k:(-float(s[k]),idx[k]));allowed_order=[k for k in order if weakix.ids[idx[k]] not in fit_ids] or order;win=idx[allowed_order[0]];target_rank=next((j+1 for j,k in enumerate(allowed_order) if weakix.ids[idx[k]]==r['target_vessel_id']),None)
   result.append(dict(query_id=r['query_id'],sample_id=r['sample_id'],target_vessel_id=r['target_vessel_id'],variant=r.get('variant',''),family=r.get('family',''),source_language=r.get('source_language',''),cohort=r.get('cohort','old9'),production_route='ko' if B.is_hangul(q['vessel_name']) else 'en65_not_retrained_diagnostic_only',method='weak91_recipe_fold2',pred=weakix.ids[win],correct=bool(weakix.ids[win]==r['target_vessel_id']),recall=bool(weakix.id2i[r['target_vessel_id']] in idx),top20_recall=bool(target_rank is not None and target_rank<=20),top100_recall=bool(target_rank is not None and target_rank<=100),target_rank=target_rank))
   if (j+1)%500==0:print('WEAK91_PROGRESS',kind,j+1,len(rows),flush=True)
  d=pd.DataFrame(result);d.to_csv(out/f'{kind}_weak91_predictions.csv',index=False);print('WEAK91',kind,d.groupby(['source_language','cohort','family','variant']).correct.mean().to_dict(),flush=True)

def evaluate_extra(fold,kind):
 tr,cat,fit,va,ix,llm,out=context(fold);rows=get_eval_rows(out,kind);assert rows is not None
 assert set(r['target_vessel_id'] for r in rows)<=set(va.target_vessel_id)
 path=eval_cache(ix,out,kind,rows)
 with np.load(path) as z:
  X=z['X']
  for seed in [0,5000]:
   seedout=out/f'seed{seed}';model=lgb.Booster(model_file=str(seedout/'ranker.txt'));scores=model.predict(X,num_threads=1);np.save(seedout/f'{kind}_scores.npy',scores);d=predict_rows(scores,z,rows,ix.ids,f'aug{seed}');d.to_csv(seedout/f'{kind}_predictions.csv',index=False)
 aggregate(fold)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,default=2);p.add_argument('--prepare',action='store_true');p.add_argument('--seed',type=int);p.add_argument('--aggregate',action='store_true');p.add_argument('--weak91',action='store_true');p.add_argument('--extra',choices=['llm_holdout']);p.add_argument('--weak91-kind',choices=['old9','benchmark','llm_holdout']);a=p.parse_args()
 if a.prepare:prepare(a.fold)
 elif a.seed is not None:train(a.fold,a.seed)
 elif a.aggregate:aggregate(a.fold)
 elif a.weak91:weak91(a.fold,[a.weak91_kind] if a.weak91_kind else None)
 elif a.extra:evaluate_extra(a.fold,a.extra)
