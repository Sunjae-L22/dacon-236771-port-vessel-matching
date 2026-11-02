"""Matched62 vs64-feature rankers. Frozen E5, official fold-safe cached training, no submissions."""
from pathlib import Path
import sys,os,json,time,datetime,pickle,gc
sys.dont_write_bytecode=True
os.environ.update(KO_DICT='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np,lightgbm as lgb
import subprocess
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'astra/research_20261101'));from channel_model import normalize,hangul,read
sys.path.insert(0,str(ROOT/'astra/byt5_reuse_20261031/package'));import baseline_phase as P
sys.path.insert(0,str(ROOT/'astra/byt5_base112_20261031/snapshot'));from kocat import KoCat,kc_features
DEADLINE=datetime.datetime(2026,10,31,22,tzinfo=datetime.timezone.utc).timestamp()
def checktime():
 if time.time()>=DEADLINE:raise RuntimeError('07:00 cutoff')

def main(fold):
 start=time.time();dest=OUT/f'fold{fold}';dest.mkdir(exist_ok=False);cache=ROOT/f'astra/incumbent97/fold{fold}';plan=json.loads((cache/'plan.json').read_text());order=plan['seeds']['0']['order'];blocks={n:json.loads((cache/f'blocks/{n}.json').read_text()) for n in order}
 qs=json.loads((ROOT/f'astra/noise113/fold{fold}/eval/queries.json').read_text());packets=[json.loads(s) for s in (ROOT/f'astra/byt5_base112_20261031/fold{fold}/packets112.jsonl').read_text().splitlines()];assert len(qs)==len(packets)
 cat,ix,fx,models=P.context(fold);excluded=set(ix.cs_map.values());kc=KoCat(cat,ROOT/'shared/kocat/out/catalog_ko.csv');ids=cat.vessel_id.astype(str).tolist();assert ids==json.loads((ROOT/'astra/new_direction_20261101/dense/catalog_ids.json').read_text())
 table={r['vessel_name']:r for r in read(ROOT/'shared/kocat/out/catalog_ko.csv')};owners=[];j=0
 for r in cat.to_dict('records'):
  a=table.get(r['vessel_name'],{});ss=sorted({normalize(a.get(f'ko{i}','')) for i in range(4) if hangul(a.get(f'ko{i}',''))});owners.append(list(range(j,j+len(ss))));j+=len(ss)
 E=np.load(ROOT/'astra/new_direction_20261101/dense/alias_vectors.npy');assert len(E)==j
 texts=list(dict.fromkeys(normalize(r['vessel_name']) for n,rr in blocks.items() if n!='llm' for r in rr if hangul(r['vessel_name'])));texts=list(dict.fromkeys(texts+[normalize(q['vessel_name']) for q in qs]))
 (dest/'encode_names.json').write_text(json.dumps(texts,ensure_ascii=False));subprocess.run([sys.executable,str(OUT/'embed_worker.py'),str(dest/'encode_names.json'),str(dest/'query_vectors.npy')],check=True);V=np.load(dest/'query_vectors.npy');vectors={s:V[i] for i,s in enumerate(texts)}
 def dense(name,idx):
  if not hangul(name):return np.full((len(idx),2),np.nan,np.float32)
  v=vectors[normalize(name)];sc=np.array([float((E[owners[i]]@v).max()) if owners[i] else np.nan for i in idx],np.float32);valid=np.isfinite(sc)
  if valid.any():
   sort=np.sort(sc[valid]);best=sort[-1];second=sort[-2] if len(sort)>1 else 0.;other=np.where(sc>=best,second,best);return np.column_stack([sc,sc-other]).astype(np.float32)
  return np.full((len(idx),2),np.nan,np.float32)
 def kc_col(name,idx):
  return kc_features(kc,name,idx) if hangul(name) else np.full((len(idx),2),np.nan,np.float32)
 # All feature rows / labels / groups identical; only two columns differ.
 xx=[];yy=[];groups=[]
 for name in order:
  rows=blocks[name]
  for lo in range(0,len(rows),500):
   checktime()
   with np.load(cache/f'features/{name}/{lo:06d}.npz') as z:
    X=z['X'];y=z['y'];g=z['groups'];idx=z['idx'];qi=z['query_index'];extra=[];off=0
    for n,q in zip(g,qi):
     ci=idx[off:off+n];nm=rows[int(q)]['vessel_name'];cols=np.full((n,4),np.nan,np.float32) if name=='llm' else np.hstack([kc_col(nm,ci),dense(nm,ci)]);extra.append(cols);off+=n
    xx.append(np.hstack([X,np.concatenate(extra)]));yy.append(y);groups.extend(g.tolist())
  print('BLOCK',fold,name,'seconds',round(time.time()-start),flush=True)
 X=np.concatenate(xx);y=np.concatenate(yy);del xx,yy;gc.collect();params=dict(plan['lgb_params'],num_threads=2);names=P.B.FEATURES+['kc_sim','kc_margin','e5_alias_sim','e5_alias_margin'];trained={}
 print('MATRIX',fold,X.shape,len(groups),flush=True)
 for width in (62,64):
  checktime();m=lgb.train(params,lgb.Dataset(X[:,:width],y,group=groups,feature_name=names[:width]),600);m.save_model(str(dest/f'ranker{width}.txt'));trained[width]=m;print('TRAINED',fold,width,round(time.time()-start),flush=True)
 del X,y;gc.collect();rr=[];evalcache={};oldkm=lgb.Booster(model_file=str(ROOT/f'astra/byt5_base112_20261031/snapshot/fold{fold}_kc.txt'));old_diff=0;matched_diff=0
 for i,(q,p) in enumerate(zip(qs,packets)):
  checktime();assert q['query_id']==p['query_id']
  if not p['primary']:continue
  r={k:p[k] for k in ('query_id','name','variant','target','baseline')};r['gate110_gap']=p['gate110_gap'];r.update(p62=p['baseline'],p64=p['baseline'])
  if p['gate110_gap']<4:
   key=tuple(str(q.get(k)) for k in ('vessel_name','gross_tonnage','international_gross_tonnage','flag_name','vessel_type','arrival_year'))
   if key not in evalcache:
    cand=fx.add(P.B.query_candidates(ix,q),q)
    for rank,(ci,_) in enumerate(kc.search(q['vessel_name'],50)):
     e=cand.setdefault(int(ci),[999,999,999]);e[0]=min(e[0],rank)
    idx,F=P.B.query_features(ix,q,cand);F62=np.hstack([F,kc_col(q['vessel_name'],idx)]);F64=np.hstack([F62,dense(q['vessel_name'],idx)]);sc62=trained[62].predict(F62,num_threads=2);sc64=trained[64].predict(F64,num_threads=2);old=oldkm.predict(F62,num_threads=2);allow=[j for j,v in enumerate(idx) if v not in excluded] or list(range(len(idx)))
    select=lambda sc:ids[idx[min(allow,key=lambda j:(-float(sc[j]),idx[j]))]]
    evalcache[key]=(select(sc62),select(sc64),select(old),float(np.max(np.abs(sc62-old))))
   a,b,c,d=evalcache[key];r.update(p62=a,p64=b);old_diff+=c!=p['baseline'];matched_diff+=a!=p['baseline'];r['matched_old_score_maxdiff']=d
  rr.append(r)
  if (i+1)%400==0:print('EVAL',fold,i+1,round(time.time()-start),flush=True)
 result={'fold':fold,'seconds':time.time()-start,'baseline':'matched600-tree seed0 62-feature model; not production112 sixKC/C_vote','unchanged_candidate_generation':True,'old_proxy_reproduction_mismatch':old_diff,'matched62_vs_old_proxy_prediction_difference':matched_diff,'metrics':{}}
 for ref in ('baseline','p62'):
  result['metrics'][ref]={}
  for condition in ('clean','noise','all'):
   xs=[r for r in rr if condition=='all' or r['variant'].startswith('clean')==(condition=='clean')];res=[r for r in xs if r[ref]!=r['target'] and r['p64']==r['target']];harm=[r for r in xs if r[ref]==r['target'] and r['p64']!=r['target']];result['metrics'][ref][condition]={'rows':len(xs),'rescue':len(res),'harm':len(harm),'net':len(res)-len(harm),'rescue_vessels':len({r['target'] for r in res}),'harm_vessels':len({r['target'] for r in harm}),'changed':sum(r['p64']!=r[ref] for r in xs)}
 (dest/'rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr));(dest/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main(int(sys.argv[1]))
