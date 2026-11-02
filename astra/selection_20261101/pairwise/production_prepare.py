"""Regenerate immutable production112 feature/score traces without truth input."""
from pathlib import Path
import os,sys,json,pickle,time,datetime,unicodedata,importlib.util,hashlib,gc
sys.dont_write_bytecode=True
assert os.environ.get('PYTHONHASHSEED')=='0'
os.environ.update(KO_DICT='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
LOCAL=ROOT/'astra/new_direction_20261101/local_cvote';sys.path.insert(0,str(LOCAL/'deps'))
DATA=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else LOCAL/'eval'
TAG=sys.argv[2] if len(sys.argv)>2 else 'production'
assert TAG.replace('_','').isalnum()
import numpy as np,pandas as pd,lightgbm as lgb
assert pd.__version__=='2.2.3'
SCRIPT=ROOT/'astra/arena_20261101/baseline/model/en/script.py'
spec=importlib.util.spec_from_file_location('production112_readonly',SCRIPT);M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
from fuzzy_ext import FuzzyExt
from kocat import KoCat,kc_features
sys.path.insert(0,str(ROOT/'astra/arena_highgain_20261101/package'))
from trace_hooks import digest_row

def checktime():
 if time.time()>=datetime.datetime(2026,10,31,22,tzinfo=datetime.timezone.utc).timestamp():raise RuntimeError('07:00 cutoff')

def main():
 checktime();t=time.monotonic();test=pd.read_csv(DATA/'test.csv');cat=pd.read_csv(ROOT/'data/open/vessel_catalog.csv')
 traces={r['sample_id']:r for r in map(json.loads,(DATA/'en_trace.jsonl').read_text().splitlines())}
 preds=pd.read_csv(DATA/'en_pred.csv').set_index('sample_id').target_vessel_id.to_dict()
 stats=pickle.loads((SCRIPT.parent/'model/stats.pkl').read_bytes());ix=M.L.Index(cat,None);ix.cs_map={k:ix.id2i[v] for k,v in stats['cs_map'].items() if v in ix.id2i};ix.flag_p,ix.type_p=stats['flag_p'],stats['type_p'];ix.train_targets=set(ix.cs_map.values());ix.ko=M.L.KA.KoMatcher(cat,stats['ko_pairs']);ix.fit_alias(stats['alias_rows'])
 fx=FuzzyExt(ix,cat);kc=KoCat(cat,SCRIPT.parent/'model/catalog_ko.csv');km=[lgb.Booster(model_file=str(p)) for p in sorted((SCRIPT.parent/'model').glob('kc113_*.txt'))];bm=[lgb.Booster(model_file=str(p)) for p in sorted((SCRIPT.parent/'model').glob('ranker_*.txt'))]
 assert len(km)==len(bm)==6
 for m in km:assert m.feature_name()==M.L.FEATURES+['kc_sim','kc_margin']
 excluded={ix.id2i[v] for v in stats['cs_map'].values() if v in ix.id2i};records=[];max_score_difference=0.;max_gap_difference=0.;row_mismatches=0
 for row in test.to_dict('records'):
  checktime();sid=str(row['sample_id']);tr=traces.get(sid);name=unicodedata.normalize('NFC',str(row.get('vessel_name') or ''))
  if not tr or tr['kind']!='ranked' or not M.L.is_hangul(name) or tr['gap110']>=4:continue
  assert tr['score_family']=='kc112' and row.get('call_sign') not in ix.cs_map
  assert digest_row(row)==tr['input_fingerprint'],('input dtype/schema mismatch',sid)
  idx0,F60=M.L.query_features(ix,row,fx.add(M.L.query_candidates(ix,row),row));s0=sum(m.predict(F60,num_threads=2) for m in bm);a0=sorted([j for j,ci in enumerate(idx0) if ci not in excluded] or list(range(len(idx0))),key=lambda j:(-float(s0[j]),idx0[j]));gap=float(s0[a0[0]]-s0[a0[1]])/6 if len(a0)>1 else 99.;gd=abs(gap-tr['gap110']);max_gap_difference=max(max_gap_difference,gd);assert gd==0.,('gap mismatch',sid,gd)
  cand=fx.add(M.L.query_candidates(ix,row),row)
  for rk,(ci,_) in enumerate(kc.search(name,50)):
   e=cand.setdefault(int(ci),[999,999,999]);e[0]=min(e[0],rk)
  idx,F=M.L.query_features(ix,row,cand);F62=np.hstack([F,kc_features(kc,name,idx)]);score=sum(m.predict(F62,num_threads=2) for m in km);ids=[str(ix.ids[int(i)]) for i in idx]
  assert ids==tr['candidate_ids'],('all candidate IDs/order mismatch',sid)
  d=float(np.max(np.abs(score-np.asarray(tr['scores']))));max_score_difference=max(max_score_difference,d);assert d==0.,('score sum mismatch',sid,d)
  allowed=[j for j,ci in enumerate(idx) if ci not in excluded] or list(range(len(idx)));order=sorted(allowed,key=lambda j:(-float(score[j]),idx[j]));top=order[:20]
  assert ids[top[0]]==preds[sid],('original112 top1 mismatch',sid)
  assert [ids[j] for j in top]==tr['top20_ids'],('top20 mismatch',sid)
  records.append({'sample_id':sid,'name':name,'gap110':gap,'baseline112':preds[sid],'ids':[ids[j] for j in top],'F':F62[top].astype(np.float32),'scores':score[top]/6,'all_ids':ids,'all_F62':F62,'all_score_sum':score,'top_indices':top,'input_fingerprint':tr['input_fingerprint']})
  if len(records)%25==0:print(json.dumps({'prepared':len(records),'seconds':time.monotonic()-t}),flush=True)
 expected=sum(str(r.sample_id) in traces and traces[str(r.sample_id)]['kind']=='ranked' and traces[str(r.sample_id)]['gap110']<4 for r in test.itertuples())
 assert len(records)==expected,(len(records),expected)
 if DATA==LOCAL/'eval':assert len(records)==198,len(records)
 tmp=HERE/f'{TAG}112_features.pkl.{os.getpid()}.tmp'
 with tmp.open('wb') as f:pickle.dump(records,f)
 tmp.replace(HERE/f'{TAG}112_features.pkl')
 report={'data_dir':str(DATA),'input_rows':len(test),'hard_rows':len(records),'pandas_version':pd.__version__,'full_candidate_ID_order_mismatches':0,'top20_and_top1_mismatches':0,'input_fingerprint_mismatches':0,'max_six_model_score_difference':max_score_difference,'max_original110_gap_difference':max_gap_difference,'seconds':time.monotonic()-t,'kc_models':len(km),'pair_score_divisor':6,'feature_names':km[0].feature_name(),'truth_loaded':False}
 (HERE/f'{TAG.upper()}_PREPARE.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
