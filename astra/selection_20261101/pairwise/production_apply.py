"""Fixed .95 pair-verifier diagnostic against actual three-voter C_vote outputs."""
from pathlib import Path
import os,sys,json,pickle,time,datetime,unicodedata,hashlib
sys.dont_write_bytecode=True
assert os.environ.get('PYTHONHASHSEED')=='0'
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;LOCAL=ROOT/'astra/new_direction_20261101/local_cvote'
DATA=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else LOCAL/'eval';TAG=sys.argv[2] if len(sys.argv)>2 else 'production'
V20=Path(sys.argv[3]).resolve() if len(sys.argv)>3 else ROOT/'astra/selection_20261101/v20_hard_pred.csv'
CELL=sys.argv[4] if len(sys.argv)>4 else 'original'
assert CELL in ('original','no_score_all19')
NFEATURES=187 if CELL=='original' else 186
OUTTAG=TAG if CELL=='original' else TAG+'_'+CELL
assert TAG.replace('_','').isalnum()
sys.path.insert(0,str(LOCAL/'deps'))
import numpy as np,pandas as pd,lightgbm as lgb
assert pd.__version__=='2.2.3'

def pair(A,B,sa,sb):return np.concatenate([A,B,A-B,np.asarray(sa-sb,dtype=np.float32)[...,None]],axis=-1).astype(np.float32)[:,:NFEATURES]
def mapping(p):
 d=pd.read_csv(p);assert not d.sample_id.duplicated().any();return d.set_index('sample_id').target_vessel_id.to_dict()
def hangul(s):return any('가'<=c<='힣' for c in unicodedata.normalize('NFC',str(s)))

def main():
 assert time.time()<datetime.datetime(2026,10,31,22,tzinfo=datetime.timezone.utc).timestamp(),'07:00 cutoff'
 t=time.monotonic();test=pd.read_csv(DATA/'test.csv');en=pd.read_csv(DATA/'en_pred.csv').set_index('sample_id');hs=mapping(DATA/'hs_pred.csv');v20=mapping(V20)
 model_path=HERE/'verifier_fitfold2plus4.txt' if CELL=='original' else HERE/'ablation/no_score_all19/fit2plus4.txt'
 model_sha=hashlib.sha256(model_path.read_bytes()).hexdigest()
 if CELL=='no_score_all19' and (HERE/'FOLLOWUP_LOCK.json').exists():assert model_sha==json.loads((HERE/'FOLLOWUP_LOCK.json').read_text())['model_sha256']
 model=lgb.Booster(model_file=str(model_path));features={r['sample_id']:r for r in pickle.loads((HERE/f'{TAG}112_features.pkl').read_bytes())}
 prep=json.loads((HERE/f'{TAG.upper()}_PREPARE.json').read_text());assert Path(prep['data_dir']).samefile(DATA)
 assert model.num_feature()==NFEATURES and model.num_trees()==250
 def one(row):
  sid=str(row['sample_id']);e=str(en.loc[sid,'target_vessel_id']);h=hs[sid];gap=float(en.loc[sid,'ko_gap']);ko=hangul(row['vessel_name']);protected=not ko or gap>=4
  if protected:cvote=e
  elif e==h:cvote=e
  else:
   assert sid in v20,('Incomplete actual v20 outputs',sid)
   cvote=v20[sid]
  reason='english' if not ko else ('gap_ge4' if gap>=4 else None);prediction=cvote;prob=None;best_rank=None;incumbent_rank=None
  if not protected:
   r=features[sid]
   if cvote not in r['ids']:reason='outside112_top20'
   else:
    incumbent=r['ids'].index(cvote);incumbent_rank=incumbent+1;others=[j for j in range(len(r['ids'])) if j!=incumbent]
    if others:
     A=r['F'][others];B=np.repeat(r['F'][incumbent:incumbent+1],len(others),axis=0);sa=r['scores'][others];sb=np.full(len(others),r['scores'][incumbent]);ab=model.predict(pair(A,B,sa,sb),num_threads=2);ba=model.predict(pair(B,A,sb,sa),num_threads=2);p=.5*(ab+1-ba);j=int(np.argmax(p));prob=float(p[j]);best_rank=others[j]+1
     if prob>=.95:prediction=r['ids'][others[j]]
  return {'sample_id':sid,'name':str(row['vessel_name']),'baseline112':e,'hsw':h,'v20':v20.get(sid),'cvote':cvote,'prediction':prediction,'gap110':gap if np.isfinite(gap) else None,'protected_reason':reason,'incumbent112_rank':incumbent_rank,'challenger112_rank':best_rank,'challenger_probability':prob,'changed':prediction!=cvote}
 raw=test.to_dict('records');out=[one(r) for r in raw];forward={r['sample_id']:r for r in out};reverse=[one(r) for r in raw[::-1]]
 assert all(r==forward[r['sample_id']] for r in reverse),'Row reversal difference'
 assert all(not r['changed'] or r['protected_reason'] is None for r in out),'Out of scope change'
 frozen=HERE/f'{OUTTAG}_predictions_before_truth.jsonl';tmp=frozen.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out));tmp.replace(frozen)
 # Truth is intentionally read only after the predictions are saved.
 truth=pd.read_csv(DATA/'truth.csv').set_index('sample_id');assert set(truth.index)==set(forward)
 for r in out:
  r['target']=str(truth.loc[r['sample_id'],'target_vessel_id']);r['variant']=str(truth.loc[r['sample_id'],'variant'])
 report={'data_dir':str(DATA),'v20_predictions':str(V20),'cell':CELL,'post_ablation_selected_followup':CELL=='no_score_all19','model':str(model_path),'features':NFEATURES,'threshold':.95,'threshold_search':False,'fit':'combined OOF2/4 same fixed recipe','incumbent':'actual C_vote; can differ from validated112-top1 incumbent','input_rows':len(out),'hard_feature_rows':len(features),'eligible_rows':sum(r['protected_reason'] is None for r in out),'outside_pool_rows':sum(r['protected_reason']=='outside112_top20' for r in out),'eligible_cvote_not112_top1':sum(r['protected_reason'] is None and r['cvote']!=r['baseline112'] for r in out),'row_reversal_mismatches':0,'protected_changes':0,'seconds':time.monotonic()-t,'independent_public_validation':False,'metrics':{},'mechanism_breakdown':{}}
 for typ in ['clean','noise','all']:
  rr=[r for r in out if typ=='all' or r['variant'].startswith('clean')==(typ=='clean')];res=[r for r in rr if r['cvote']!=r['target'] and r['prediction']==r['target']];harm=[r for r in rr if r['cvote']==r['target'] and r['prediction']!=r['target']]
  report['metrics'][typ]={'rows':len(rr),'vessels':len({r['target'] for r in rr}),'base_correct':sum(r['cvote']==r['target'] for r in rr),'new_correct':sum(r['prediction']==r['target'] for r in rr),'changed':sum(r['changed'] for r in rr),'rescue':len(res),'harm':len(harm),'net':len(res)-len(harm),'rescue_vessels':len({r['target'] for r in res}),'harm_vessels':len({r['target'] for r in harm})}
 report['diagnostic_pass']=report['metrics']['clean']['harm']==0 and report['metrics']['noise']['net']>0
 report['model_sha256']=model_sha
 for kind in ['restore112top1','other_selection']:
  rr=[r for r in out if r['changed'] and (r['prediction']==r['baseline112'])==(kind=='restore112top1')];res=[r for r in rr if r['cvote']!=r['target'] and r['prediction']==r['target']];harm=[r for r in rr if r['cvote']==r['target'] and r['prediction']!=r['target']]
  report['mechanism_breakdown'][kind]={'changed':len(rr),'rescue':len(res),'harm':len(harm),'net':len(res)-len(harm),'incumbent112_ranks':[r['incumbent112_rank'] for r in rr],'challenger112_ranks':[r['challenger112_rank'] for r in rr],'rows':[r['sample_id'] for r in rr]}
 (HERE/f'{OUTTAG}_CVOTE_ROWS.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out));(HERE/f'{OUTTAG}_CVOTE_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
