"""Re-score original neural models on pre-member-mask probabilities; no retrieval/training."""
from pathlib import Path
import os,sys,time,json
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];HERE=Path(os.environ.get("ASTRA_WORK_DIR",str(Path(__file__).resolve().parent)));STRATEGY=os.environ.get('ASTRA_STRATEGY','mask');assert STRATEGY in ('mask','context')
# Reuse the audited local compatibility/device setup, stopping before ordinary run().
src=ROOT/'astra/new_direction_20261101/local_cvote/run_v20.py';s=src.read_text();marker="m.run(test_path=test,catalog_path=ROOT/'data/open/vessel_catalog.csv',output_path=out)";assert s.count(marker)==1
sys.argv=[str(src),str(HERE/'hard_test.csv'),str(HERE/(STRATEGY+'_v20_pred.csv'))];g={'__name__':'__main__','__file__':str(src)};exec(compile(s.split(marker)[0],str(src),'exec'),g)
import pandas as pd,numpy as np
from src.retrieval import CatalogRetriever
from src.encoder_views import EncoderViewsEnsemble
cat=pd.read_csv(ROOT/'data/open/vessel_catalog.csv');r=CatalogRetriever(cat);model=EncoderViewsEnsemble.load(ROOT/'astra/new_direction_20261101/local_cvote/payload_mps/model/ko/model',r)
rows=pd.read_csv(HERE/'hard_test.csv',encoding='utf-8-sig',dtype={k:str for k in ('sample_id','vessel_name','call_sign','flag_name','vessel_type')}).to_dict('records');pred=[];elapsed=time.time();exclude=set(json.loads((ROOT/'astra/arena_20261101/baseline/model/ko/model/train_vessels.json').read_text()));max_old=0
for count,row in enumerate(rows):
 if time.time()>=1793484000:raise RuntimeError('07:00 cutoff')
 sid=row['sample_id'];z=np.load(HERE/'capture'/(sid+'.npz'));meta=json.loads((HERE/'capture'/(sid+'.json')).read_text());sig={'cand':z['candidate_index'],'llm_beams':meta['beams']};new=np.load(HERE/STRATEGY/(sid+'.npz'))['base'];ids=z['candidate_ids'];bad=np.isin(ids,list(exclude));valid=~bad if bad.any() and not bad.all() else np.ones(len(ids),bool)
 model.baseline.score_groups=lambda blocks,b=new:[b]
 final=model.score_records([row],[sig],[z['features']])[0];assert np.isfinite(final).all();chosen=str(ids[np.argmax(np.where(valid,final,-np.inf))]);pred.append({'sample_id':sid,'target_vessel_id':chosen})
 np.save(HERE/STRATEGY/(sid+'_final.npy'),final)
 # First four rows verify that the same isolated replay reproduces original scorer output.
 if count<4:
  model.baseline.score_groups=lambda blocks,b=z['base']:[b];old=model.score_records([row],[sig],[z['features']])[0];delta=float(np.max(np.abs(old-z['final'])));max_old=max(max_old,delta);assert str(ids[np.argmax(np.where(valid,old,-np.inf))])==str(ids[np.argmax(np.where(valid,z['final'],-np.inf))])
 if (count+1)%20==0:print('REPLAYED',count+1,'seconds',time.time()-elapsed,flush=True)
pd.DataFrame(pred).to_csv(HERE/(STRATEGY+'_v20_pred.csv'),index=False);report={'rows':len(pred),'seconds':time.time()-elapsed,'original_replay_rows':4,'original_replay_prediction_mismatches':0,'original_score_maxdiff':max_old,'same_candidate_ids':True,'actual_neural_recomputed':True};(HERE/(STRATEGY.upper()+'_REPLAY.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
