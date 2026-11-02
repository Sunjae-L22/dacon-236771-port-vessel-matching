"""Separate sensitivity experiment: allowed-only relative context, fixed pair features and weights."""
from pathlib import Path
import sys,os,json,csv,time
sys.dont_write_bytecode=True
import numpy as np,lightgbm,pandas as pd
ROOT=Path(__file__).resolve().parents[2];HERE=Path(os.environ.get("ASTRA_WORK_DIR",str(Path(__file__).resolve().parent)));folder=ROOT/'astra/new_direction_20261101/local_cvote/payload_mps/model/ko/model';sys.path.insert(0,str(folder))
from src.ensemble import Ensemble
from src.features import FEATURE_NAMES
sys.path.insert(0,str(ROOT/'astra/arena_highgain_20261101/package'));from trace_hooks import load_entry
N={v:i for i,v in enumerate(FEATURE_NAMES)}
COLS=['key_rank_in_group','key_lev_gap_to_best','llm_ratio_rank_in_group','llm_ratio_gap_to_best','sim_word_gap_to_best','sim_compact_gap_to_best','dl_rank_in_group','group_size']
def rebuild(F,valid):
 out=F.copy();positions=np.flatnonzero(valid)
 def col(name):return F[valid,N[name]]
 def assign(name,a):out[positions,N[name]]=a
 assign('key_rank_in_group',(-col('key_lev')).argsort(kind='stable').argsort(kind='stable'))
 assign('key_lev_gap_to_best',col('key_lev').max()-col('key_lev'))
 if np.any(col('has_llm')):
  assign('llm_ratio_rank_in_group',(-col('llm_ratio_max')).argsort(kind='stable').argsort(kind='stable'))
  assign('llm_ratio_gap_to_best',col('llm_ratio_max').max()-col('llm_ratio_max'))
 for source,target in [('sim_word','sim_word_gap_to_best'),('sim_compact','sim_compact_gap_to_best')]:assign(target,col(source).max()-col(source))
 assign('dl_rank_in_group',(-col('dl_norm')).argsort(kind='stable').argsort(kind='stable'));assign('group_size',len(positions))
 return out

def main():
 start=time.time();dest=HERE/'context';dest.mkdir(exist_ok=True);rows=list(csv.DictReader((HERE/'hard_test.csv').open(encoding='utf-8-sig')));excluded=set(json.loads((folder/'train_vessels.json').read_text()));data=[];blocks=[];changed_columns=set();noop=0
 for row in rows:
  sid=row['sample_id'];z=np.load(HERE/'capture'/(sid+'.npz'));F=z['features'];bad=np.isin(z['candidate_ids'],list(excluded));valid=~bad if bad.any() and not bad.all() else np.ones(len(F),bool)
  same=rebuild(F,np.ones(len(F),bool));assert np.array_equal(same,F,equal_nan=True),(sid,'full-context formula mismatch');noop+=1
  new=rebuild(F,valid);outside=[i for i,k in enumerate(FEATURE_NAMES) if k not in COLS];assert np.array_equal(F[:,outside],new[:,outside],equal_nan=True);assert np.array_equal(F[~valid],new[~valid],equal_nan=True)
  changed_columns.update(FEATURE_NAMES[i] for i in range(F.shape[1]) if not np.array_equal(F[:,i],new[:,i],equal_nan=True));blocks.append(new);data.append((row,z['candidate_ids'],z['candidate_index'],valid))
 model=Ensemble.load(folder);scores=model.score_groups(blocks);m=load_entry(folder.parent/'script.py');cat=pd.read_csv(ROOT/'data/open/vessel_catalog.csv');table=json.loads((folder/'catalog_ko.json').read_text());kn=[m._ko_norm(table.get(str(n),'')) for n in cat.vessel_name]
 for (row,ids,idx,valid),base in zip(data,scores):
  base=np.where(valid,base,0.);base/=base.sum();zz=np.full(len(base),-np.inf);zz[valid]=np.log(base[valid])+m.KO_W*m.ko_similarity(row['vessel_name'],idx,kn)[valid];zz-=zz[valid].max();p=np.exp(zz);p/=p.sum();assert np.isfinite(p).all();np.savez_compressed(dest/(row['sample_id']+'.npz'),base=p)
 report={'rows':len(rows),'full_context_reconstruction_exact':noop,'changed_columns':sorted(changed_columns),'only_allowed_relative_columns_changed':True,'candidate_ids_unchanged':True,'original_ensemble_masking_order_kept':True,'seconds':time.time()-start,'scope':'inference sensitivity test; existing training used original full-candidate relative features, so this is a distribution shift, not asserted bug correction'};(HERE/'CONTEXT_PREPARE.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
