"""Original v20 weights/code; runtime device-only redirection to local MPS."""
from pathlib import Path
import os,sys,json,time,hashlib,shutil
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTORCH_ENABLE_MPS_FALLBACK='1')
sys.path.insert(0,str(HERE/'deps'))
# Import libomp users in this order on this Mac; torch-first crashes LightGBM.
import lightgbm,catboost
import torch
assert torch.backends.mps.is_available();torch.set_num_threads(2)
prepared=json.loads((HERE/'PREPARE.json').read_text());dest=HERE/'payload_mps';base=ROOT/'astra/arena_20261101/baseline';restored=[]
for r in prepared['only_source_edits']:
 p=dest/r['file'];orig=base/r['file'];h=hashlib.sha256(p.read_bytes()).hexdigest();assert h in (r['before'],r['after']);assert hashlib.sha256(orig.read_bytes()).hexdigest()==r['before']
 if h!=r['before']:shutil.copyfile(orig,p)
 restored.append(r['file'])
folder=dest/'model/ko/model';spec=json.loads((folder/'encoder_views.json').read_text())
for name,h in spec['source_hashes'].items():assert hashlib.sha256((folder/'src'/name).read_bytes()).hexdigest()==h,(name,'source integrity')
sys.path.insert(0,str(folder))
from src.encoder_pair import EncoderPairScorer
from src.neural_pair import PairScorer
from src.llm_translit import Transliterator
assert EncoderPairScorer.__init__.__defaults__==('cuda',);assert PairScorer.__init__.__defaults__==('cuda',)
EncoderPairScorer.__init__.__defaults__=('mps',);PairScorer.__init__.__defaults__=('mps',)
assert Transliterator.__init__.__defaults__[0]=='cuda';Transliterator.__init__.__defaults__=('mps',Transliterator.__init__.__defaults__[1])
from src.encoder_views import EncoderViewsEnsemble
import numpy as np
original_scores=EncoderViewsEnsemble.score_records
def finite_scores(self,*args,**kwargs):
 result=original_scores(self,*args,**kwargs)
 assert all(np.isfinite(a).all() for a in result),'Nonfinite MPS reference scores'
 return result
EncoderViewsEnsemble.score_records=finite_scores
sys.path.insert(0,str(ROOT/'astra/arena_highgain_20261101/package'));from trace_hooks import load_entry
m=load_entry(dest/'model/ko/script.py');test=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();t=time.time()
(HERE/'RUNTIME_DEVICE_MANIFEST.json').write_text(json.dumps({'source_zip_sha256':prepared['source_zip_sha256'],'all_encoder_source_hashes_verified':True,'restored_original_sources':restored,'only_runtime_change':'three constructor defaults CUDA -> MPS; original BF16 and inference rules retained','finite_score_validation':True,'reference_scope':'MPS reference, not proven CUDA numerical identity; no L4 timing'},indent=2))
m.run(test_path=test,catalog_path=ROOT/'data/open/vessel_catalog.csv',output_path=out)
report={'seconds':time.time()-t,'input':str(test),'output':str(out),'torch':torch.__version__,'device':'MPS','mps_allocated_bytes':torch.mps.current_allocated_memory()};out.with_suffix('.timing.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
