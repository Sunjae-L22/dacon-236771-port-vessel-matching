from pathlib import Path
import sys,os,json
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;LOCAL=ROOT/'astra/new_direction_20261101/local_cvote';MODEL=LOCAL/'payload_mps/model/ko/model/llm'
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTORCH_ENABLE_MPS_FALLBACK='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
sys.path.insert(0,str(LOCAL/'deps'))
import lightgbm,catboost
import torch,transformers
torch.set_num_threads(2)
assert transformers.__version__=='4.57.6'
from transformers import AutoTokenizer,AutoModelForCausalLM
SYSTEM=json.loads((HERE/'PROTOCOL.json').read_text())['system_prompt']
def messages(q,a,b):return [{'role':'system','content':SYSTEM},{'role':'user','content':f'한글 이름: {q}\nA: {a}\nB: {b}'}]
def key(q,a,b):return json.dumps([q,a,b],ensure_ascii=False,separators=(',',':'))
