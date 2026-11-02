"""Fixed-shape forward scorer; no source package or original weights modified."""
from pathlib import Path
import ast,unicodedata,sys,os
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
LOCAL=ROOT/'astra/new_direction_20261101/local_cvote'
sys.path.insert(0,str(LOCAL/'deps'))
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTORCH_ENABLE_MPS_FALLBACK='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
# Keep the proven local libomp import order.
import lightgbm,catboost
import torch,transformers
assert transformers.__version__=='4.57.6'
torch.set_num_threads(2)
from reverse_local import ReverseNameScorer
MODEL=LOCAL/'payload_mps/model/ko/model/llm'

def literal(name):
 tree=ast.parse((ROOT/'external/ksh/src/llm_translit.py').read_text())
 return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))

FEW_SHOT=literal('FEW_SHOT')
PROMPT=literal('SYSTEM_PROMPT')+' Use UPPERCASE Latin letters. Render only the name words present in the input; do not invent missing words.'
def normalize_name(value):
 value=unicodedata.normalize('NFKC',str(value)).upper()
 return ' '.join(''.join(c if c.isalnum() else ' ' for c in value).split())
def messages(query):
 out=[{'role':'system','content':PROMPT}]
 for ko,en in FEW_SHOT:out.extend([{'role':'user','content':ko},{'role':'assistant','content':normalize_name(en)}])
 return out+[{'role':'user','content':query}]

class ForwardPMIScorer(ReverseNameScorer):
 def __init__(self):
  super().__init__(model_dir=MODEL,device='mps',batch_size=8,pad_length=512,dtype=torch.bfloat16,attn_implementation='sdpa')
  prefix=self.tokenizer.apply_chat_template(messages('TEST'),tokenize=True,add_generation_prompt=True,enable_thinking=False)
  complete=self.tokenizer.apply_chat_template(messages('TEST')+[{'role':'assistant','content':'X'}],tokenize=True,add_generation_prompt=False,enable_thinking=False)
  target=self.tokenizer.encode('X',add_special_tokens=False)
  assert complete[:len(prefix)]==prefix and complete[len(prefix):len(prefix)+len(target)]==target
  self.eos_token_id=int(complete[len(prefix)+len(target)])
  assert self.eos_token_id==128009
 def prefix_ids(self,query):
  return self.tokenizer.apply_chat_template(messages(query),tokenize=True,add_generation_prompt=True,enable_thinking=False)
