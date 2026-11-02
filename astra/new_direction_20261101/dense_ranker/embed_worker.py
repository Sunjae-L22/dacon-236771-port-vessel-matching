from pathlib import Path
import sys,json,os
sys.dont_write_bytecode=True
os.environ.update(TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np,torch
from transformers import AutoTokenizer,AutoModel
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'astra/new_direction_20261101/dense'));from probe import encode
torch.set_num_threads(2);names=json.loads(Path(sys.argv[1]).read_text());tok=AutoTokenizer.from_pretrained(ROOT/'astra/semantic_prior/weights',local_files_only=True);model=AutoModel.from_pretrained(ROOT/'astra/semantic_prior/weights',local_files_only=True).to('mps').eval();v=encode(model,tok,['query: '+s for s in names],'mps');np.save(sys.argv[2],v)
