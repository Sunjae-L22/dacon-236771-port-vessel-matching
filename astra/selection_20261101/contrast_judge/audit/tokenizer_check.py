import os,sys,json,hashlib
from pathlib import Path
os.environ.update(USE_TORCH='0',USE_TF='0',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;BASE=HERE.parent
sys.path.insert(0,str(ROOT/'astra/new_direction_20261101/local_cvote/deps'))
from transformers import AutoTokenizer
system=json.loads((BASE/'PROTOCOL.json').read_text())['system_prompt'];tok=AutoTokenizer.from_pretrained(ROOT/'astra/new_direction_20261101/local_cvote/payload_mps/model/ko/model/llm',local_files_only=True,padding_side='left')
labels={s:tok.encode(s,add_special_tokens=False) for s in 'AB'};assert labels=={'A':[32],'B':[33]};report={}
for name,folder in [('primary90',BASE),('validation174',BASE/'validation')]:
 lengths=[];ends=set();keys=json.loads((folder/'PROMPT_KEYS.json').read_text());pad=json.loads((folder/'MANIFEST.json').read_text())['fixed_pad_length']
 for k in keys:
  q,a,b=json.loads(k);m=[{'role':'system','content':system},{'role':'user','content':f'한글 이름: {q}\nA: {a}\nB: {b}'}];prefix=tok.apply_chat_template(m,tokenize=True,add_generation_prompt=True,enable_thinking=False);lengths.append(len(prefix))
  for s in 'AB':
   complete=tok.apply_chat_template(m+[{'role':'assistant','content':s}],tokenize=True,add_generation_prompt=False,enable_thinking=False)
   assert complete[:len(prefix)]==prefix and complete[len(prefix):len(prefix)+1]==labels[s];ends.add(complete[len(prefix)+1])
 assert max(lengths)<=pad and ends=={128009}
 report[name]={'ordered_prompts':len(keys),'assistant_prefix_target_checks':len(keys)*2,'min_prompt_tokens':min(lengths),'max_prompt_tokens':max(lengths),'pad':pad,'min_headroom':pad-max(lengths),'prefix_or_target_mismatches':0,'assistant_end_tokens':list(ends)}
assert 'torch' not in sys.modules
report.update({'torch_imported':False,'model_loaded':False,'gpu_used':False,'A_B_ids':labels,'protocol_sha256':hashlib.sha256((BASE/'PROTOCOL.json').read_bytes()).hexdigest()})
(HERE/'TOKENIZER_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
