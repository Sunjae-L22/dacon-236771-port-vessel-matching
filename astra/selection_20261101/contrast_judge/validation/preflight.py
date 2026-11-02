"""Tokenizer-only exact primary prompt/schema preflight; no torch or model load."""
import os,sys,json,csv,hashlib,datetime
from pathlib import Path
os.environ.update(USE_TORCH='0',USE_TF='0',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;PARENT=HERE.parent
sys.path.insert(0,str(ROOT/'astra/new_direction_20261101/local_cvote/deps'))
from transformers import AutoTokenizer
SYSTEM=json.loads((PARENT/'PROTOCOL.json').read_text())['system_prompt']
def messages(q,a,b):return [{'role':'system','content':SYSTEM},{'role':'user','content':f'한글 이름: {q}\nA: {a}\nB: {b}'}]
def key(q,a,b):return json.dumps([q,a,b],ensure_ascii=False,separators=(',',':'))
with (ROOT/'data/open/vessel_catalog.csv').open(encoding='utf-8-sig') as f:catalog=list(csv.DictReader(f))
order={r['vessel_id']:i for i,r in enumerate(catalog)}
rows=[];prompts=set()
for r in json.loads((HERE/'MODEL_ROWS.json').read_text()):
 names={c['id']:c['name'] for c in r['candidates']};pool=sorted(names,key=order.get) if r['hard'] else []
 out={'sample_id':r['query_id'],'name':r['name'],'hard':r['hard'],'gap110':r['gap110'],'baseline112':r['incumbent'],'cvote':r['incumbent'],'hsw':None,'v20':None,'pool':pool,'pool_names':[names[v] for v in pool],'original112_ranks':{c['id']:i+1 for i,c in enumerate(r['candidates'])},'comparisons':[],'fold':r['fold'],'variant':r['variant'],'comparator_is_oof112_not_cvote':True}
 for i,a in enumerate(pool):
  for b in pool[i+1:]:
   ab=key(r['name'],names[a],names[b]);ba=key(r['name'],names[b],names[a]);prompts.update((ab,ba));out['comparisons'].append({'a':a,'b':b,'ab':ab,'ba':ba})
 rows.append(out)
tok=AutoTokenizer.from_pretrained(ROOT/'astra/new_direction_20261101/local_cvote/payload_mps/model/ko/model/llm',local_files_only=True,padding_side='left');labels={s:tok.encode(s,add_special_tokens=False) for s in ['A','B']};assert all(len(v)==1 for v in labels.values());lens=[]
for k in sorted(prompts):
 q,a,b=json.loads(k);m=messages(q,a,b);prefix=tok.apply_chat_template(m,tokenize=True,add_generation_prompt=True,enable_thinking=False);lens.append(len(prefix))
 for s in ['A','B']:
  full=tok.apply_chat_template(m+[{'role':'assistant','content':s}],tokenize=True,add_generation_prompt=False,enable_thinking=False);assert full[:len(prefix)]==prefix and full[len(prefix):len(prefix)+1]==labels[s]
assert 'torch' not in sys.modules
primary=json.loads((PARENT/'MANIFEST.json').read_text());pad=primary['fixed_pad_length'];assert max(lens)<=pad
manifest=json.loads((HERE/'MANIFEST.json').read_text());manifest.update({'exact_primary_prompt_sha256':hashlib.sha256(SYSTEM.encode()).hexdigest(),'primary_protocol_sha256':hashlib.sha256((PARENT/'PROTOCOL.json').read_bytes()).hexdigest(),'max_prompt_tokens':max(lens),'min_prompt_tokens':min(lens),'fixed_pad_length':pad,'minimum_headroom_tokens':pad-max(lens),'A_B_token_ids':{s:v[0] for s,v in labels.items()},'all_assistant_prefix_boundary_checks_passed':True,'tokenizer_only_no_torch':True,'primary_schema_adapted':True,'model_loaded':False,'model_weights_modified':False})
rate=json.loads((ROOT/'astra/selection_20261101/likelihood/FOLLOWUP_TIME_ESTIMATE.json').read_text())['steady_state_seconds_per_pair_max'];manifest['conservative512_seconds_25pct_plus120loadQA']=len(prompts)*rate*1.25+120;manifest['scaled_pad_seconds_25pct_plus120loadQA']=len(prompts)*rate*(pad/512)*1.25+120
for name,obj in [('ROWS_WITHOUT_TRUTH.json',rows),('PROMPT_KEYS.json',sorted(prompts)),('MANIFEST.json',manifest)]:
 (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
