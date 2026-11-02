"""Tokenizer-only audit. No torch/model/GPU load."""
from pathlib import Path
import os,sys,ast,csv,json,re,unicodedata
os.environ.update(USE_TORCH='0',USE_TF='0',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'astra/new_direction_20261101/local_cvote/deps'))
from transformers import AutoTokenizer

def normalize(s):
    s=unicodedata.normalize('NFKC',str(s)).upper()
    return ' '.join(''.join(c if c.isalnum() else ' ' for c in s).split())

def main():
    src=ROOT/'external/ksh/src/llm_translit.py';values={}
    for n in ast.parse(src.read_text()).body:
        if isinstance(n,ast.Assign):
            for t in n.targets:
                if isinstance(t,ast.Name) and t.id in ('SYSTEM_PROMPT','FEW_SHOT'):
                    values[t.id]=ast.literal_eval(n.value)
    prompt=values['SYSTEM_PROMPT']+' Use UPPERCASE Latin letters. Render only the name words present in the input; do not invent missing words.'
    tok=AutoTokenizer.from_pretrained(ROOT/'astra/new_direction_20261101/local_cvote/payload_mps/model/ko/model/llm',local_files_only=True,padding_side='left')
    def messages(q):
        m=[{'role':'system','content':prompt}]
        for k,e in values['FEW_SHOT']:m.extend([{'role':'user','content':k},{'role':'assistant','content':normalize(e)}])
        return m+[{'role':'user','content':q}]
    catalog=list(csv.DictReader((ROOT/'data/open/vessel_catalog.csv').open(encoding='utf-8-sig')))
    candidates=list(dict.fromkeys(['X','A','1','NO 2','OCEAN ACE','N A']+[normalize(r['vessel_name']) for r in catalog[::397]]))
    queries=['TEST','N/A','모닝 메나드','에스티아이 롬바드','가'*48]
    mismatches=[];ends=set();maximum=0;long=0;counts=[]
    for q in queries:
        m=messages(q);prefix=tok.apply_chat_template(m,tokenize=True,add_generation_prompt=True,enable_thinking=False)
        counts.append({'query':q,'prefix_tokens':len(prefix)})
        for candidate in candidates:
            target=tok.encode(candidate,add_special_tokens=False)
            complete=tok.apply_chat_template(m+[{'role':'assistant','content':candidate}],tokenize=True,add_generation_prompt=False,enable_thinking=False)
            good=complete[:len(prefix)]==prefix and complete[len(prefix):len(prefix)+len(target)]==target
            if not good:mismatches.append([q,candidate])
            if len(complete)>len(prefix)+len(target):ends.add(complete[len(prefix)+len(target)])
            maximum=max(maximum,len(prefix)+len(target));long+=len(prefix)+len(target)>512
    assert not mismatches
    assert ends=={128009}
    assert 'torch' not in sys.modules
    result={'model_loaded':False,'torch_imported':False,'gpu_used':False,'queries_checked':len(queries),'candidates_checked':len(candidates),
            'full_chat_vs_separate_target_mismatches':len(mismatches),'assistant_turn_end_ids':sorted(ends),
            'tokenizer_eos_token_id':tok.eos_token_id,'prefix_counts':counts,'maximum_prompt_plus_target_tokens':maximum,
            'above_512_probe_pairs':long,'scope':'Sampled real catalog targets and deliberate long artificial query; not the final96 query/token preflight'}
    Path(__file__).with_name('TOKENIZER_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
