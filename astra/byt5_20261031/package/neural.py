"""ByT5 conditional scoring. No catalog labels, metadata, or other test rows enter scoring."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json,random,unicodedata,math,contextlib
from pathlib import Path
import numpy as np
import torch
from transformers import T5ForConditionalGeneration,T5Config
from transformers.modeling_outputs import BaseModelOutput

SOURCE_LIMIT=160
TARGET_LIMIT=96
MODEL='google/byt5-small'
SEED=20261031

def canonical(s):return ' '.join(unicodedata.normalize('NFC',str(s)).upper().split())
def source(s):return 'ko2en: '+unicodedata.normalize('NFC',str(s))
def ids(s):return [b+3 for b in s.encode('utf-8')]+[1]
def eligible(name,names):return len(ids(source(name)))<=SOURCE_LIMIT and all(len(ids(canonical(n)))<=TARGET_LIMIT for n in names)
def batch_tokens(texts,device,labels=False):
    seq=[ids(s) for s in texts];n=max(map(len,seq));pad=-100 if labels else 0
    t=torch.full((len(seq),n),pad,dtype=torch.long,device=device)
    for i,s in enumerate(seq):t[i,:len(s)]=torch.tensor(s,device=device)
    return t
def setup(seed=SEED):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    return 'cuda' if torch.cuda.is_available() else 'cpu'
def amp(device):return torch.autocast('cuda',dtype=torch.bfloat16) if device=='cuda' and torch.cuda.is_bf16_supported() else contextlib.nullcontext()
def load_model(path,device,local=True):
    model=T5ForConditionalGeneration.from_pretrained(str(path),local_files_only=local,attn_implementation='eager')
    model.to(device);model.eval();return model

@torch.inference_mode()
def score_names(model,name,names,device,microbatch=16):
    if not eligible(name,names):raise ValueError('overlength: must fall back to110, never truncate')
    inp=batch_tokens([source(name)],device);mask=inp.ne(0)
    scores=[]
    with amp(device):
        enc=model.get_encoder()(input_ids=inp,attention_mask=mask,return_dict=True).last_hidden_state
        for lo in range(0,len(names),microbatch):
            target=batch_tokens([canonical(n) for n in names[lo:lo+microbatch]],device,True);b=len(target)
            enc_b=BaseModelOutput(last_hidden_state=enc.expand(b,-1,-1).contiguous())
            out=model(encoder_outputs=enc_b,attention_mask=mask.expand(b,-1),labels=target,decoder_attention_mask=target.ne(-100),use_cache=False)
            lp=out.logits.float().log_softmax(-1).gather(-1,target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            values=(lp*target.ne(-100)).sum(1)/target.ne(-100).sum(1)
            if not torch.isfinite(values).all():raise RuntimeError('non-finite neural score')
            scores.extend(values.cpu().tolist())
    return scores

def decision(packet,scores):
    cs=packet['candidates'];old=packet['baseline']
    if packet['gap']>=4 or scores is None:return old
    if len(cs)!=len(scores) or not all(math.isfinite(s) for s in scores):raise ValueError('invalid scores')
    top=max(scores)
    k=min(range(len(cs)),key=lambda i:(-(cs[i]['score']+2*(scores[i]-top)),cs[i]['index']))
    return cs[k]['id']

def edit(s,rng):
    pos=[i for i,c in enumerate(s) if '가'<=c<='힣']
    if len(pos)<2:return s
    op=rng.choices(['delete','duplicate','swap'],[.35,.35,.30])[0]
    if op=='swap':
        pairs=[i for i in pos if i+1<len(s) and '가'<=s[i+1]<='힣' and s[i]!=s[i+1]]
        if pairs:
            i=rng.choice(pairs);return s[:i]+s[i+1]+s[i]+s[i+2:]
        op='duplicate'
    i=rng.choice(pos)
    return s[:i]+s[i+1:] if op=='delete' else s[:i+1]+s[i]+s[i+1:]

class Sampler:
    def __init__(self,rows,seed=SEED):
        self.rng=random.Random(seed);self.groups={};self.dropped=0
        for r in rows:
            if not eligible(r['name'],[r['target']]):self.dropped+=1;continue
            self.groups.setdefault(r['source'],{}).setdefault(r['vessel'],[]).append(r)
        assert set(self.groups)=={'native','llm'}
        self.keys={s:sorted(g) for s,g in self.groups.items()}
    def batch(self,n):
        rows=[]
        for i in range(n):
            s='native' if i%2==0 else 'llm';v=self.rng.choice(self.keys[s]);r=self.rng.choice(self.groups[s][v]);name=r['name']
            p=self.rng.random();edits=0 if p<.35 else 1 if p<.8 else 2
            for _ in range(edits):name=edit(name,self.rng)
            if not eligible(name,[r['target']]):name=r['name']
            rows.append((source(name),canonical(r['target'])))
        return rows

def read_jsonl(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def save_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(p)
