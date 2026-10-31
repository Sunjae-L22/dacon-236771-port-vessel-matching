"""Name-only ordered character/Jamo cross encoder. No pretrained weights."""
import re
import unicodedata
import torch
from torch import nn

CHO = ['g','kk','n','d','tt','r','m','b','pp','s','ss','','j','jj','ch','k','t','p','h']
JUNG = ['a','ae','ya','yae','eo','e','yeo','ye','o','wa','wae','oe','yo','u','wo','we','wi','yu','eu','ui','i']
JONG = ['','k','k','k','n','n','n','t','l','l','l','l','l','l','l','l','m','p','p','s','ss','ng','j','ch','k','t','p','h']

def normalize(value):
    return ''.join(c for c in unicodedata.normalize('NFKC', str(value)).lower() if c.isalnum())

def romanize(value):
    out=[]
    for c in unicodedata.normalize('NFKC', str(value)).lower():
        n=ord(c)-0xAC00
        if 0<=n<11172:
            out.append(CHO[n//588]+JUNG[n%588//28]+JONG[n%28])
        else: out.append(c)
    return ' '.join(''.join(out).split())

def fields(query, candidate):
    return [' '.join(unicodedata.normalize('NFKD', str(query)).lower().split()), romanize(query),
            ' '.join(unicodedata.normalize('NFKD', candidate['name']).lower().split()), candidate['phon']]

class Tokenizer:
    def __init__(self, config):
        self.vocab=config['vocab']; self.budgets=config.get('field_budgets',[64,24,40,28]); self.max_length=config.get('max_length',160)
    def encode(self, query, candidate):
        ids=[1]; segments=[0]
        for part,(text,limit) in enumerate(zip(fields(query,candidate),self.budgets)):
            chars=list(text)
            if len(chars)>limit:
                # Preserve both leading and trailing name evidence when truncation is unavoidable.
                chars=chars[:(limit+1)//2]+chars[-limit//2:]
            ids.extend(self.vocab.get(c,3) for c in chars);segments.extend([part]*len(chars))
            if part<3:ids.append(2);segments.append(part)
        assert len(ids)<=self.max_length
        return ids,segments
    def batch(self, pairs, device):
        encoded=[self.encode(q,c) for q,c in pairs];length=max(len(x[0]) for x in encoded)
        ids=torch.zeros((len(encoded),length),dtype=torch.long);segments=torch.zeros_like(ids)
        for i,(tok,seg) in enumerate(encoded):
            ids[i,:len(tok)]=torch.tensor(tok);segments[i,:len(seg)]=torch.tensor(seg)
        return ids.to(device),segments.to(device)

class CrossEncoder(nn.Module):
    def __init__(self, config):
        super().__init__();d=config.get('d_model',128)
        self.char=nn.Embedding(max(config['vocab'].values())+1,d,padding_idx=0)
        self.segment=nn.Embedding(4,d);self.position=nn.Embedding(config.get('max_length',160),d)
        layer=nn.TransformerEncoderLayer(d,config.get('heads',4),dim_feedforward=config.get('feedforward',384),dropout=config.get('dropout',.1),activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,config.get('layers',2),enable_nested_tensor=False)
        self.head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,64),nn.GELU(),nn.Linear(64,1))
    def forward(self, ids, segments):
        position=torch.arange(ids.shape[1],device=ids.device)[None,:]
        x=self.char(ids)+self.segment(segments)+self.position(position)
        x=self.encoder(x,src_key_padding_mask=ids.eq(0))
        return self.head(x[:,0]).squeeze(-1)

@torch.inference_mode()
def score_pairs(model, tokenizer, pairs, device='cpu', batch_size=256):
    if hasattr(torch.backends,'mha'):torch.backends.mha.set_fastpath_enabled(False)
    model.eval();out=[]
    for start in range(0,len(pairs),batch_size):
        ids,segments=tokenizer.batch(pairs[start:start+batch_size],device)
        out.extend(model(ids,segments).float().cpu().tolist())
    return out

def load_checkpoint(path, device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model=CrossEncoder(saved['config']);model.load_state_dict(saved['model']);model.to(device).eval()
    return model,Tokenizer(saved['config']),saved
