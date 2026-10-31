"""Catalog-constrained byte beam search, independent for each query."""
from functools import lru_cache
from neural import canonical,ids,source,batch_tokens,amp,torch,SOURCE_LIMIT,TARGET_LIMIT

class CatalogTrie:
    def __init__(self,records,excluded):
        self.root={};self.by_name={};self.nodes=1
        for r in records:
            if r['vessel_id'] in excluded:continue
            name=canonical(r['vessel_name'])
            if not name or len(ids(name))>TARGET_LIMIT:continue
            self.by_name.setdefault(name,[]).append(r['vessel_id'])
        for name in sorted(self.by_name):
            node=self.root
            for token in ids(name):
                if token not in node:node[token]={};self.nodes+=1
                node=node[token]
        self.allowed=lru_cache(maxsize=300000)(self._allowed)
    def _allowed(self,prefix):
        node=self.root
        for t in prefix:
            if t==1:return (1,)
            if t not in node:raise ValueError(f'invalid constrained prefix {prefix}')
            node=node[t]
        return tuple(sorted(node)) or (1,)
    def callback(self,batch_id,tokens):
        assert batch_id==0
        prefix=tokens.tolist();assert prefix[0]==0
        return list(self.allowed(tuple(prefix[1:])))

def decode(sequence):
    tokens=sequence.tolist() if hasattr(sequence,'tolist') else list(sequence)
    assert tokens[0]==0
    body=[]
    for token in tokens[1:]:
        if token==1:break
        if token==0:continue
        if not 3<=token<=258:raise ValueError('non-byte generated token')
        body.append(token-3)
    return bytes(body).decode('utf-8')

@torch.inference_mode()
def search(model,name,trie,device,beams=16):
    if len(ids(source(name)))>SOURCE_LIMIT:return []
    x=batch_tokens([source(name)],device)
    with amp(device):
        generated=model.generate(input_ids=x,attention_mask=x.ne(0),max_new_tokens=TARGET_LIMIT,num_beams=beams,num_return_sequences=beams,do_sample=False,length_penalty=0.0,early_stopping=True,use_cache=True,renormalize_logits=False,prefix_allowed_tokens_fn=trie.callback)
    names=list(dict.fromkeys(decode(seq) for seq in generated))
    assert all(n in trie.by_name for n in names),names
    return [dict(name=n,ids=sorted(trie.by_name[n])) for n in names]
