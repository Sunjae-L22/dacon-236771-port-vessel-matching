"""Author: Astra. Public pretrained weights + official metadata strings only.

This is a model-generated similarity table, not a hand-written maritime mapping.
No external dataset, vessel database, test input, or LLM-generated mapping is used.
"""
from pathlib import Path
import os,json,hashlib
os.environ['USE_TF']='0';os.environ['USE_FLAX']='0';os.environ['HF_HUB_DISABLE_XET']='1'
import torch
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer,AutoModel
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'astra/semantic_prior';OUT.mkdir(exist_ok=True)
MODEL='intfloat/multilingual-e5-small'
REV='614241f622f53c4eeff9890bdc4f31cfecc418b3'

def main():
    path=snapshot_download(MODEL,revision=REV,local_dir=OUT/'weights',allow_patterns=['config.json','model.safetensors','pytorch_model.bin','tokenizer.json','tokenizer_config.json','sentencepiece.bpe.model','special_tokens_map.json'])
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=AutoModel.from_pretrained(path,local_files_only=True).eval()
    torch.set_num_threads(2)
    tr=pd.read_csv(ROOT/'data/open/train.csv');cat=pd.read_csv(ROOT/'data/open/vessel_catalog.csv')
    source={k:sorted(tr[k].dropna().astype(str).unique()) for k in ['flag_name','vessel_type']}
    targets={k:sorted(cat[k].dropna().astype(str).unique()) for k in source}
    labels=sorted(set(sum(source.values(),[])+sum(targets.values(),[])))
    vectors=[]
    with torch.inference_mode():
        for start in range(0,len(labels),32):
            batch=tok(['query: '+s for s in labels[start:start+32]],padding=True,truncation=False,return_tensors='pt')
            h=model(**batch).last_hidden_state
            h=h.masked_fill(~batch['attention_mask'][...,None].bool(),0.)
            emb=h.sum(1)/batch['attention_mask'].sum(1)[:,None]
            vectors.extend(torch.nn.functional.normalize(emb,p=2,dim=1).numpy())
    vec=dict(zip(labels,vectors)); rows=[]
    for field in source:
        for q in source[field]:
            scores=np.array([vec[q]@vec[c] for c in targets[field]])
            order=np.argsort(-scores,kind='stable')
            for rank,j in enumerate(order,1): rows.append(dict(field=field,query=q,catalog=targets[field][j],cosine=float(scores[j]),rank=rank))
    d=pd.DataFrame(rows);d.to_csv(OUT/'similarities.csv',index=False)
    d[d['rank']<=5].to_csv(OUT/'top5.csv',index=False)
    manifest=dict(model=MODEL,revision=REV,inputs=['data/open/train.csv: flag_name, vessel_type','data/open/vessel_catalog.csv: flag_name, vessel_type'],method='query prefix, masked mean pooling, L2-normalized cosine',manual_mapping=False,adopted=False,
                  file_sha256=hashlib.sha256((OUT/'similarities.csv').read_bytes()).hexdigest())
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print(d[(d.field=='vessel_type')&(d['rank']<=3)].to_string(index=False),flush=True)

if __name__=='__main__':main()
