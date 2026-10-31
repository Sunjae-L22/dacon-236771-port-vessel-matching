"""Author: Astra. Public pretrained translation; inputs are train metadata labels only."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,re,time
import pandas as pd
import torch
from huggingface_hub import HfApi,snapshot_download
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from rapidfuzz import process,fuzz
ROOT=Path(__file__).resolve().parents[1]

def clean(s):return re.sub('[^a-z0-9]','',str(s).lower())

def main():
    out=ROOT/'astra/translate_metadata87';out.mkdir(exist_ok=True)
    model_id='Helsinki-NLP/opus-mt-ko-en';revision=HfApi().model_info(model_id).sha
    path=snapshot_download(model_id,revision=revision,local_dir=out/'weights',allow_patterns=['config.json','generation_config.json','tokenizer_config.json','vocab.json','source.spm','target.spm','pytorch_model.bin'])
    torch.set_num_threads(1)
    tok=AutoTokenizer.from_pretrained(path);model=AutoModelForSeq2SeqLM.from_pretrained(path).eval()
    train=pd.read_csv(ROOT/'data/open/train.csv');cat=pd.read_csv(ROOT/'data/open/vessel_catalog.csv');rows=[];start=time.time()
    for field in ['flag_name','vessel_type']:
        queries=sorted(train[field].dropna().astype(str).unique());labels=sorted(cat[field].dropna().astype(str).unique());norm=[clean(s) for s in labels]
        for q in queries:
            inputs=tok([q],return_tensors='pt',padding=True)
            with torch.inference_mode():ids=model.generate(**inputs,max_new_tokens=32,num_beams=4,do_sample=False)
            translated=tok.batch_decode(ids,skip_special_tokens=True)[0]
            matches=process.extract(clean(translated),norm,scorer=fuzz.ratio,limit=3)
            rows.append(dict(field=field,query=q,translated=translated,top1=labels[matches[0][2]],score=matches[0][1],top3=json.dumps([(labels[i],float(score)) for _,score,i in matches],ensure_ascii=False)))
        print('TRANSLATED',field,len(queries),round(time.time()-start),flush=True)
    d=pd.DataFrame(rows);d.to_csv(out/'predictions.csv',index=False)
    result=[];cat=cat.set_index('vessel_id')
    for field in ['flag_name','vessel_type']:
        pred=d[d.field.eq(field)].set_index('query').top1
        v=train[['target_vessel_id',field]].drop_duplicates();v['pred']=v[field].map(pred);v['truth']=v.target_vessel_id.map(cat[field])
        result.append(dict(field=field,observed_label_agreement=float(v.pred.eq(v.truth).mean()),meaning='diagnostic agreement only; not heldout vessel accuracy'))
    (out/'manifest.json').write_text(json.dumps(dict(model_id=model_id,revision=revision,inputs='unique flag_name/vessel_type strings from train.csv only',catalog_use='fuzzy-match generated translations against allowed catalog metadata labels',manual_corrections=False,metrics=result,seconds=time.time()-start),indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
