from pathlib import Path
import os,sys,json,csv,time,hashlib,ast,random
sys.dont_write_bytecode=True
os.environ.update(TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np,torch
from transformers import AutoTokenizer,AutoModel
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'astra/research_20261101'))
from channel_model import read,normalize,hangul

def encode(model,tok,texts,device,batch=128):
 out=[];t=time.time()
 with torch.inference_mode():
  for start in range(0,len(texts),batch):
   b=tok(texts[start:start+batch],padding=True,truncation=True,max_length=48,return_tensors='pt').to(device);h=model(**b).last_hidden_state;mask=b['attention_mask'].unsqueeze(-1);v=(h*mask).sum(1)/mask.sum(1);v=torch.nn.functional.normalize(v,p=2,dim=1);out.append(v.cpu().numpy())
   if start%(batch*50)==0:print('encoded',start,'/',len(texts),'seconds',round(time.time()-t,1),flush=True)
 return np.concatenate(out)

def main():
 torch.set_num_threads(2);device='mps' if torch.backends.mps.is_available() else 'cpu';tok=AutoTokenizer.from_pretrained(ROOT/'astra/semantic_prior/weights',local_files_only=True);model=AutoModel.from_pretrained(ROOT/'astra/semantic_prior/weights',local_files_only=True).to(device).eval()
 cat=read(ROOT/'data/open/vessel_catalog.csv');names=[r['vessel_name'] for r in cat];ids=[r['vessel_id'] for r in cat];byid={r['vessel_id']:r for r in cat};train=read(ROOT/'data/open/train.csv');shots={normalize(x[0]) for x in json.loads((ROOT/'shared/kocat/fewshot_pairs.json').read_text())}
 tree=ast.parse((ROOT/'astra/arena_20261101/baseline/model/ko/model/src/llm_translit.py').read_text())
 for n in tree.body:
  if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='FEW_SHOT' for x in n.targets):shots.update(normalize(x[1]) for x in ast.literal_eval(n.value))
 rows=[];seen=set()
 for r in train:
  vid=r['target_vessel_id'];q=normalize(r['vessel_name']);key=(vid,q)
  if not hangul(q) or key in seen or normalize(byid[vid]['vessel_name']) in shots:continue
  seen.add(key);fold=int(hashlib.sha256(('dense-v1:'+vid).encode()).hexdigest()[:8],16)%5
  rows.append(dict(vessel_id=vid,name=q,fold=fold))
 (HERE/'native_queries.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));(HERE/'catalog_ids.json').write_text(json.dumps(ids))
 f=HERE/'english_vectors.npy'
 if not f.exists():np.save(f,encode(model,tok,['passage: '+n for n in names],device))
 E=np.load(f);Q=encode(model,tok,['query: '+r['name'] for r in rows],device);np.save(HERE/'native_query_vectors.npy',Q)
 # All catalog entities are legitimate prototypes. No fitting to human evaluation query labels.
 pos={v:i for i,v in enumerate(ids)};rr=[]
 for i,r in enumerate(rows):
  sc=E@Q[i];rank=int(np.count_nonzero(sc>sc[pos[r['vessel_id']]]))+1;top=np.argsort(-sc)[:100];rr.append(dict(r,rank=rank,top100=[ids[j] for j in top]))
 (HERE/'base_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr));report={'model':'frozen public multilingual-e5-small','device':device,'rows':len(rows),'ships':len(set(r['vessel_id'] for r in rows)),'scope':'actual human Korean queries; catalog English prototypes only; fewshot excluded; not deployed C_vote','recall':{str(k):sum(r['rank']<=k for r in rr)/len(rr) for k in (1,10,50,100)},'max_rank':max(r['rank'] for r in rr)};(HERE/'BASE_RESULTS.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
