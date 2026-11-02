from pathlib import Path
import os,sys,json,time,random,hashlib,datetime
sys.dont_write_bytecode=True
os.environ.update(TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np,torch
from transformers import AutoModel,AutoTokenizer
from probe import ROOT,HERE,encode,read,normalize,hangul
CUTOFF=datetime.datetime(2026,10,31,22,tzinfo=datetime.timezone.utc).timestamp()

def corrupt(s,rng):
 u=rng.random()
 if u<.5:return s
 ops=[rng.choice(('delete','dup','swap'))]
 if u>.9:ops.append(rng.choice(('delete','dup','swap')))
 for op in ops:
  if len(s)<2:break
  i=rng.randrange(len(s)-(op=='swap'))
  if op=='delete':s=s[:i]+s[i+1:]
  elif op=='dup':s=s[:i+1]+s[i]+s[i+1:]
  else:s=s[:i]+s[i+1]+s[i]+s[i+2:]
 return s

def main():
 torch.set_num_threads(2);torch.manual_seed(112);rng=random.Random(112);device='mps';start=time.time()
 cfg={'hypothesis':'catalog-wide dense query encoder supervised against frozen English catalog prototypes','steps':1000,'batch':96,'lr':2e-5,'temperature':.03,'trainable':'last two Bert encoder layers only','real_query_fit_folds':[2,3,4],'development_fold':1,'test_fold':0,'split':'sha256 dense-v1 + vesselID mod5','real_query_share':.2,'catalog_alias_share':.8,'catalog_aliases_include_eval_entity_ids':'allowed known-catalog supervision, NOT heldout catalog entities; no eval human query in training','noise':'50%clean,40%one uniform event,10%two events; delete/dup/swap','evaluation':'all realK query folds reported separately; no model selection on test','cutoff':'2026-11-01T07:00:00+09:00','seed':112}
 (HERE/'TRAIN_PROTOCOL.json').write_text(json.dumps(cfg,indent=2))
 cat=read(ROOT/'data/open/vessel_catalog.csv');table={r['vessel_name']:r for r in read(ROOT/'shared/kocat/out/catalog_ko.csv')};queries=json.loads((HERE/'native_queries.json').read_text());fit=[r for r in queries if r['fold'] in (2,3,4)]
 # Collapse duplicate canonical-name prototype classes in training; do not penalize identical names as negatives.
 keys={};labels={};indices=[]
 for i,r in enumerate(cat):
  k=normalize(r['vessel_name'])
  if k not in keys:keys[k]=len(keys);indices.append(i)
  labels[r['vessel_id']]=keys[k]
 E=np.load(HERE/'english_vectors.npy');prototypes=torch.from_numpy(E[indices]).to(device)
 pairs=[]
 for r in cat:
  a=table.get(r['vessel_name'],{});vs=sorted({normalize(a.get(f'ko{i}','')) for i in range(4) if hangul(a.get(f'ko{i}',''))})
  for v in vs:pairs.append((v,labels[r['vessel_id']]))
 real=[(r['name'],labels[r['vessel_id']]) for r in fit]
 cfg.update(catalog_alias_pairs=len(pairs),prototype_classes=len(keys),real_train_query_pairs=len(real),real_test_queries=sum(r['fold']==0 for r in queries),real_development_queries=sum(r['fold']==1 for r in queries));(HERE/'TRAIN_PROTOCOL.json').write_text(json.dumps(cfg,indent=2))
 tok=AutoTokenizer.from_pretrained(ROOT/'astra/semantic_prior/weights',local_files_only=True);model=AutoModel.from_pretrained(ROOT/'astra/semantic_prior/weights',local_files_only=True).to(device)
 for p in model.parameters():p.requires_grad=False
 for layer in model.encoder.layer[-2:]:
  for p in layer.parameters():p.requires_grad=True
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=.01);history=[]
 model.eval()
 for step in range(cfg['steps']):
  if time.time()>=CUTOFF:raise RuntimeError('07:00 cutoff reached')
  batch=[rng.choice(real if rng.random()<.2 else pairs) for _ in range(cfg['batch'])];texts=['query: '+corrupt(s,rng) for s,y in batch];target=torch.tensor([y for s,y in batch],device=device)
  x=tok(texts,padding=True,truncation=True,max_length=48,return_tensors='pt').to(device);out=model(**x).last_hidden_state;mask=x['attention_mask'].unsqueeze(-1);q=torch.nn.functional.normalize((out*mask).sum(1)/mask.sum(1),p=2,dim=1);logits=(q@prototypes.T)/cfg['temperature'];loss=torch.nn.functional.cross_entropy(logits,target);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);opt.step()
  if (step+1)%50==0:
   rec={'step':step+1,'loss':float(loss.detach().cpu()),'seconds':time.time()-start};history.append(rec);print(json.dumps(rec),flush=True)
 # Save only changed layers, keeping public original read-only.
 weights={k:v.detach().cpu() for k,v in model.state_dict().items() if k.startswith(('encoder.layer.10.','encoder.layer.11.'))};torch.save(weights,HERE/'query_last2.pt');(HERE/'TRAIN_HISTORY.json').write_text(json.dumps(history,indent=2))
 model.eval();Q=encode(model,tok,['query: '+r['name'] for r in queries],device);np.save(HERE/'trained_native_vectors.npy',Q);pos={r['vessel_id']:i for i,r in enumerate(cat)};rr=[]
 for i,r in enumerate(queries):
  sc=E@Q[i];rank=int(np.count_nonzero(sc>sc[pos[r['vessel_id']]]))+1;top=np.argsort(-sc)[:100];rr.append(dict(r,rank=rank,top100=[cat[j]['vessel_id'] for j in top]))
 (HERE/'trained_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr));summary={}
 for fold in range(5):
  xs=[r for r in rr if r['fold']==fold];summary[str(fold)]={'rows':len(xs),'recall':{str(k):sum(r['rank']<=k for r in xs)/len(xs) for k in (1,10,50,100)}}
 result={'seconds':time.time()-start,'device':device,'folds':summary,'c_vote_predictions_used':False};(HERE/'TRAINED_RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
