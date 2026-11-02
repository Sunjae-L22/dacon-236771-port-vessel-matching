"""One frozen-protocol same-script contrastive adaptation; no public inference."""
from pathlib import Path
import os,sys,json,csv,hashlib,random,time,datetime,collections
sys.dont_write_bytecode=True
os.environ.update(TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModel

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DENSE=HERE.parent/'dense'
WEIGHTS=ROOT/'astra/semantic_prior/weights'
sys.path.insert(0,str(ROOT/'astra/research_20261101'))
from channel_model import read,normalize,hangul
CUTOFF=datetime.datetime(2026,10,31,22,tzinfo=datetime.timezone.utc).timestamp()

def checktime():
 if time.time()>=CUTOFF:raise RuntimeError('07:00 KST cutoff reached')

def noise(text,op,number):
 idx=[i for i,c in enumerate(text) if '가'<=c<='힣']
 if op=='swap':idx=[i for i in idx if i+1<len(text) and '가'<=text[i+1]<='힣' and text[i]!=text[i+1]]
 if not idx:return text
 i=idx[number%len(idx)]
 if op=='delete':return text[:i]+text[i+1:]
 if op=='dup':return text[:i+1]+text[i]+text[i+1:]
 return text[:i]+text[i+1]+text[i]+text[i+2:]

def encode(model,tok,texts,batch=80):
 vv=[]
 with torch.inference_mode():
  for start in range(0,len(texts),batch):
   checktime();x=tok(['query: '+s for s in texts[start:start+batch]],padding=True,truncation=True,max_length=48,return_tensors='pt').to('mps')
   h=model(**x).last_hidden_state;mask=x['attention_mask'].unsqueeze(-1);v=torch.nn.functional.normalize((h*mask).sum(1)/mask.sum(1),p=2,dim=1);vv.append(v.cpu().numpy())
 return np.concatenate(vv)

def ranked(E,Q,owners,ids,rows,label):
 out=[];order=np.arange(len(ids));pos={v:i for i,v in enumerate(ids)}
 # Identical numerical pipeline for baseline and adapted query vectors.
 proto=torch.from_numpy(E).to('mps')
 with torch.inference_mode():
  for start in range(0,len(rows),32):
   checktime();aa=(torch.from_numpy(Q[start:start+32]).to('mps')@proto.T).cpu().numpy()
   for j,alias_scores in enumerate(aa):
    r=rows[start+j];scores=np.full(len(ids),-np.inf,dtype=np.float32);np.maximum.at(scores,owners,alias_scores);ci=pos[r['vessel_id']]
    rank=int(np.count_nonzero((scores>scores[ci])|((scores==scores[ci])&(order<ci))))+1
    top=np.lexsort((order,-scores))[:100]
    out.append({**r,'rank':rank,'top1':ids[top[0]],'top100':[ids[k] for k in top]})
 del proto
 (HERE/(label+'_rows.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out))
 return out

def summarize(base,adapted):
 summary={}
 for fold in ('all',0,1):
  for variant in ('all','clean','delete','dup','swap'):
   pairs=[(a,b) for a,b in zip(base,adapted) if (fold=='all' or a['fold']==fold) and (variant=='all' or a['variant']==variant)]
   entry={'rows':len(pairs)}
   for k in (1,10,100):
    entry[str(k)]={'frozen_hits':sum(a['rank']<=k for a,b in pairs),'adapted_hits':sum(b['rank']<=k for a,b in pairs),'rescue':sum(a['rank']>k and b['rank']<=k for a,b in pairs),'harm':sum(a['rank']<=k and b['rank']>k for a,b in pairs)}
   summary[str(fold)+'/'+variant]=entry
 return summary

def main():
 start=time.monotonic();checktime();torch.set_num_threads(2);torch.manual_seed(112);rng=random.Random(112)
 if not torch.backends.mps.is_available():raise RuntimeError('This bounded experiment requires available MPS')
 cat=read(ROOT/'data/open/vessel_catalog.csv');ids=[r['vessel_id'] for r in cat];assert ids==json.loads((DENSE/'catalog_ids.json').read_text());pos={v:i for i,v in enumerate(ids)}
 table={r['vessel_name']:r for r in read(ROOT/'shared/kocat/out/catalog_ko.csv')};aliases=[];owners=[]
 for r in cat:
  a=table.get(r['vessel_name'],{})
  for text in sorted({normalize(a.get(f'ko{i}','')) for i in range(4) if hangul(a.get(f'ko{i}',''))}):aliases.append(text);owners.append(pos[r['vessel_id']])
 owners=np.array(owners,dtype=np.int64);E=np.load(DENSE/'alias_vectors.npy');assert E.shape==(len(aliases),384)
 queries=json.loads((DENSE/'native_queries.json').read_text());hold=[r for r in queries if r['fold'] in (0,1)];fit=[r for r in queries if r['fold'] in (2,3,4)];holdids={r['vessel_id'] for r in hold}
 text_owners=collections.defaultdict(set)
 for a,o in zip(aliases,owners):text_owners[a].add(int(o))
 eligible=np.array([len(text_owners[a])==1 for a in aliases]);keep=np.flatnonzero(eligible);owner_proto=collections.defaultdict(list)
 for new,old in enumerate(keep):owner_proto[int(owners[old])].append(new)
 pairs=[(aliases[old],int(owners[old])) for old in keep if ids[owners[old]] not in holdids]
 real=[(r['name'],pos[r['vessel_id']]) for r in fit if pos[r['vessel_id']] in owner_proto and len(text_owners.get(r['name'],set()))<=1]
 assert not holdids & {ids[o] for a,o in pairs+real}
 evalrows=[]
 for r in hold:
  for variant in ('clean','delete','dup','swap'):
   seed=int(hashlib.sha256(('metric-eval-v1:'+r['vessel_id']+':'+variant).encode()).hexdigest()[:16],16)
   q=r['name'] if variant=='clean' else noise(r['name'],variant,seed)
   evalrows.append({**r,'name':q,'native_name':r['name'],'variant':variant})
 cfg={'method':'same-script query adaptation to frozen Korean catalog-alias prototypes; multi-positive contrastive loss','seed':112,'steps':1000,'batch':80,'catalog_per_batch':64,'real_per_batch':16,'lr':2e-5,'temperature':.03,'weight_decay':.01,'grad_clip':1.,'dropout':'disabled using model.eval() throughout adaptation','trainable':'encoder last two layers only','catalog_noise':'50 percent unchanged, otherwise one uniform delete/dup/swap on Hangul positions; no valid position means unchanged','real_noise':'none; authentic fit query text; remove queries that equal a multi-owner normalized catalog alias','real_fit_folds':[2,3,4],'evaluation_folds':[0,1],'evaluation_alias_positive_training_excluded':True,'eval_query_count':len(hold),'eval_rows':len(evalrows),'real_fit_count':len(real),'catalog_training_pairs':len(pairs),'training_prototypes':len(keep),'ambiguous_alias_prototypes_excluded':int((~eligible).sum()),'ambiguity_policy':'remove all normalized strings owned by more than one vessel from training prototypes and catalog samples','positives':'all retained aliases of target vessel, numerator logsumexp; other alias vectors denominator logsumexp','evaluation_prototypes':'original full alias_vectors.npy, all official catalog IDs, no training-ID exclusions, stable descending vessel-max cosine then original catalog index','model_selection':'none; compare original public query encoder and final step1000 once','limitations':['Same KoCat table as training and target prototypes; style biased','These human folds were previously evaluated in other experiments; not untouched confirmation','Catalog alias of evaluation vessels excluded as training positives; remain known frozen prototypes/negatives','Training fit query text can overlap other vessels catalog aliases; frozen catalog supervision is not independent natural text','Not C_vote inference or Public prediction'],'cutoff_kst':'2026-11-01T07:00:00+09:00'}
 (HERE/'PROTOCOL.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n');(HERE/'eval_queries.json').write_text(json.dumps(evalrows,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'protocol_frozen':True,'real':len(real),'catalog':len(pairs),'train_prototypes':len(keep),'evaluation_rows':len(evalrows)}),flush=True)
 tok=AutoTokenizer.from_pretrained(WEIGHTS,local_files_only=True);model=AutoModel.from_pretrained(WEIGHTS,local_files_only=True).to('mps').eval()
 Qbase=encode(model,tok,[r['name'] for r in evalrows]);np.save(HERE/'frozen_query_vectors.npy',Qbase)
 for p in model.parameters():p.requires_grad=False
 for layer in model.encoder.layer[-2:]:
  for p in layer.parameters():p.requires_grad=True
 trainparams=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(trainparams,lr=2e-5,weight_decay=.01);P=torch.from_numpy(E[keep]).to('mps');history=[]
 for step in range(1000):
  checktime();batch=[rng.choice(pairs) for _ in range(64)]+[rng.choice(real) for _ in range(16)];texts=[]
  for j,(s,y) in enumerate(batch):
   if j<64 and rng.random()<.5:s=noise(s,rng.choice(('delete','dup','swap')),rng.randrange(2**31))
   texts.append('query: '+s)
  x=tok(texts,padding=True,truncation=True,max_length=48,return_tensors='pt').to('mps');h=model(**x).last_hidden_state;mask=x['attention_mask'].unsqueeze(-1);q=torch.nn.functional.normalize((h*mask).sum(1)/mask.sum(1),p=2,dim=1);logits=q@P.T/.03
  positives=[owner_proto[y] for s,y in batch];width=max(map(len,positives));ii=torch.tensor([z+[0]*(width-len(z)) for z in positives],device='mps');valid=torch.tensor([[True]*len(z)+[False]*(width-len(z)) for z in positives],device='mps');pp=logits.gather(1,ii).masked_fill(~valid,-torch.inf)
  loss=(torch.logsumexp(logits,dim=1)-torch.logsumexp(pp,dim=1)).mean();opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(trainparams,1.);opt.step()
  if (step+1)%50==0:
   record={'step':step+1,'loss':float(loss.detach().cpu()),'seconds':time.monotonic()-start};history.append(record);print(json.dumps(record),flush=True)
 del P,opt,logits,q,h,x,loss,pp
 last=len(model.encoder.layer);prefixes=tuple(f'encoder.layer.{i}.' for i in (last-2,last-1));torch.save({k:v.detach().cpu() for k,v in model.state_dict().items() if k.startswith(prefixes)},HERE/'query_last2_final.pt')
 (HERE/'TRAIN_HISTORY.json').write_text(json.dumps(history,indent=2)+'\n');Qnew=encode(model,tok,[r['name'] for r in evalrows]);np.save(HERE/'adapted_query_vectors.npy',Qnew);del model
 base=ranked(E,Qbase,owners,ids,evalrows,'frozen');adapted=ranked(E,Qnew,owners,ids,evalrows,'adapted');results={'comparisons':summarize(base,adapted),'seconds':time.monotonic()-start,'clean_top1_harm':sum(a['variant']=='clean' and a['rank']==1 and b['rank']!=1 for a,b in zip(base,adapted)),'scope':'name-only full-catalog retrieval, not C_vote and not Public','steps_completed':1000}
 results['clean_harm_zero']=results['clean_top1_harm']==0
 (HERE/'RESULTS.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n');print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
