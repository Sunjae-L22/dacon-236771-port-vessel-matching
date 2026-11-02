"""Frozen forward domain-PMI screening with fixed-shape numerical QA first."""
from pathlib import Path
import sys,os,json,time,datetime,hashlib
sys.dont_write_bytecode=True
from scorer import ForwardPMIScorer,torch,transformers,normalize_name,MODEL,PROMPT
import numpy as np
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent;START=time.monotonic()
DEADLINE=datetime.datetime(2026,10,31,21,35,tzinfo=datetime.timezone.utc).timestamp()
def checktime():
 if time.time()>=DEADLINE or time.monotonic()-START>45*60:raise RuntimeError('06:35/45min watchdog; no further inference')
def predict_logits(s,inputs):
 with torch.inference_mode():return s.model(**inputs,use_cache=False,logits_to_keep=33).logits.float()

def qa(s,pairs):
 # Eight distinct pairs span prior/conditional text and candidate lengths.
 chosen=[pairs[i] for i in np.linspace(0,len(pairs)-1,8,dtype=int)]
 forward=s.score_many_pairs(chosen);reverse=s.score_many_pairs(chosen[::-1])[::-1]
 singles=np.concatenate([s.score_many_pairs([p]) for p in chosen])
 target,query=chosen[0];inputs,tokens,_=s.encode(target,[query]);labels=torch.cat([inputs['input_ids'][:,1:],torch.full((8,1),s.eos_token_id,dtype=torch.long,device=s.device)],dim=1)
 logits=predict_logits(s,inputs);assert labels[0,-len(tokens)-1:].tolist()==tokens+[128009]
 nll=F.cross_entropy(logits[0].reshape(-1,logits.shape[-1]),labels[0,-33:].reshape(-1),reduction='none')
 independent=-nll[-len(tokens)-1:].detach().cpu().numpy();ce=np.array([independent[:-1].sum(dtype=np.float64),independent[:-1].mean(dtype=np.float64),float(independent[-1]),len(tokens)])
 masked={k:v.clone() for k,v in inputs.items()};masked['input_ids'][masked['attention_mask']==0]=(s.tokenizer.pad_token_id+17)%s.model.config.vocab_size
 alternative=predict_logits(s,masked);padding=float((logits[0,-len(tokens)-1:]-alternative[0,-len(tokens)-1:]).abs().max().item())
 result={'batch_size':8,'pad_length':512,'dtype':str(s.dtype),'turn_end':s.eos_token_id,'shifted_CE_max_abs':float(np.max(np.abs(ce-forward[0]))),'reversed_batch_max_abs':float(np.max(np.abs(forward-reverse))),'singleton_repeated_to8_max_abs':float(np.max(np.abs(forward-singles))),'masked_padding_logit_max_abs':padding,'tolerance':.001,'qa_pairs':chosen,'torch':torch.__version__,'transformers':transformers.__version__}
 assert max(result[k] for k in ('shifted_CE_max_abs','reversed_batch_max_abs','singleton_repeated_to8_max_abs','masked_padding_logit_max_abs'))<=.001,result
 result['pass']=True;(HERE/'QA.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'qa':result},ensure_ascii=False),flush=True)
 del inputs,masked,logits,alternative,nll;torch.mps.empty_cache();return result

def evaluate(scores,pairs):
 lookup={tuple(p):v for p,v in zip(pairs,scores)};rows=json.loads((HERE/'SELECTED_ROWS.json').read_text());out=[]
 for r in rows:
  x=dict(r);x['pmi_prediction']=x['conditional_prediction']=r['baseline'];x['primary_changed']=False
  if r['hard']:
   cond=np.asarray([lookup[(normalize_name(c['name']),r['name'])] for c in r['candidates']]);prior=np.asarray([lookup[(normalize_name(c['name']),'N/A')] for c in r['candidates']]);assert np.array_equal(cond[:,3],prior[:,3])
   delta=(cond[:,0]+cond[:,2]-prior[:,0]-prior[:,2])/(cond[:,3]+1);uncorrected=(cond[:,0]+cond[:,2])/(cond[:,3]+1);base=np.asarray([c['score'] for c in r['candidates']]);j=int(np.argmax(base+delta));u=int(np.argmax(base+uncorrected));x.update(pmi_prediction=r['candidates'][j]['id'],conditional_prediction=r['candidates'][u]['id'],pmi_candidate_rank=j+1,conditional_candidate_rank=u+1,delta=delta.tolist(),conditional_mean=uncorrected.tolist(),base_scores=base.tolist());x['primary_changed']=x['pmi_prediction']!=r['baseline']
  out.append(x)
 report={'method':'Frozen lambda1 forward domainPMI','folds':{},'public_score_estimate':None,'labels_or_IDs_used_by_model':False,'selection':'48 ID-hash vessels, genuine native names, fixed one noise variant each','seconds':time.monotonic()-START}
 for fold in (2,4):
  d={}
  for name,field in [('primary_pmi','pmi_prediction'),('diagnostic_uncorrected','conditional_prediction')]:
   d[name]={}
   for typ in ('clean','noise','all'):
    rr=[r for r in out if r['fold']==fold and (typ=='all' or r['variant'].startswith('clean')==(typ=='clean'))];res=[r for r in rr if r['baseline']!=r['target'] and r[field]==r['target']];harm=[r for r in rr if r['baseline']==r['target'] and r[field]!=r['target']]
    d[name][typ]={'rows':len(rr),'vessels':len({r['vessel'] for r in rr}),'hard_rows':sum(r['hard'] for r in rr),'baseline_correct':sum(r['baseline']==r['target'] for r in rr),'new_correct':sum(r[field]==r['target'] for r in rr),'changed':sum(r[field]!=r['baseline'] for r in rr),'rescue':len(res),'harm':len(harm),'net':len(res)-len(harm),'target_absent_top20':sum(r['target'] not in [c['id'] for c in r['candidates']] for r in rr)}
  d['primary_pass']=d['primary_pmi']['clean']['harm']==0 and d['primary_pmi']['noise']['net']>0;report['folds'][str(fold)]=d
 report['primary_both_pass']=all(d['primary_pass'] for d in report['folds'].values());report['protected_changes']=sum(not r['hard'] and r['primary_changed'] for r in out);assert report['protected_changes']==0
 (HERE/'RESULTS_ROWS.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out));(HERE/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)

def main():
 checktime();pairs=[tuple(p) for p in json.loads((HERE/'MODEL_PAIRS.json').read_text())];assert pairs==sorted(set(pairs))
 print(json.dumps({'stage':'load_model','pairs':len(pairs),'model':str(MODEL)}),flush=True);s=ForwardPMIScorer();checktime();qa(s,pairs);values=[]
 for lo in range(0,len(pairs),8):
  checktime();values.append(s.score_many_pairs(pairs[lo:lo+8]))
  if lo%64==0:print(json.dumps({'scored':min(lo+8,len(pairs)),'total':len(pairs),'seconds':time.monotonic()-START}),flush=True)
 values=np.concatenate(values);assert len(values)==len(pairs) and np.isfinite(values).all()
 tmp=HERE/'PAIR_SCORES.tmp.npy';np.save(tmp,values);tmp.replace(HERE/'PAIR_SCORES.npy')
 (HERE/'SCORE_MANIFEST.json').write_text(json.dumps({'pairs':len(pairs),'input_pairs_sha256':hashlib.sha256((HERE/'MODEL_PAIRS.json').read_bytes()).hexdigest(),'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),'weights_sha256_from_source':'Existing immutable C_vote weight provenance; no model training or new download','seconds':time.monotonic()-START,'dtype':str(s.dtype),'batch':8,'pad':512,'turn_end':s.eos_token_id},indent=2)+'\n')
 del s;torch.mps.empty_cache();evaluate(values,pairs)

if __name__=='__main__':main()
