from pathlib import Path
import os,sys,json,time,datetime,threading,hashlib,csv,math
sys.dont_write_bytecode=True
from runtime import ROOT,HERE,MODEL,torch,transformers,AutoTokenizer,AutoModelForCausalLM,messages
from decision import decide
import numpy as np
START=time.monotonic();DEADLINE=datetime.datetime(2026,10,31,21,45,tzinfo=datetime.timezone.utc).timestamp();RATES=[]
def checktime():
 if time.time()>=DEADLINE:raise RuntimeError('06:45 hard stop')
def hard_stop():
 (HERE/'TIMEOUT.json').write_text(json.dumps({'stage':'hard process stop','time':time.time(),'reason':'06:45 deadline'}));os._exit(124)
TIMER=threading.Timer(max(.01,DEADLINE-time.time()),hard_stop);TIMER.daemon=True;TIMER.start()

class Judge:
 def __init__(self):
  checktime();self.tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,padding_side='left');self.tok.pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id
  self.ids={x:self.tok.encode(x,add_special_tokens=False)[0] for x in ('A','B')};assert self.ids=={'A':32,'B':33}
  self.pad=json.loads((HERE/'MANIFEST.json').read_text())['fixed_pad_length'];assert self.pad==256
  self.model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('mps').eval();self.model.config.use_cache=False;self.encoded={}
 def inputs(self,keys):
  checktime();seq=[]
  for k in keys:
   if k not in self.encoded:self.encoded[k]=self.tok.apply_chat_template(messages(*json.loads(k)),tokenize=True,add_generation_prompt=True,enable_thinking=False)
   seq.append(self.encoded[k])
  assert 1<=len(seq)<=8 and max(map(len,seq))<=self.pad
  seq+= [seq[-1]]*(8-len(seq));ids=torch.full((8,self.pad),self.tok.pad_token_id,dtype=torch.long,device='mps');mask=torch.zeros_like(ids)
  for i,s in enumerate(seq):ids[i,-len(s):]=torch.tensor(s,dtype=torch.long,device='mps');mask[i,-len(s):]=1
  return {'input_ids':ids,'attention_mask':mask,'position_ids':(mask.cumsum(-1)-1).clamp_min(0)}
 @torch.inference_mode()
 def logits(self,inputs):
  checktime();z=self.model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1,:].float();assert torch.isfinite(z).all();return z
 def describe(self,z):
  v,idx=torch.topk(z,2,dim=-1);vals=v.cpu().numpy();tops=idx.cpu().numpy();a=z[:,32].cpu().numpy();b=z[:,33].cpu().numpy();out=[]
  for i in range(len(z)):
   token=int(tops[i,0]);valid=token in (32,33) and vals[i,0]>vals[i,1]
   out.append({'choice':('A' if token==32 else 'B') if valid else None,'top_token_id':token,'top_token':self.tok.decode([token]),'top_logit':float(vals[i,0]),'second_logit':float(vals[i,1]),'logit_A':float(a[i]),'logit_B':float(b[i]),'format_valid':bool(valid)})
  return out
 def score(self,keys):return self.describe(self.logits(self.inputs(keys)))[:len(keys)]

def qa(j,keys):
 selected=[keys[i] for i in np.linspace(0,len(keys)-1,8,dtype=int)];x=j.inputs(selected);a=j.logits(x);b=j.logits(j.inputs(selected[::-1])).flip(0);c=torch.stack([j.logits(j.inputs([k]))[0] for k in selected]);pert={k:v.clone() for k,v in x.items()};pert['input_ids'][pert['attention_mask']==0]=(j.tok.pad_token_id+17)%j.model.config.vocab_size;d=j.logits(pert)
 reports={};da=j.describe(a)
 for name,z in [('reverse',b),('singleton_repeat8',c),('masked_padding',d)]:
  dz=j.describe(z);reports[name]={'max_logit_abs_difference':float((a-z).abs().max().item()),'full_vocab_argmax_identical':bool(torch.equal(a.argmax(-1),z.argmax(-1))),'valid_A_B_choices_identical':all(u['choice']==v['choice'] for u,v in zip(da,dz))}
 assert all(v['full_vocab_argmax_identical'] and v['valid_A_B_choices_identical'] for v in reports.values()),reports
 report={'sampled_prompts':8,'fixed_batch':8,'pad':j.pad,'dtype':'BF16','torch':torch.__version__,'transformers':transformers.__version__,'reports':reports,'QA_answers':da,'pass':True,'scope':'Sampled numerical/prediction stability, not proof of all inputs or CUDA equivalence'};(HERE/'QA.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'QA':report},ensure_ascii=False),flush=True);del a,b,c,d,x,pert;torch.mps.empty_cache()

def getkeys(folder):return json.loads((folder/'PROMPT_KEYS.json').read_text())
def score_set(j,folder,tag,cache):
 needed=[k for k in getkeys(folder) if k not in cache];checktime();start=time.monotonic()
 with (HERE/'PROMPT_SCORES.jsonl').open('a',encoding='utf-8') as f:
  for lo in range(0,len(needed),8):
   checktime();chunk=needed[lo:lo+8];t=time.monotonic();values=j.score(chunk);elapsed=time.monotonic()-t;RATES.append(elapsed/len(chunk))
   for k,v in zip(chunk,values):cache[k]=v;f.write(json.dumps({'key':k,**v},ensure_ascii=False)+'\n')
   f.flush()
   if lo%64==0:print(json.dumps({'tag':tag,'scored':min(lo+8,len(needed)),'new_prompts':len(needed),'seconds':time.monotonic()-START}),flush=True)
 print(json.dumps({'tag_complete':tag,'new_prompts':len(needed),'seconds':time.monotonic()-start}),flush=True)
 return {'needed':len(needed),'seconds':time.monotonic()-start}

def evaluate(folder,tag,cache,aux=False):
 rows=json.loads((folder/'ROWS_WITHOUT_TRUTH.json').read_text());out=[]
 for r in rows:
  d=decide(r,cache);out.append({**r,**d,'changed':d['prediction']!=r['cvote']})
 assert [r['sample_id'] for r in out]==[r['sample_id'] for r in rows]
 rev={r['sample_id']:decide(r,cache) for r in rows[::-1]};assert all(rev[r['sample_id']]==decide(r,cache) for r in rows)
 assert all(not r['changed'] or r['hard'] for r in out)
 (HERE/f'{tag}_PREDICTIONS_BEFORE_TRUTH.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out))
 if aux:truth=json.loads((folder/'TRUTH.json').read_text())
 else:
  truth_path=ROOT/'astra/selection_20261101/confirm/eval/truth.csv' if tag=='actual90' else ROOT/'astra/new_direction_20261101/local_cvote/eval/truth.csv'
  with truth_path.open(encoding='utf-8-sig') as f:truth={r['sample_id']:{'target':r['target_vessel_id'],'variant':r['variant']} for r in csv.DictReader(f)}
 assert set(truth)=={r['sample_id'] for r in out}
 for r in out:r.update(target=truth[r['sample_id']]['target'],variant=truth[r['sample_id']]['variant'])
 report={'tag':tag,'comparator':'OOF112' if aux else 'actual C_vote MPS reference','independent_public_validation':False,'rows':len(out),'hard_rows':sum(r['hard'] for r in out),'metrics':{},'protected_changes':0,'row_reversal_mismatches':0,'condorcet_winners':sum(r['condorcet_winner'] is not None for r in out),'decisive_pairs':sum(r['decisive_pairs'] for r in out),'draw_pairs':sum(r['draw_pairs'] for r in out),'invalid_pairs':sum(r['invalid_pairs'] for r in out),'order_disagreement_pairs':sum(r['order_disagreement_pairs'] for r in out),'full_vocab_format_valid_prompts':sum(cache[k]['format_valid'] for k in getkeys(folder)),'ordered_prompts':len(getkeys(folder)),'mechanism':{}}
 for typ in ('clean','noise','all'):
  rr=[r for r in out if typ=='all' or r['variant'].startswith('clean')==(typ=='clean')];res=[r for r in rr if r['cvote']!=r['target'] and r['prediction']==r['target']];harm=[r for r in rr if r['cvote']==r['target'] and r['prediction']!=r['target']]
  report['metrics'][typ]={'rows':len(rr),'vessels':len({r['target'] for r in rr}),'hard_rows':sum(r['hard'] for r in rr),'baseline_correct':sum(r['cvote']==r['target'] for r in rr),'new_correct':sum(r['prediction']==r['target'] for r in rr),'changed':sum(r['changed'] for r in rr),'rescue':len(res),'harm':len(harm),'net':len(res)-len(harm),'rescue_vessels':len({r['target'] for r in res}),'harm_vessels':len({r['target'] for r in harm})}
 for kind in ('restore112','other_selection','all3_original_voters_wrong'):
  rr=[r for r in out if r['changed'] and ((r['prediction']==r['baseline112']) if kind=='restore112' else (r['prediction']!=r['baseline112']) if kind=='other_selection' else (not aux and all(r['target']!=r.get(k) for k in ('baseline112','hsw','v20'))))]
  report['mechanism'][kind]={'changed':len(rr),'rescue':sum(r['prediction']==r['target'] and r['cvote']!=r['target'] for r in rr),'harm':sum(r['cvote']==r['target'] and r['prediction']!=r['target'] for r in rr),'rows':[r['sample_id'] for r in rr],'winner112_ranks':[r.get('original112_ranks',{}).get(r['prediction']) for r in rr]}
 report['fixed_condition_pass']=report['metrics']['clean']['harm']==0 and report['metrics']['noise' if aux else 'all']['net']>0
 (HERE/f'{tag}_ROWS.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out));(HERE/f'{tag}_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True);return report

def budget(folder,cache,tag):
 needed=[k for k in getkeys(folder) if k not in cache];rate=max(RATES) if RATES else .9164;seconds=len(needed)*rate*1.25+120;fits=time.time()+seconds<DEADLINE
 r={'tag':tag,'unseen_prompts':len(needed),'max_measured_seconds_per_prompt':rate,'safety_multiplier':1.25,'extra_safety_seconds':120,'estimated_seconds':seconds,'fits_0645':fits};(HERE/f'{tag}_BUDGET.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);return fits

def main():
 checktime();assert not (HERE/'PROMPT_SCORES.jsonl').exists(),'Do not accidentally mix runs';manifest=json.loads((HERE/'MANIFEST.json').read_text());assert manifest['fits_0645_conservative']
 print(json.dumps({'stage':'load','model':str(MODEL),'ordered_prompts':manifest['unique_ordered_prompts']}),flush=True);j=Judge();checktime();qa(j,getkeys(HERE));cache={};score_set(j,HERE,'actual90',cache);actual=evaluate(HERE,'actual90',cache);aux=None
 folder=HERE/'validation'
 if folder.exists() and budget(folder,cache,'native174'):
  score_set(j,folder,'native174',cache);aux=evaluate(folder,'native174',cache,aux=True)
 ext=HERE/'extended';extended=None
 if actual['fixed_condition_pass'] and aux is not None and aux['fixed_condition_pass'] and ext.exists() and budget(ext,cache,'development400'):
  score_set(j,ext,'development400',cache);extended=evaluate(ext,'development400',cache)
 state={'completed_at_kst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),'seconds':time.monotonic()-START,'actual90_complete':True,'native174_complete':aux is not None,'development400_complete':extended is not None,'actual90_condition':actual['fixed_condition_pass'],'native174_condition':aux['fixed_condition_pass'] if aux else None,'unique_prompts_cached':len(cache),'new_zip':False,'submissions':0};(HERE/'FINAL_STATE.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True);del j;torch.mps.empty_cache();TIMER.cancel()

if __name__=='__main__':main()
