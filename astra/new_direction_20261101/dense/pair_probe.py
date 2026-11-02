"""Frozen E5 same-script alias feature vs heldout112 recipe proxy; no Public inference."""
from pathlib import Path
import os,sys,json,time,hashlib
sys.dont_write_bytecode=True
os.environ.update(TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np,torch
from transformers import AutoTokenizer,AutoModel
from probe import ROOT,HERE,encode,read,normalize,hangul

def summarize(xs,key):
 return {group:{'rows':len(rs),'rescue':sum(r['baseline']!=r['target'] and r[key]==r['target'] for r in rs),'harm':sum(r['baseline']==r['target'] and r[key]!=r['target'] for r in rs),'changed':sum(r['baseline']!=r[key] for r in rs)} for group,rs in [('clean',[r for r in xs if r['variant'].startswith('clean')]),('noise',[r for r in xs if not r['variant'].startswith('clean')])]}

def main():
 torch.set_num_threads(2);t=time.time();cfg={'methods':[0,5,10,20,40],'feature':'max cosine over original72B aliases using frozen publicE5; score=112_raw_score+lambda*cosine','gate':'original110 gap<4, original112 candidate top100 only','development_fold':2,'confirmation_fold':4,'selection':'max noise rescue-harm among clean-harm0; ties smallerlambda; baseline0 eligible','caveat':'reused historical112-recipe proxy folds, not untouched dataset or deployed6-KC ensemble or C_vote; exploratory screen only','trainable_e5_used':False};(HERE/'PAIR_PROTOCOL.json').write_text(json.dumps(cfg,indent=2))
 cat=read(ROOT/'data/open/vessel_catalog.csv');table={r['vessel_name']:r for r in read(ROOT/'shared/kocat/out/catalog_ko.csv')};owner={};j=0
 for r in cat:
  a=table.get(r['vessel_name'],{});ss=sorted({normalize(a.get(f'ko{i}','')) for i in range(4) if hangul(a.get(f'ko{i}',''))});owner[r['vessel_id']]=list(range(j,j+len(ss)));j+=len(ss)
 E=np.load(HERE/'alias_vectors.npy');assert len(E)==j
 packets={fold:[json.loads(s) for s in (ROOT/f'astra/byt5_base112_20261031/fold{fold}/packets112.jsonl').read_text().splitlines() if s] for fold in (2,4)}
 names=list(dict.fromkeys(normalize(r['name']) for rr in packets.values() for r in rr));tok=AutoTokenizer.from_pretrained(ROOT/'astra/semantic_prior/weights',local_files_only=True);model=AutoModel.from_pretrained(ROOT/'astra/semantic_prior/weights',local_files_only=True).to('mps').eval();V=encode(model,tok,['query: '+s for s in names],'mps');qmap={s:V[i] for i,s in enumerate(names)}
 outputs={};summaries={}
 for fold in (2,4):
  rr=[]
  for p in packets[fold]:
   if not p['primary']:continue
   x={k:p[k] for k in ('query_id','name','variant','target','baseline')};cand=p['candidates'];q=qmap[normalize(p['name'])];es=np.array([float(np.max(E[owner[c['id']]]@q)) if owner.get(c['id']) else 0. for c in cand]);scores=np.array([c['score'] for c in cand]);order=np.array([c['index'] for c in cand])
   for w in cfg['methods']:
    pred=p['baseline']
    if p['gate110_gap']<4 and w>0:
     sc=scores+w*es;k=np.lexsort((order,-sc))[0];pred=cand[k]['id']
    x[str(w)]=pred
   x['gate110_gap']=p['gate110_gap'];x['dense_candidates']=[dict(id=c['id'],dense=float(z)) for c,z in zip(cand,es)];rr.append(x)
  outputs[fold]=rr;summaries[fold]={str(w):summarize(rr,str(w)) for w in cfg['methods']}
  if fold==2:
   eligible=[w for w in cfg['methods'] if summaries[fold][str(w)]['clean']['harm']==0];chosen=max(eligible,key=lambda w:(summaries[fold][str(w)]['noise']['rescue']-summaries[fold][str(w)]['noise']['harm'],-w));(HERE/'PAIR_DEV_SELECTION.json').write_text(json.dumps({'chosen_lambda':chosen,'dev_metrics':summaries[fold]},indent=2));print('DEV_SELECTED',chosen,flush=True)
 result={'seconds':time.time()-t,'chosen_lambda':chosen,'dev_all':summaries[2],'confirmation_selected':summaries[4][str(chosen)],'not_c_vote':True}
 (HERE/'PAIR_RESULTS.json').write_text(json.dumps(result,indent=2));
 for fold,rr in outputs.items():(HERE/f'pair_fold{fold}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr))
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
