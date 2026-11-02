"""Observe unchanged v20 inference. Does not read labels or alter scores."""
from pathlib import Path
import json,time,hashlib
import numpy as np

def install(folder):
 from src.encoder_views import EncoderViewsEnsemble
 from src.llm_translit import Transliterator
 prior_generate=Transliterator.generate;progress={'n':0};started=time.time()
 def traced_generate(self,*args,**kwargs):
  value=prior_generate(self,*args,**kwargs);progress['n']+=1
  if progress['n']%10==0:print('TRANSLITERATED',progress['n'],'unique names in',round(time.time()-started),'seconds',flush=True)
  return value
 Transliterator.generate=traced_generate
 import src.ensemble as E
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 original=EncoderViewsEnsemble.score_records
 def capture(self,rows,signals,blocks):
  raw=[];base=[];views=[];oldmember=E.member_score;oldbase=self.baseline.score_groups;oldviews=[]
  def member(kind,model,X):
   x=oldmember(kind,model,X);raw.append(np.asarray(x).copy());return x
  def baseline(bs):
   x=oldbase(bs);base.extend([np.asarray(v).copy() for v in x]);return x
  E.member_score=member;self.baseline.score_groups=baseline
  for scorer,weight in self.views:
   prior=scorer.score;oldviews.append((scorer,prior))
   def scored(row,candidates,beams,prior=prior,name=scorer.name):
    v=prior(row,candidates,beams);views.append({'sample_id':str(row['sample_id']),'view':name,'ids':[str(c['vessel_id']) for c in candidates],'scores':np.asarray(v).tolist()});return v
   scorer.score=scored
  try:final=original(self,rows,signals,blocks)
  finally:
   E.member_score=oldmember;self.baseline.score_groups=oldbase
   for scorer,prior in oldviews:scorer.score=prior
  assert len(base)==len(rows)==len(final);assert len(raw)==len(self.baseline.members)
  offset=0
  for row,sig,F,b,f in zip(rows,signals,blocks,base,final):
   sid=str(row['sample_id']);assert '/' not in sid;n=len(F);ids=np.array([str(self.r.ids[int(i)]) for i in sig['cand']]);raws=np.stack([x[offset:offset+n] for x in raw]);offset+=n
   np.savez_compressed(folder/(sid+'.npz'),features=F,raw=raws,base=b,final=f,candidate_ids=ids,candidate_index=sig['cand'])
   record={'sample_id':sid,'name':str(row['vessel_name']),'beams':list(sig.get('llm_beams') or []),'views':[v for v in views if v['sample_id']==sid],'members':[{'kind':k,'weight':w} for k,m,w in self.baseline.members]}
   (folder/(sid+'.json')).write_text(json.dumps(record,ensure_ascii=False,allow_nan=False))
  print('CAPTURED',len(rows),'rows',flush=True)
  return final
 EncoderViewsEnsemble.score_records=capture
