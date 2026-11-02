"""Independent whole-name corruption channel. No labels, no cross-row fitting."""
from pathlib import Path
import csv,json,math,unicodedata,collections
from rapidfuzz import process
from rapidfuzz.distance import DamerauLevenshtein
try:
 from normalized_channel import path_coefficients,coefficient_score
except ImportError:
 from translit.normalized_channel import path_coefficients,coefficient_score

def normalize(s):return ''.join(c.upper() for c in unicodedata.normalize('NFC',str(s or '')) if c.isalnum())
def hangul(s):return any('가'<=c<='힣' for c in unicodedata.normalize('NFC',str(s or '')))
def read(p):
 with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

class Channel:
 def __init__(self,catalog,table,params,excluded=()):
  self.params=params;self.excluded=set(excluded);self.ids={str(r['vessel_id']) for r in catalog}
  for k,keys in (('operation_probabilities',{'delete','dup','swap'}),('step_probabilities',{'0','1','2'})):
   if set(params[k])!=keys:raise ValueError('Invalid channel parameter keys')
   if not math.isclose(sum(params[k].values()),1.,abs_tol=1e-9) or any(not math.isfinite(v) or v<=0 for v in params[k].values()):raise ValueError('Invalid normalized channel parameters')
  byname={r['vessel_name']:r for r in table};self.aliases=[];self.owners=[];self.alias_counts={}
  for c in catalog:
   vid=str(c['vessel_id'])
   if vid in self.excluded:continue
   r=byname.get(c['vessel_name'],{});ss=sorted({normalize(r.get(f'ko{i}','')) for i in range(4) if hangul(r.get(f'ko{i}',''))})
   self.alias_counts[vid]=len(ss)
   for a in ss:self.aliases.append(a);self.owners.append(vid)
  if not self.aliases:raise ValueError('No Korean aliases')
 @classmethod
 def load(cls,catalog,table,params,excluded=()):return cls(read(catalog),read(table),json.loads(Path(params).read_text()),excluded)
 def query(self,name):
  q=normalize(name);mass=collections.Counter();best_alias={};alias_hits=0;deletion_only=set()
  if not q or not hangul(q):return {'winner':None,'score_mass_share':0.,'supported_vessels':0,'top':[]}
  for a,dist,j in process.extract(q,self.aliases,scorer=DamerauLevenshtein.distance,score_cutoff=2,limit=None):
   coeff=path_coefficients(a,q)
   p=coefficient_score(coeff,self.params)
   if p<=0:continue
   vid=self.owners[j];v=p/self.alias_counts[vid];mass[vid]+=v;alias_hits+=1
   if any(coeff.get(k,0)>0 for k in ('','delete','delete/delete')):deletion_only.add(vid)
   if vid not in best_alias or v>best_alias[vid][0] or v==best_alias[vid][0] and a<best_alias[vid][1]:best_alias[vid]=(v,a)
  if not mass:return {'winner':None,'score_mass_share':0.,'supported_vessels':0,'top':[]}
  order=sorted(mass,key=lambda v:(-mass[v],v));total=math.fsum(mass.values());win=order[0]
  return {'winner':win,'winner_has_deletion_only_path':win in deletion_only,'score_mass_share':mass[win]/total,'supported_vessels':len(order),'supported_aliases':alias_hits,'total_channel_mass':total,'top':[{'vessel_id':v,'mass':mass[v],'best_alias':best_alias[v][1]} for v in order[:5]]}

def decide(row,baseline,gap,trace):
 """Fixed exploratory overlay; score share is not calibrated correctness."""
 if not isinstance(baseline,str) or not baseline:raise ValueError('Missing baseline prediction')
 if not hangul(row.get('vessel_name','')):return baseline,'english'
 if not math.isfinite(float(gap)):raise ValueError('Missing original110 gap')
 if float(gap)>=4:return baseline,'protected110'
 if not trace.get('winner'):return baseline,'channel_unsupported'
 share=float(trace['score_mass_share'])
 if not math.isfinite(share) or not 0<=share<=1:raise ValueError('Invalid channel mass share')
 if share<.99:return baseline,'ambiguous_channel'
 if trace.get('winner_has_deletion_only_path',True):return baseline,'deletion_only_ambiguous'
 return trace['winner'],'nondeletion_channel_dominant'
