from pathlib import Path
import sys,os,csv,json,hashlib,collections,math,time,unicodedata
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
from rapidfuzz.distance import DamerauLevenshtein

def norm(s):return ''.join(c.upper() for c in unicodedata.normalize('NFC',str(s)) if c.isalnum())
def hangul(s):return any('가'<=c<='힣' for c in str(s))
def forward(s):
 n=len(s)
 for i,c in enumerate(s):
  yield s[:i]+s[i+1:],'delete',1/n
  yield s[:i+1]+c+s[i+1:],'dup',1/n
 pairs=[i for i in range(n-1) if s[i]!=s[i+1]]
 for i in pairs:yield s[:i]+s[i+1]+s[i]+s[i+2:],'swap',1/len(pairs)
def one(s,t):
 n=len(s);out=collections.Counter()
 if not n:return out
 if len(t)==n-1:
  out['delete']=sum(s[:i]+s[i+1:]==t for i in range(n))/n
 elif len(t)==n+1:
  out['dup']=sum(s[:i+1]+s[i]+s[i+1:]==t for i in range(n))/n
 elif len(t)==n:
  dif=[i for i,(a,b) in enumerate(zip(s,t)) if a!=b]
  if len(dif)==2 and dif[1]==dif[0]+1:
   i=dif[0]
   if s[i]==t[i+1] and s[i+1]==t[i]:out['swap']=1/sum(s[j]!=s[j+1] for j in range(n-1))
 return {k:v for k,v in out.items() if v}
def pathways(s,t):
 paths=[]
 if s==t:paths.append(((),1.))
 paths.extend(((k,),v) for k,v in one(s,t).items())
 if abs(len(s)-len(t))<=2 and DamerauLevenshtein.distance(s,t)<=2:
  for mid,a,pa in forward(s):
   if mid:
    paths.extend(((a,b),pa*pb) for b,pb in one(mid,t).items())
 return paths

def main():
 cat=list(csv.DictReader((ROOT/'data/open/vessel_catalog.csv').open(encoding='utf-8-sig')));byid={r['vessel_id']:r for r in cat};train=list(csv.DictReader((ROOT/'data/open/train.csv').open(encoding='utf-8-sig')))
 seen=set();rr=[];stats=collections.Counter();examples={}
 t=time.time()
 for r in train:
  if hangul(r['vessel_name']):continue
  s=norm(byid[r['target_vessel_id']]['vessel_name']);q=norm(r['vessel_name']);key=(r['target_vessel_id'],q)
  if not s or not q or key in seen:continue
  seen.add(key);dist=DamerauLevenshtein.distance(s,q);paths=pathways(s,q) if dist<=2 else []
  label='exact' if s==q else 'one_operation' if one(s,q) else 'two_operations' if paths else 'other'
  stats[label]+=1;examples.setdefault(label,[])
  if len(examples[label])<8:examples[label].append([s,q])
  fold=int(hashlib.sha256(('channel-v1:'+r['target_vessel_id']).encode()).hexdigest()[:8],16)%5
  rr.append(dict(vessel_id=r['target_vessel_id'],clean=s,query=q,fold=fold,label=label,paths=paths))
 # Fit observed non-identity records with <=2 operation explanations on folds1..4 only.
 fit=[r for r in rr if r['fold']!=0 and r['label'] in ('one_operation','two_operations')]
 op={'delete':1/3,'dup':1/3,'swap':1/3};steps={1:.7,2:.3}
 for iteration in range(15):
  oc=collections.Counter({k:1. for k in op});sc=collections.Counter({1:1.,2:1.})
  for r in fit:
   pp=[(ops,w*steps[len(ops)]*math.prod(op[k] for k in ops)) for ops,w in r['paths'] if ops]
   z=sum(w for _,w in pp)
   for ops,w in pp:
    v=w/z;sc[len(ops)]+=v
    for k in ops:oc[k]+=v
  op={k:oc[k]/sum(oc.values()) for k in op};steps={k:sc[k]/sum(sc.values()) for k in steps}
 result={'unique_english_ship_name_pairs':len(rr),'counts':dict(stats),'examples':examples,'fit_nonidentity_pairs':len(fit),'fit_folds':[1,2,3,4],'evaluation_fold':0,'operation_probabilities':op,'conditional_nonidentity_step_probabilities':steps,'seconds':time.time()-t,'fit_unit':'unique ship/query form; not repeated rows','not_Korean_noise_estimate':True,'note':'English operation mixture transfers to Korean only as a testable hypothesis. No test rows or Public submissions.'}
 (OUT/'ENGLISH_CHANNEL.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));(OUT/'english_pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr));print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
