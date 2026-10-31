from pathlib import Path
import sys,csv,json,hashlib,random,time,collections,math,ast
sys.dont_write_bytecode=True
import numpy as np
from rapidfuzz import process,fuzz
from rapidfuzz.distance import DamerauLevenshtein
from channel_probe import norm,hangul,pathways
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'shared/src'))
from kocat import jamo

def read(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def hashfold(vid):return int(hashlib.sha256(('channel-v1:'+vid).encode()).hexdigest()[:8],16)%5

def main():
 t0=time.time();fit=json.loads((OUT/'ENGLISH_CHANNEL.json').read_text());ops=fit['operation_probabilities'];steps={int(k):v for k,v in fit['conditional_nonidentity_step_probabilities'].items()}
 en=[json.loads(x) for x in (OUT/'english_pairs.jsonl').read_text().splitlines()];fitrows=[r for r in en if r['fold']!=0];p0=sum(r['label']=='exact' for r in fitrows)/len(fitrows);pm={0:p0,1:(1-p0)*steps[1],2:(1-p0)*steps[2]}
 train=read(ROOT/'data/open/train.csv');cat=read(ROOT/'data/open/vessel_catalog.csv');catid={r['vessel_id']:r for r in cat}
 shots={norm(x[0]) for x in json.loads((ROOT/'shared/kocat/fewshot_pairs.json').read_text())}
 tree=ast.parse((ROOT/'astra/arena_20261101/baseline/model/ko/model/src/llm_translit.py').read_text())
 for n in tree.body:
  if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='FEW_SHOT' for x in n.targets):shots.update(norm(x[1]) for x in ast.literal_eval(n.value))
 # Development = hash fold1 Korean vessels, final = fold0, never used in English fitting.
 enfitships={r['vessel_id'] for r in fitrows};roots={}
 for r in train:
  vid=r['target_vessel_id'];q=norm(r['vessel_name'])
  if not hangul(q) or vid in enfitships or norm(catid[vid]['vessel_name']) in shots or len(q)<4:continue
  if hashfold(vid) not in (0,1):continue
  old=roots.get(vid)
  if old is None or r['sample_id']<old['sample_id']:roots[vid]=r
 # Read fixed table without changing it; every official vessel remains a candidate.
 table={r['vessel_name']:r for r in read(ROOT/'shared/kocat/out/catalog_ko.csv')};aliases=[];owner=[];count={}
 for c in cat:
  r=table.get(c['vessel_name'],{});aa=sorted({norm(r.get(f'ko{i}','')) for i in range(4) if hangul(r.get(f'ko{i}',''))})
  count[c['vessel_id']]=len(aa)
  for a in aa:aliases.append(a);owner.append(c['vessel_id'])
 ja=[jamo(a) for a in aliases]
 config={'p0_p1_p2':pm,'operations':ops,'alias_prior':'uniform over distinct official72B-table aliases per vessel','channel':'sum all <=2 sequential delete/dup/swap paths with per-position probabilities','fallback':'original jamo nearest if no positive channel candidate','evaluation':'real Korean train name roots; fewshot & ENfit ships excluded; artificial noise clearly labelled','fold1':'development','fold0':'untouched validation; reported independently, no tuning between folds','metadata':'none; intrinsic name matching, NOT C_vote comparison'}
 (OUT/'CHANNEL_PREREG.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
 results=[]
 for num,(vid,r) in enumerate(sorted(roots.items())):
  clean=norm(r['vessel_name']);variants={'clean':clean}
  for op in ('delete','dup','swap','double'):
   rng=random.Random('channel-eval-v1:'+vid+':'+op);q=clean
   for op2 in (('delete','swap') if op=='double' else (op,)):
    if op2=='swap':
     pos=[i for i in range(len(q)-1) if q[i]!=q[i+1]];i=rng.choice(pos);q=q[:i]+q[i+1]+q[i]+q[i+2:]
    else:
     i=rng.randrange(len(q));q=q[:i]+q[i+1:] if op2=='delete' else q[:i+1]+q[i]+q[i+1:]
   variants[op]=q
  for condition,q in variants.items():
   matches=process.extract(jamo(q),ja,scorer=fuzz.ratio,limit=400)
   bs={}
   for _,score,idx in matches:bs[owner[idx]]=max(score,bs.get(owner[idx],-1))
   base=sorted(bs,key=lambda v:(-bs[v],v))[0]
   candidates=process.extract(q,aliases,scorer=DamerauLevenshtein.distance,score_cutoff=2,limit=None)
   scores=collections.Counter();hitpaths=0
   for alias,dist,idx in candidates:
    pp=pathways(alias,q)
    val=sum(pm[len(path)]*weight*math.prod(ops[op] for op in path) for path,weight in pp)
    if val:scores[owner[idx]]+=val/count[owner[idx]];hitpaths+=len(pp)
   new=sorted(scores,key=lambda v:(-scores[v],v))[0] if scores else base
   order=sorted(scores,key=lambda v:(-scores[v],v));top2=[(v,scores[v]) for v in order[:2]]
   results.append(dict(vessel_id=vid,fold=hashfold(vid),condition=condition,clean=clean,query=q,baseline=base,channel=new,base_correct=base==vid,new_correct=new==vid,rescue=base!=vid and new==vid,harm=base==vid and new!=vid,channel_covered=vid in scores,channel_candidates=len(scores),dl_alias_candidates=len(candidates),top2=top2))
  if (num+1)%10==0:print('ships',num+1,'rows',len(results),'seconds',round(time.time()-t0,1),flush=True)
 summary={'config':config,'unique_vessels':len(roots),'rows':len(results),'seconds':time.time()-t0,'cohorts':{}}
 for fold in (1,0):
  for condition in variants:
   rows=[r for r in results if r['fold']==fold and r['condition']==condition]
   summary['cohorts'][str(fold)+'_'+condition]={k:sum(r[k] for r in rows) for k in ('base_correct','new_correct','rescue','harm','channel_covered')};summary['cohorts'][str(fold)+'_'+condition]['rows']=len(rows)
 (OUT/'KOREAN_CHANNEL_RESULTS.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));(OUT/'korean_channel_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results));print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
