from pathlib import Path
import sys,csv,json,hashlib,random,collections,time
sys.dont_write_bytecode=True
from channel_model import Channel,normalize,hangul,read
from rapidfuzz import process,fuzz
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'shared/src'));from kocat import jamo

def main():
 t=time.time();params=json.loads((OUT/'translit/NORMALIZED_FIT.json').read_text());fitids=set(params['train_vessel_ids']);cat=read(ROOT/'data/open/vessel_catalog.csv');byid={r['vessel_id']:r for r in cat};train=read(ROOT/'data/open/train.csv');shots={normalize(x[0]) for x in json.loads((ROOT/'shared/kocat/fewshot_pairs.json').read_text())}
 # Include the other published Kanana few-shot sources.
 import ast
 tree=ast.parse((ROOT/'astra/arena_20261101/baseline/model/ko/model/src/llm_translit.py').read_text())
 for n in tree.body:
  if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='FEW_SHOT' for x in n.targets):shots.update(normalize(x[1]) for x in ast.literal_eval(n.value))
 roots={}
 for r in train:
  vid=r['target_vessel_id'];q=normalize(r['vessel_name']);fold=int(hashlib.sha256(('channel-v1:'+vid).encode()).hexdigest()[:8],16)%5
  if fold!=3 or vid in fitids or not hangul(q) or len(q)<4 or normalize(byid[vid]['vessel_name']) in shots:continue
  if vid not in roots or r['sample_id']<roots[vid]['sample_id']:roots[vid]=r
 model=Channel(cat,read(ROOT/'shared/kocat/out/catalog_ko.csv'),params,fitids);ja=[jamo(a) for a in model.aliases];rr=[]
 for vid,r in sorted(roots.items()):
  clean=normalize(r['vessel_name']);vv={'clean':clean}
  for op in ('delete','dup','swap','double'):
   q=clean;rng=random.Random('channel-eval-v1:'+vid+':'+op)
   for k in (('delete','swap') if op=='double' else (op,)):
    if k=='swap':
     pos=[i for i in range(len(q)-1) if q[i]!=q[i+1]];i=rng.choice(pos);q=q[:i]+q[i+1]+q[i]+q[i+2:]
    else:
     i=rng.randrange(len(q));q=q[:i]+q[i+1:] if k=='delete' else q[:i+1]+q[i]+q[i+1:]
   vv[op]=q
  for condition,q in vv.items():
   matches=process.extract(jamo(q),ja,scorer=fuzz.ratio,limit=400);bs={}
   for _,score,j in matches:bs[model.owners[j]]=max(score,bs.get(model.owners[j],-1))
   old=sorted(bs,key=lambda v:(-bs[v],v))[0];tr=model.query(q);new=tr['winner'] if tr['winner'] and tr['score_mass_share']>=.99 else old
   rr.append(dict(vessel_id=vid,condition=condition,query=q,baseline=old,candidate=new,rescue=old!=vid and new==vid,harm=old==vid and new!=vid,base_correct=old==vid,new_correct=new==vid,trace=tr))
 summary={'evidence':'fresh hashfold3 realKname roots + labelled injectedcorruptions; name-onlybaseline NOT112/C_vote','normalized_channel':True,'strict_share':.99,'this_channel_hypothesis_previously_evaluated_hashfold3':False,'fits_on_Korean_labels':False,'ships':len(roots),'rows':len(rr),'seconds':time.time()-t,'cohorts':{}}
 for condition in vv:
  xs=[r for r in rr if r['condition']==condition];summary['cohorts'][condition]={'rows':len(xs),**{k:sum(r[k] for r in xs) for k in ('base_correct','new_correct','rescue','harm')}}
 (OUT/'FRESH_V2_RESULTS.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));(OUT/'fresh_v2_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr));print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
