from pathlib import Path
import ast,csv,hashlib,json,os,pickle,sys,time,unicodedata,collections
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['KO_DICT']='0';sys.dont_write_bytecode=True
import numpy as np
from rapidfuzz import fuzz,process
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
M=ROOT/'astra/arena_20261101/baseline/model/en/model';sys.path.insert(0,str(M))
import opus_lib as L
from ksh_hangul import english_key,hangul_key
from kocat import jamo

def read(path):
 with Path(path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def fold(vid):return int(hashlib.sha256(('structural-v1:'+vid).encode()).hexdigest()[:8],16)%5
def clean(s):return ''.join(c.upper() for c in unicodedata.normalize('NFKC',s) if c.isalnum())
def choose_simple(scores,eligible,k):
 rank=np.argsort(-scores,kind='stable');a=rank[:k];b=rank[eligible[rank]][:k]
 return set(a[eligible[a]].tolist()),set(b.tolist()),int((~eligible[a]).sum())
def choose_kocat(scores,owners,eligible,k):
 def sel(keep):
  ix=np.flatnonzero(keep)
  if not len(ix):return set(),0
  sc=scores[ix];pos=np.argpartition(-sc,min(len(sc)-1,4*k))[:4*k];pos=pos[np.lexsort((pos,-sc[pos]))];seen=[]
  for z in pos:
   v=int(owners[ix[z]])
   if v not in seen:seen.append(v)
   if len(seen)==k:break
  return set(seen),sum(not eligible[v] for v in seen)
 a,waste=sel(np.ones(len(scores),bool));b,_=sel(eligible[owners]);return {v for v in a if eligible[v]},b,waste

def main():
 start=time.perf_counter();cat=read(ROOT/'data/open/vessel_catalog.csv');train=read(ROOT/'data/open/train.csv');ids=[r['vessel_id'] for r in cat];id2i={v:i for i,v in enumerate(ids)}
 alltrain={r['target_vessel_id'] for r in train};stats=pickle.loads((M/'stats.pkl').read_bytes());ex112=set(stats['cs_map'].values());ex20=set(json.loads((ROOT/'astra/arena_20261101/baseline/model/ko/model/train_vessels.json').read_text()))
 shots={clean(x[0]) for x in json.loads((ROOT/'shared/kocat/fewshot_pairs.json').read_text())}
 tree=ast.parse((ROOT/'astra/arena_20261101/baseline/model/ko/model/src/llm_translit.py').read_text())
 for node in tree.body:
  if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='FEW_SHOT' for x in node.targets):shots.update(clean(en) for ko,en in ast.literal_eval(node.value))
 shotids={r['vessel_id'] for r in cat if clean(r['vessel_name']) in shots}
 native={ (r['target_vessel_id'],r['vessel_name']):r for r in train if L.is_hangul(r['vessel_name']) }
 query=[r for (v,n),r in sorted(native.items()) if v not in shotids]
 ckeys=[L.phon_key(L.acronym_expand(r['vessel_name'])) for r in cat];kkeys=[english_key(r['vessel_name']).replace(' ','') for r in cat]
 table={r['vessel_name']:r for r in read(M/'catalog_ko.csv')};aliases=[];owners=[]
 for i,r in enumerate(cat):
  row=table.get(r['vessel_name'],{});aa=list(dict.fromkeys(jamo(row.get(f'ko{j}','')) for j in range(4) if any('가'<=c<='힣' for c in row.get(f'ko{j}',''))))
  for a in aa:aliases.append(a);owners.append(i)
 owners=np.asarray(owners,dtype=int)
 elig={f:np.asarray([v not in alltrain or fold(v)==f for v in ids],bool) for f in range(5)}
 rows=[]
 for z,r in enumerate(query):
  vid=r['target_vessel_id'];f=fold(vid);truth=id2i[vid];eligible=elig[f];assert eligible[truth]
  q=r['vessel_name'];channels={};before=set();after=set()
  for label,key,choices,k in [('112_consonant500',L.ko_keys(q)[1],ckeys,500),('ksh_phonetic500',hangul_key(q).replace(' ',''),kkeys,500)]:
   sc=process.cdist([key],choices,scorer=fuzz.ratio,workers=1)[0];a,b,w=choose_simple(sc,eligible,k);before|=a;after|=b;channels[label]={'before':truth in a,'after':truth in b,'wasted_slots':w,'before_n':len(a),'after_n':len(b)}
  sc=process.cdist([jamo(q)],aliases,scorer=fuzz.ratio,workers=1)[0];a,b,w=choose_kocat(sc,owners,eligible,50);before|=a;after|=b;channels['112_kocat50']={'before':truth in a,'after':truth in b,'wasted_slots':w,'before_n':len(a),'after_n':len(b)}
  rows.append({'vessel_id':vid,'vessel_name':q,'fold':f,'before':truth in before,'after':truth in after,'before_union_n':len(before),'after_union_n':len(after),'channels':channels})
  if (z+1)%100==0:print(json.dumps({'done':z+1,'seconds':time.perf_counter()-start}),flush=True)
 summary={'scope':'Native real Korean name retrieval; no trained ranker, no C_vote predictions, no Public extrapolation','native_unique_forms':len(native),'fewshot_excluded_native_forms':len(native)-len(query),'evaluated_forms':len(rows),'evaluated_vessels':len({r['vessel_id'] for r in rows}),'catalog':len(cat),'train_vessels':len(alltrain),'112_exclude_symmetric_difference_train':len(ex112^alltrain),'v20_exclude_symmetric_difference_train':len(ex20^alltrain),'before_union_hits':sum(r['before'] for r in rows),'after_union_hits':sum(r['after'] for r in rows),'union_rescue':sum(not r['before'] and r['after'] for r in rows),'union_harm':sum(r['before'] and not r['after'] for r in rows),'channels':{},'folds':{},'seconds':time.perf_counter()-start}
 for label in rows[0]['channels']:
  cc=[r['channels'][label] for r in rows];summary['channels'][label]={'before_hits':sum(r['before'] for r in cc),'after_hits':sum(r['after'] for r in cc),'rescue':sum(not r['before'] and r['after'] for r in cc),'harm':sum(r['before'] and not r['after'] for r in cc),'mean_wasted_slots':sum(r['wasted_slots'] for r in cc)/len(cc),'max_wasted_slots':max(r['wasted_slots'] for r in cc)}
 for f in range(5):
  rr=[r for r in rows if r['fold']==f];summary['folds'][str(f)]={'forms':len(rr),'fit_vessels':sum(fold(v)!=f for v in alltrain),'before_hits':sum(r['before'] for r in rr),'after_hits':sum(r['after'] for r in rr)}
 summary['cached_native_recall84']={}
 for name in ['recall84','recall84_fold1','recall84_fold2','recall84_fold3','recall84_fold4']:
  p=ROOT/'astra'/name/'rows.csv'
  if p.exists():
   rr=[r for r in read(p) if r['variant']=='orig' and r['ko']=='True'];summary['cached_native_recall84'][name]={'rows':len(rr),'old_hits':sum(r['old']=='True' for r in rr),'new_hits':sum(r['new']=='True' for r in rr),'note':'historical fold-specific retrieval, not current C_vote'}
 (OUT/'rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));(OUT/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
