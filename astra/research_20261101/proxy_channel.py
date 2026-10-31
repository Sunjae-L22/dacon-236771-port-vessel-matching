from pathlib import Path
import sys,csv,json,collections,math,pickle,time
sys.dont_write_bytecode=True
from rapidfuzz import process
from rapidfuzz.distance import DamerauLevenshtein
from channel_probe import norm,hangul,pathways
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent

def read(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def main():
 t=time.time();config=json.loads((OUT/'CHANNEL_PREREG.json').read_text());pm={int(k):v for k,v in config['p0_p1_p2'].items()};op=config['operations']
 cat=read(ROOT/'data/open/vessel_catalog.csv');table={r['vessel_name']:r for r in read(ROOT/'shared/kocat/out/catalog_ko.csv')};aliases=[];owners=[];cnt={}
 for c in cat:
  r=table.get(c['vessel_name'],{});aa=sorted({norm(r.get(f'ko{i}','')) for i in range(4) if hangul(r.get(f'ko{i}',''))});cnt[c['vessel_id']]=len(aa)
  for a in aa:aliases.append(a);owners.append(c['vessel_id'])
 enfit={r['vessel_id'] for r in [json.loads(s) for s in (OUT/'english_pairs.jsonl').read_text().splitlines()] if r['fold']!=0}
 result={};records=[]
 for fold in (2,4):
  stats=pickle.loads((ROOT/f'astra/byt5_reuse_20261031/package/baseline/fold{fold}/stats.pkl').read_bytes());exclude=set(stats['cs_map'].values());cache={};rr=[]
  rows=[json.loads(s) for s in (ROOT/f'astra/byt5_base112_20261031/fold{fold}/rows.jsonl').read_text().splitlines()]
  for r in rows:
   if not r['primary'] or r['target'] in enfit:continue
   new=r['p112'];strict=new;q=norm(r['name'])
   if r['gap110']<4:
    if q not in cache:
     ss=collections.Counter()
     for alias,dist,idx in process.extract(q,aliases,scorer=DamerauLevenshtein.distance,score_cutoff=2,limit=None):
      owner=owners[idx]
      if owner in exclude:continue
      val=sum(pm[len(path)]*weight*math.prod(op[k] for k in path) for path,weight in pathways(alias,q))
      if val:ss[owner]+=val/cnt[owner]
     ranked=sorted(ss,key=lambda v:(-ss[v],v));cache[q]=[(v,ss[v]) for v in ranked]
    ss=cache[q]
    if ss:
     new=ss[0][0]
     if ss[0][1]/sum(x[1] for x in ss)>=.99:strict=new
   z={**r,'fold':fold,'channel':new,'strict':strict};rr.append(z);records.append(z)
  result[str(fold)]={}
  for method in ('channel','strict'):
   result[str(fold)][method]={}
   for cohort in ('clean','noise','all'):
    xs=[x for x in rr if cohort=='all' or x['variant'].startswith('clean')==(cohort=='clean')];resc=[x for x in xs if x['p112']!=x['target'] and x[method]==x['target']];harm=[x for x in xs if x['p112']==x['target'] and x[method]!=x['target']]
    result[str(fold)][method][cohort]={'rows':len(xs),'rescue':len(resc),'harm':len(harm),'rescue_vessels':len({x['target'] for x in resc}),'harm_vessels':len({x['target'] for x in harm}),'changed':sum(x[method]!=x['p112'] for x in xs),'undo112repairs':sum(x['p110']!=x['target'] for x in harm)}
  print('fold',fold,json.dumps(result[str(fold)]),flush=True)
 (OUT/'PROXY112_CHANNEL.json').write_text(json.dumps({'scope':'reused heldout112 recipe proxy, NOT deployed112/C_vote; same corpus previously used; falsification only','seconds':time.time()-t,'metrics':result},indent=2));(OUT/'proxy112_channel_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
if __name__=='__main__':main()
