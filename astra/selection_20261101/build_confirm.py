"""Fixed secondary diagnostic from authentic-train token composition, not alias query copying."""
from pathlib import Path
import json,csv,collections,re,ast,hashlib,random,datetime
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=HERE/'confirm/eval'
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def norm(s):return re.sub(r'[^가-힣A-Z0-9]','',s.upper())
cat=read(ROOT/'data/open/vessel_catalog.csv');byid={r['vessel_id']:r for r in cat};train=read(ROOT/'data/open/train.csv');excluded={r['target_vessel_id'] for r in train};pool=json.loads((ROOT/'astra/research_20261101/COMPOSITIONAL_POOL.json').read_text());kan=json.loads((ROOT/'astra/arena_20261101/baseline/model/ko/model/catalog_ko.json').read_text());kc={r['vessel_name']:r for r in read(ROOT/'shared/kocat/out/catalog_ko.csv')}
shots={norm(x[0]) for x in json.loads((ROOT/'shared/kocat/fewshot_pairs.json').read_text())};tree=ast.parse((ROOT/'astra/arena_20261101/baseline/model/ko/model/src/llm_translit.py').read_text())
for n in tree.body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FEW_SHOT' for t in n.targets):shots.update(norm(x[1]) for x in ast.literal_eval(n.value))
selected=[]
for r in pool:
 aliases=[kan.get(r['vessel_name'],'')]+[kc.get(r['vessel_name'],{}).get('ko'+str(i),'') for i in range(4)]
 if r['vessel_id'] not in excluded and norm(r['vessel_name']) not in shots and norm(r['query_ko']) not in {norm(s) for s in aliases}:selected.append(r)
selected.sort(key=lambda r:hashlib.sha256(('selection-confirm-v1:'+r['vessel_id']).encode()).digest())
maps={k:collections.defaultdict(collections.Counter) for k in ('flag_name','vessel_type')}
for r in train:
 c=byid[r['target_vessel_id']]
 for k in maps:
  if r[k] and c[k]:maps[k][c[k]][r[k]]+=1

def mutate(s,op,rng):
 pos=[i for i,c in enumerate(s) if '가'<=c<='힣']
 if op=='swap':pos=[i for i in pos if i+1<len(s) and '가'<=s[i+1]<='힣' and s[i]!=s[i+1]]
 if not pos:return s
 i=rng.choice(pos)
 if op=='delete':return s[:i]+s[i+1:]
 if op=='dup':return s[:i+1]+s[i]+s[i+1:]
 return s[:i]+s[i+1]+s[i]+s[i+2:]
rows=[];truth=[]
for i,c in enumerate(selected):
 metadata={k:sorted(maps[k][c[k]],key=lambda s:(-maps[k][c[k]][s],s))[0] if maps[k][c[k]] else '' for k in maps}
 for var in ('clean_keep','clean_nogt','delete_nogt','dup_nogt','swap_nogt','double_nogt'):
  q=c['query_ko'];rng=random.Random('selection-confirm-v1:'+c['vessel_id']+var)
  if var=='double_nogt':q=mutate(mutate(q,'delete',rng),'swap',rng)
  elif not var.startswith('clean'):q=mutate(q,var.split('_')[0],rng)
  sid=f'SELCONF_{i:03d}_{var}';gt=c['gross_tonnage'] if var=='clean_keep' else ''
  rows.append(dict(sample_id=sid,vessel_name=q,call_sign='',gross_tonnage=gt,international_gross_tonnage=gt,**metadata,arrival_year='2025'));truth.append(dict(sample_id=sid,target_vessel_id=c['vessel_id'],variant=var,source='native_train_token_composition_different_from_both_alias_tables'))
for name,rr in [('test.csv',rows),('truth.csv',truth)]:
 with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
report={'registered_kst':datetime.datetime.now().astimezone().isoformat(),'ships':len(selected),'rows':len(rows),'selection':'all eligible unseen target ships in fixed old native-token composition pool, with clean query different from every stored v20/72B alias; fewshots excluded; no model-error filtering','query_source':'official train aligned token lexicon; composed labels are synthetic, not observed human names for these target ships','limitations':['Only a small synthetic style-shift diagnostic, not independent human test','Metadata derived from target catalog and official train mappings','Native token vocabulary overlaps learner training vocabulary'],'models_frozen_before_confirmation_results':True,'test_sha256':hashlib.sha256((OUT/'test.csv').read_bytes()).hexdigest()};(OUT/'MANIFEST.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
