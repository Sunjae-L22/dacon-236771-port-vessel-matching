"""Fixed catalog-derived stress data; explicitly NOT independent validation."""
from pathlib import Path
import sys,ast,csv,json,hashlib,pickle,random,re,shutil
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
DATA=HERE/'package/data';DATA.mkdir(parents=True,exist_ok=True)
BASE=ROOT/'astra/arena_20261101/baseline'
def read(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(s):return ''.join(s.upper().split())
def acceptable(s):return bool(re.fullmatch(r'[가-힣0-9 .-]+',s or '')) and 4<=sum('가'<=c<='힣' for c in s)<=24

cat=read(ROOT/'data/open/vessel_catalog.csv');train=read(ROOT/'data/open/train.csv')
excluded={r['target_vessel_id'] for r in train}|set(json.loads((BASE/'model/ko/model/train_vessels.json').read_text()))
shots=json.loads((ROOT/'shared/kocat/fewshot_pairs.json').read_text());shotnames={canonical(x[0]) for x in shots}
tree=ast.parse((BASE/'model/ko/model/src/llm_translit.py').read_text())
for n in tree.body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FEW_SHOT' for t in n.targets):shotnames.update(canonical(x[1]) for x in ast.literal_eval(n.value))
for r in cat:
 if canonical(r['vessel_name']) in shotnames:excluded.add(r['vessel_id'])
names={}
for r in cat:names.setdefault(canonical(r['vessel_name']),[]).append(r['vessel_id'])
kanana=json.loads((BASE/'model/ko/model/catalog_ko.json').read_text())
kocat={r['vessel_name']:r['ko0'] for r in read(BASE/'model/en/model/catalog_ko.csv')}
pool=[r for r in cat if r['vessel_id'] not in excluded and len(names[canonical(r['vessel_name'])])==1 and acceptable(kanana.get(r['vessel_name'],'')) and acceptable(kocat.get(r['vessel_name'],''))]
pool.sort(key=lambda r:hashlib.sha256(('astra-highgain-20261101:'+r['vessel_id']).encode()).digest())
selected=pool[:64];assert len(selected)==64

def mutate(s,kind,rng):
 positions=[i for i,c in enumerate(s) if '가'<=c<='힣'];i=rng.choice(positions)
 if kind=='delete':return s[:i]+s[i+1:]
 if kind=='dup':return s[:i+1]+s[i]+s[i+1:]
 pairs=[j for j in positions if j+1<len(s) and '가'<=s[j+1]<='힣' and s[j]!=s[j+1]]
 if not pairs:return mutate(s,'dup',rng)
 i=rng.choice(pairs);return s[:i]+s[i+1]+s[i]+s[i+2:]
rows=[];truth=[]
for i,c in enumerate(selected):
 source='kanana_table' if i%2==0 else '72b_table';name=(kanana if i%2==0 else kocat)[c['vessel_name']]
 for variant in ['clean_keep','clean_nogt','delete_nogt','dup_nogt','swap_nogt','double_nogt']:
  rng=random.Random(f'20261101:{c["vessel_id"]}:{variant}');q=name
  if variant=='double_nogt':q=mutate(mutate(q,'delete',rng),'swap',rng)
  elif not variant.startswith('clean'):q=mutate(q,variant.split('_')[0],rng)
  sid=f'ASTRAHG_{i:03d}_{variant}';gt=c['gross_tonnage'] if variant=='clean_keep' else ''
  rows.append(dict(sample_id=sid,vessel_name=q,call_sign='',gross_tonnage=gt,international_gross_tonnage=gt,flag_name=c['flag_name'],vessel_type=c['vessel_type'],arrival_year='2025'))
  truth.append(dict(sample_id=sid,target_vessel_id=c['vessel_id'],variant=variant,source=source,evidence_class='SYNTHETIC_TABLE_SEEN_STYLE_BIASED',supervised_train_vessel=False))
for i,c in enumerate(selected[:16]):
 sid=f'ASTRAHG_E_{i:03d}';rows.append(dict(sample_id=sid,vessel_name=c['vessel_name'],call_sign='',gross_tonnage='',international_gross_tonnage='',flag_name=c['flag_name'],vessel_type=c['vessel_type'],arrival_year='2025'))
 truth.append(dict(sample_id=sid,target_vessel_id=c['vessel_id'],variant='english',source='official_catalog',evidence_class='SYNTHETIC_TABLE_SEEN_STYLE_BIASED',supervised_train_vessel=False))
assert all(r['target_vessel_id'] not in excluded for r in truth)
def write(name,data):
 with (DATA/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
write('full_test.csv',rows);write('full_truth.csv',truth)
pilot={r['sample_id'] for r in rows if (r['sample_id'].startswith('ASTRAHG_E_') and int(r['sample_id'].split('_')[-1])<8) or (not r['sample_id'].startswith('ASTRAHG_E_') and int(r['sample_id'].split('_')[1])<16)}
write('pilot_test.csv',[r for r in rows if r['sample_id'] in pilot]);write('pilot_truth.csv',[r for r in truth if r['sample_id'] in pilot])
shutil.copy(ROOT/'data/open/vessel_catalog.csv',DATA/'vessel_catalog.csv')
manifest=dict(seed='20261101',selection='hash of vessel ID, never model errors',catalog_rows=len(cat),eligible_vessels=len(pool),selected_vessels=64,pilot_vessels=16,full_rows=len(rows),pilot_rows=len(pilot),production_train_exclusions_unchanged=True,known_train_and_fewshot_targets_excluded=True,query_tables_are_seen_by_models=True,independent_holdout=False,synthetic_gain_qualifies_for_submission=False,files={p.name:sha(p) for p in DATA.iterdir() if p.is_file()})
(DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
