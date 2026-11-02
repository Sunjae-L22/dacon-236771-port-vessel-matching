from pathlib import Path
import csv,json,hashlib,ast,random,sys
sys.dont_write_bytecode=True
from channel_model import normalize,read
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent

def write(path,rows):
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 split=sys.argv[1] if len(sys.argv)>1 else 'dev';offset={'dev':0,'validation':64,'replication':128}[split]
 out=HERE/('cohort' if split=='dev' else 'cohort_'+split);out.mkdir(exist_ok=False)
 pool=json.loads((HERE/'COMPOSITIONAL_POOL.json').read_text());shots={normalize(x[0]) for x in json.loads((ROOT/'shared/kocat/fewshot_pairs.json').read_text())}
 tree=ast.parse((ROOT/'astra/arena_20261101/baseline/model/ko/model/src/llm_translit.py').read_text())
 for n in tree.body:
  if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='FEW_SHOT' for x in n.targets):shots.update(normalize(x[1]) for x in ast.literal_eval(n.value))
 pool=[r for r in pool if normalize(r['vessel_name']) not in shots];selected=pool[offset:offset+64];rows=[];truth=[]
 for c in selected:
  clean=normalize(c['query_ko']);vid=c['vessel_id']
  for variant in ('clean_keep','clean_nogt','delete','dup','swap','double'):
   q=clean;rng=random.Random('astra-composition-v1:'+vid+':'+variant)
   for op in (() if variant.startswith('clean') else ('delete','swap') if variant=='double' else (variant,)):
    if op=='swap':
     ps=[i for i in range(len(q)-1) if q[i]!=q[i+1]]
     if not ps:raise ValueError('No eligible swap')
     i=rng.choice(ps);q=q[:i]+q[i+1]+q[i]+q[i+2:]
    else:
     i=rng.randrange(len(q));q=q[:i]+q[i+1:] if op=='delete' else q[:i+1]+q[i]+q[i+1:]
   sid=f'C{len(rows):04d}';gt=c['gross_tonnage'] if variant=='clean_keep' else ''
   rows.append(dict(sample_id=sid,vessel_name=q,call_sign='',gross_tonnage=gt,international_gross_tonnage=gt,flag_name=c['flag_name'],vessel_type=c['vessel_type'],arrival_year='2025'))
   truth.append(dict(sample_id=sid,target_vessel_id=vid,variant=variant,canonical_name=c['vessel_name'],composed_clean=clean,source='TRAIN_TOKEN_COMPOSITION_SYNTHETIC'))
 for c in selected[:16]:
  sid=f'E{len(rows):04d}';rows.append(dict(sample_id=sid,vessel_name=c['vessel_name'],call_sign='',gross_tonnage=c['gross_tonnage'],international_gross_tonnage=c['gross_tonnage'],flag_name=c['flag_name'],vessel_type=c['vessel_type'],arrival_year='2025'));truth.append(dict(sample_id=sid,target_vessel_id=c['vessel_id'],variant='english_control',canonical_name=c['vessel_name'],composed_clean='',source='CATALOG_ENGLISH_CONTROL'))
 write(out/'test.csv',rows);write(out/'truth.csv',truth)
 seen={r['target_vessel_id'] for r in read(ROOT/'data/open/train.csv')};assert not seen.intersection(r['target_vessel_id'] for r in truth)
 manifest={'rows':len(rows),'ships':len(selected),'eligible_after_fewshot_exclusion':len(pool),'train_target_overlap':0,'fewshot_name_overlap':0,'source':'TRAIN_TOKEN_COMPOSITION_SYNTHETIC','queries_from_kocat_or_kanana':False,'human_independent_names':False,'full_model_independent_holdout':False,'limitations':['Bilingual token dictionary is derived from the same official train used by original models.','Composed names are synthetic and may not match natural whole-name transliteration.','No Public error count or Public uplift can be estimated from this set.','Noise conditions deliberately balanced, unlike unknown test distribution.'],'selection':f'IDs {offset} to {offset+63} under predeclared SHA256 order; no selection by model outputs','split':split,'strict_mass_share':.99,'protected_original110_gap':4,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.csv')}}
 (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
