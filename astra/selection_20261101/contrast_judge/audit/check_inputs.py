"""Pure CPU audit of prepared routing, ordered cache keys and prompt schema."""
import json,csv,itertools,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;BASE=HERE.parent
with (ROOT/'data/open/vessel_catalog.csv').open(encoding='utf-8-sig') as f:cat={r['vessel_id']:r['vessel_name'] for r in csv.DictReader(f)}
def key(q,a,b):return json.dumps([q,a,b],ensure_ascii=False,separators=(',',':'))
report={}
for name,path in [('primary90',BASE),('validation174',BASE/'validation')]:
 rows=json.loads((path/'ROWS_WITHOUT_TRUTH.json').read_text());keys=json.loads((path/'PROMPT_KEYS.json').read_text());expected=set();same_name_pairs=0;instances=0
 assert len(keys)==len(set(keys));assert len({r['sample_id'] for r in rows})==len(rows)
 for r in rows:
  assert 'target' not in r and 'truth' not in r
  hard=any('가'<=c<='힣' for c in r['name']) and r['gap110'] is not None and r['gap110']<4
  assert hard==r['hard']
  if not hard:assert not r['pool'] and not r['comparisons'];continue
  pool=r['pool'];assert len(pool)==len(set(pool));assert r['cvote'] in pool
  if name=='primary90':assert r['cvote']==(r['baseline112'] if r['baseline112']==r['hsw'] else r['v20'])
  assert r['pool_names']==[cat[v] for v in pool]
  assert len(r['comparisons'])==len(pool)*(len(pool)-1)//2
  actual=set()
  for pair in r['comparisons']:
   a,b=pair['a'],pair['b'];assert a!=b and a in pool and b in pool
   actual.add(frozenset((a,b)))
   assert pair['ab']==key(r['name'],cat[a],cat[b]);assert pair['ba']==key(r['name'],cat[b],cat[a])
   same_name_pairs+=cat[a]==cat[b];instances+=2;expected.update((pair['ab'],pair['ba']))
  assert actual=={frozenset(p) for p in itertools.combinations(pool,2)}
 assert expected==set(keys)
 # Cache keys cannot depend on preceding row/order: reconstruction is a pure ordered serialization.
 assert {k for r in reversed(rows) for p in r['comparisons'] for k in (p['ab'],p['ba'])}==expected
 report[name]={'rows':len(rows),'hard_rows':sum(r['hard'] for r in rows),'unique_keys':len(keys),'ordered_pair_instances':instances,'same_name_unordered_pairs':same_name_pairs,'all_routing_pool_orientation_checks_pass':True,'prepared_source_sha256':hashlib.sha256((path/'ROWS_WITHOUT_TRUTH.json').read_bytes()).hexdigest()}
(HERE/'INPUT_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
