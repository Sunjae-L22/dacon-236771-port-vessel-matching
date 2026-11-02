"""Prepare untouched remaining eligible vessels; CPU only, no inference."""
from pathlib import Path
import sys,csv,json,hashlib,collections,ast,unicodedata
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent

def read(p):
 with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def h(s):return hashlib.sha256(s.encode()).hexdigest()
def normalize_name(s):
 s=unicodedata.normalize('NFKC',str(s)).upper()
 return ' '.join(''.join(c if c.isalnum() else ' ' for c in s).split())
def compact(s):return normalize_name(s).replace(' ','')
values={}
for n in ast.parse((ROOT/'external/ksh/src/llm_translit.py').read_text()).body:
 if isinstance(n,ast.Assign):
  for t in n.targets:
   if isinstance(t,ast.Name) and t.id=='FEW_SHOT':values[t.id]=ast.literal_eval(n.value)
FEW_SHOT=values['FEW_SHOT']
train={r['sample_id']:r for r in read(ROOT/'data/open/train.csv')};catalog={r['vessel_id']:r['vessel_name'] for r in read(ROOT/'data/open/vessel_catalog.csv')}
previous=json.loads((ROOT/'astra/selection_20261101/likelihood/SELECTED_ROWS.json').read_text())
excluded={r['vessel'] for r in previous}
shotko={compact(k) for k,e in FEW_SHOT};shoten={compact(e) for k,e in FEW_SHOT};selected=[];truth={};counts={}
for fold in (2,4):
 packets=[json.loads(s) for s in (ROOT/f'astra/byt5_base112_20261031/fold{fold}/packets112.jsonl').read_text().splitlines()];queries={q['query_id']:q for q in json.loads((ROOT/f'astra/noise113/fold{fold}/eval/queries.json').read_text())};by=collections.defaultdict(list)
 for p in packets:
  if p['primary'] and p['variant'].endswith('_missing'):by[p['target']].append(p)
 eligible={}
 for vid,ps in by.items():
  if compact(catalog[vid]) in shoten:continue
  clean=[p for p in ps if p['variant']=='clean_missing' and compact(p['name']) not in shotko]
  choices=[]
  for c in clean:
   origin=c['query_id'].split('::')[0];q=queries[c['query_id']];assert train[origin]['vessel_name']==q['vessel_name']==c['name'];assert train[origin]['target_vessel_id']==vid
   ns=[p for p in ps if not p['variant'].startswith('clean') and p['query_id'].split('::')[0]==origin and compact(p['name']) not in shotko]
   if ns:choices.append((c,ns))
  if choices:eligible[vid]=choices
 ordered=sorted(eligible,key=lambda v:(h(f'kanana-pmi-v1:ship:{fold}:{v}'),v))
 assert set(ordered[:24])=={r['vessel'] for r in previous if r['fold']==fold}
 chosen=ordered[24:]
 for vid in chosen:
  assert vid not in excluded
  c,ns=min(eligible[vid],key=lambda z:h(f'kanana-pmi-v1:clean:{fold}:{z[0]["query_id"]}'));n=min(ns,key=lambda p:h(f'kanana-pmi-v1:noise:{fold}:{p["query_id"]}'))
  for p in (c,n):
   candidates=p['candidates'][:3];assert candidates and candidates[0]['id']==p['baseline']
   origin=p['query_id'].split('::')[0];assert train[origin]['target_vessel_id']==vid
   selected.append({'fold':fold,'query_id':p['query_id'],'name':p['name'],'variant':p['variant'],'incumbent':p['baseline'],'gap110':p['gate110_gap'],'hard':p['gate110_gap']<4 and any('가'<=x<='힣' for x in p['name']),'candidates':[{'id':x['id'],'name':x['name']} for x in candidates],'clean_native_verified':p is c,'source_sample_id':origin})
   truth[p['query_id']]={'target':vid,'fold':fold,'variant':p['variant'],'source_sample_id':origin,'source_native_name':train[origin]['vessel_name']}
 counts[fold]={'eligible_vessels':len(eligible),'previous_excluded_vessels':24,'remaining_vessels':len(chosen)}
assert len(selected)==174 and len(set(x['target'] for x in truth.values()))==87
assert len({r['query_id'] for r in selected})==174
assert all('target' not in r and 'vessel' not in r for r in selected)
pairs=sorted({(r['name'],a['name'],b['name']) for r in selected if r['hard'] for a in r['candidates'] for b in r['candidates'] if a['id']!=b['id']})
manifest={'rows':174,'vessels':87,'folds':counts,'hard_rows':sum(r['hard'] for r in selected),'hard_clean_rows':sum(r['hard'] and r['variant'].startswith('clean') for r in selected),'hard_noise_rows':sum(r['hard'] and not r['variant'].startswith('clean') for r in selected),'unique_ordered_prompts':len(pairs),'ordered_pair_instances':sum(len(r['candidates'])*(len(r['candidates'])-1) for r in selected if r['hard']),'pool':'OOF112 top3 only, unlike primary actualC_vote90 (no v20/HSW voters)','truth_absent_top3_retained':sum(truth[r['query_id']]['target'] not in [c['id'] for c in r['candidates']] for r in selected),'primary90_data_read':False,'selection_by_correctness_or_truth_in_pool':False,'all_remaining_eligible_used':True,'previous_pmi_vessels_overlap':0,'model_inputs_only':'name and each ordered pair of English names; truth file separate','clean_provenance':'Exact official human-written name, verified original sample_id/name/target; metadata missing only','noise_provenance':'Same source as clean; unchanged kanana-pmi-v1 hash choice from existing fixed augmentation','GPU_executed':False,'source_hash_method':'sha256 raw read_bytes; no newline normalization','source_shas':{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in ['astra/selection_20261101/likelihood/prepare.py','astra/selection_20261101/likelihood/SELECTED_ROWS.json','data/open/train.csv']}}
for name,obj in [('MODEL_ROWS.json',selected),('TRUTH.json',truth),('MODEL_PAIRS.json',pairs),('MANIFEST.json',manifest)]:
 (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
