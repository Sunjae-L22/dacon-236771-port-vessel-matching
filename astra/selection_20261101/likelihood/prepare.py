from pathlib import Path
import sys,csv,json,hashlib,collections
sys.dont_write_bytecode=True
from scorer import ROOT,MODEL,FEW_SHOT,normalize_name,messages
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent
def read(p):
 with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def h(s):return hashlib.sha256(s.encode()).hexdigest()
def compact(s):return normalize_name(s).replace(' ','')
train={r['sample_id']:r for r in read(ROOT/'data/open/train.csv')};catalog={r['vessel_id']:r['vessel_name'] for r in read(ROOT/'data/open/vessel_catalog.csv')}
shotko={compact(k) for k,e in FEW_SHOT};shoten={compact(e) for k,e in FEW_SHOT};selected=[];counts={}
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
 chosen=sorted(eligible,key=lambda v:(h(f'kanana-pmi-v1:ship:{fold}:{v}'),v))[:24];assert len(chosen)==24
 for vid in chosen:
  c,ns=min(eligible[vid],key=lambda z:h(f'kanana-pmi-v1:clean:{fold}:{z[0]["query_id"]}'));n=min(ns,key=lambda p:h(f'kanana-pmi-v1:noise:{fold}:{p["query_id"]}'))
  for p in (c,n):
   candidates=p['candidates'][:20];assert candidates and candidates[0]['id']==p['baseline']
   selected.append({'fold':fold,'vessel':vid,'query_id':p['query_id'],'name':p['name'],'variant':p['variant'],'baseline':p['baseline'],'target':vid,'gap110':p['gate110_gap'],'hard':p['gate110_gap']<4,'candidates':candidates,'clean_native_verified':p is c,'source_sample_id':p['query_id'].split('::')[0]})
 counts[fold]={'eligible_vessels':len(eligible),'selected_vessels':len(chosen)}
assert len(selected)==96 and len({r['vessel'] for r in selected})==48
pairs=sorted({(normalize_name(c['name']),q) for r in selected if r['hard'] for c in r['candidates'] for q in (r['name'],'N/A')})
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,padding_side='left');prefix={q:tok.apply_chat_template(messages(q),tokenize=True,add_generation_prompt=True,enable_thinking=False) for _,q in pairs};maxlen=max(len(prefix[q])+len(tok.encode(e,add_special_tokens=False)) for e,q in pairs);max_target=max(len(tok.encode(e,add_special_tokens=False)) for e,q in pairs);assert maxlen<=512 and max_target<=32,(maxlen,max_target)
manifest={'rows':96,'vessels':48,'folds':counts,'hard_rows':sum(r['hard'] for r in selected),'hard_clean_rows':sum(r['hard'] and r['variant'].startswith('clean') for r in selected),'hard_noise_rows':sum(r['hard'] and not r['variant'].startswith('clean') for r in selected),'unique_pairs':len(pairs),'max_prompt_plus_target_tokens':maxlen,'max_target_tokens':max_target,'clean_source':'Exact human-written officialtrain names, verified sample_id/name/target; no alias-generated clean strings','selection_uses_prediction_correctness':False,'selected_by_target_in_pool':False,'truth_absent_top20':sum(r['target'] not in [c['id'] for c in r['candidates']] for r in selected),'fewshot_filter':'both Korean input and English truth names plus originalprimary flag','source_shas':{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in ['external/ksh/src/forward_name_likelihood.py','external/ksh/src/reverse_name_likelihood.py','external/ksh/src/llm_translit.py','data/open/train.csv']}}
(HERE/'SELECTED_ROWS.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n');(HERE/'MODEL_PAIRS.json').write_text(json.dumps(pairs,ensure_ascii=False)+'\n');(HERE/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps(manifest,ensure_ascii=False,indent=2))
