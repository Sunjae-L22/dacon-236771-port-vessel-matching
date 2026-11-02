from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import csv,json,pickle,hashlib,unicodedata,datetime,math
from runtime import ROOT,MODEL,AutoTokenizer,messages,key
HERE=Path(__file__).resolve().parent
BASE=ROOT/'astra/selection_20261101';DATA=ROOT/'astra/new_direction_20261101/local_cvote/eval'
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def preds(p):return {r['sample_id']:r['target_vessel_id'] for r in read(p)}
test=read(DATA/'test.csv');enrows=read(DATA/'en_pred.csv');en={r['sample_id']:r for r in enrows};hs=preds(DATA/'hs_pred.csv');v20=preds(BASE/'v20_hard_pred.csv');catrows=read(ROOT/'data/open/vessel_catalog.csv');cat={r['vessel_id']:r['vessel_name'] for r in catrows};order={r['vessel_id']:i for i,r in enumerate(catrows)}
features={r['sample_id']:r for r in pickle.loads((ROOT/'astra/selection_20261101/pairwise/production112_features.pkl').read_bytes())};excluded=set(json.loads((ROOT/'astra/arena_20261101/baseline/model/ko/model/train_vessels.json').read_text()))
rows=[];prompts=set();ncomparisons=0
for q in test:
 sid=q['sample_id'];name=unicodedata.normalize('NFC',q['vessel_name']);e=en[sid]['target_vessel_id'];h=hs[sid];gap=float(en[sid]['ko_gap']) if en[sid]['ko_gap'] else math.nan;ko=any('가'<=c<='힣' for c in name);hard=bool(ko and gap<4);cv=e if not hard or e==h else v20[sid]
 r={'sample_id':sid,'name':name,'hard':hard,'gap110':gap if math.isfinite(gap) else None,'baseline112':e,'hsw':h,'v20':v20.get(sid),'cvote':cv,'pool':[],'comparisons':[]}
 if hard:
  f=features[sid];assert f['gap110']==gap and f['baseline112']==e
  pool=sorted(set(f['ids'][:3]+[v20[sid],h,cv]),key=order.get);assert 2<=len(pool)<=5 and not set(pool)&excluded
  r['pool']=pool;r['pool_names']=[cat[v] for v in pool];r['original112_ranks']={v:(f['ids'].index(v)+1 if v in f['ids'] else None) for v in pool}
  for i,a in enumerate(pool):
   for b in pool[i+1:]:
    ab=key(name,cat[a],cat[b]);ba=key(name,cat[b],cat[a]);prompts.update((ab,ba));r['comparisons'].append({'a':a,'b':b,'ab':ab,'ba':ba});ncomparisons+=1
 rows.append(r)
assert len(rows)==400 and sum(r['hard'] for r in rows)==198
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,padding_side='left');labels={s:tok.encode(s,add_special_tokens=False) for s in ['A','B']};assert all(len(v)==1 for v in labels.values());maxlen=0
for k in sorted(prompts):
 q,a,b=json.loads(k);m=messages(q,a,b);prefix=tok.apply_chat_template(m,tokenize=True,add_generation_prompt=True,enable_thinking=False);maxlen=max(maxlen,len(prefix))
 for s in ['A','B']:
  full=tok.apply_chat_template(m+[{'role':'assistant','content':s}],tokenize=True,add_generation_prompt=False,enable_thinking=False);assert full[:len(prefix)]==prefix and full[len(prefix):len(prefix)+1]==labels[s]
pad=256 if maxlen<=256 else 512;assert maxlen<=pad
rate=json.loads((ROOT/'astra/selection_20261101/likelihood/FOLLOWUP_TIME_ESTIMATE.json').read_text())['steady_state_seconds_per_pair_max'];bound=len(prompts)*rate*1.25+120;now=datetime.datetime.now(datetime.timezone.utc);deadline=datetime.datetime(2026,10,31,21,45,tzinfo=datetime.timezone.utc)
report={'dataset_rows':400,'hard_rows':198,'unordered_comparisons':ncomparisons,'ordered_before_cache':2*ncomparisons,'unique_ordered_prompts':len(prompts),'prompt_lengths_max':maxlen,'fixed_pad_length':pad,'A_B_token_ids':{s:t[0] for s,t in labels.items()},'assistant_prefix_boundary_verified_for_all_prompts':True,'truth_file_read':False,'conservative512_based_seconds_including25percent_and_load':bound,'estimated_half_length_seconds_including25percent_and_load':len(prompts)*rate*(pad/512)*1.25+120,'conservative_finish_kst':(now+datetime.timedelta(seconds=bound)).astimezone(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),'fits_0645_conservative':now+datetime.timedelta(seconds=bound)<=deadline,'new_retrieval':False,'all_pool_IDs_outside_training_exclusion':True,'protocol_sha256':hashlib.sha256((HERE/'PROTOCOL.json').read_bytes()).hexdigest()}
(HERE/'ROWS_WITHOUT_TRUTH.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n');(HERE/'PROMPT_KEYS.json').write_text(json.dumps(sorted(prompts),ensure_ascii=False)+'\n');(HERE/'MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
