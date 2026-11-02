from pathlib import Path
import sys,csv,json,collections,re,hashlib,random,ast
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent

def read(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def main():
 cat=read(ROOT/'data/open/vessel_catalog.csv');train=read(ROOT/'data/open/train.csv');byid={r['vessel_id']:r for r in cat};seen={r['target_vessel_id'] for r in train};mapping=collections.defaultdict(lambda:collections.defaultdict(set));pairs=[];used=set()
 for r in train:
  vid=r['target_vessel_id'];en=byid[vid]['vessel_name'].upper().split();ko=r['vessel_name'].split()
  if (vid,tuple(ko)) in used:continue
  used.add((vid,tuple(ko)))
  if len(en)!=len(ko) or not 2<=len(en)<=4 or not all(re.fullmatch('[A-Z]+',w) for w in en) or not all(re.fullmatch('[가-힣]+',w) for w in ko):continue
  pairs.append((vid,en,ko))
  for a,b in zip(en,ko):mapping[a][b].add(vid)
 lex={}
 for en,variants in mapping.items():
  total=set().union(*variants.values());ordered=sorted(variants,key=lambda s:(-len(variants[s]),s));winner=ordered[0]
  if len(variants[winner])>=2 and len(variants[winner])/len(total)>=.75:lex[en]={'ko':winner,'support':len(variants[winner]),'total_support':len(total),'vessels':sorted(variants[winner])}
 # Clean compositional evaluation entities must be absent from all supervised train targets.
 dupe=collections.Counter(r['vessel_name'].upper() for r in cat);pool=[]
 for r in cat:
  words=r['vessel_name'].upper().split()
  if r['vessel_id'] in seen or dupe[r['vessel_name'].upper()]!=1 or not 2<=len(words)<=4 or not all(w in lex for w in words):continue
  ko=' '.join(lex[w]['ko'] for w in words)
  if len(ko.replace(' ',''))<4:continue
  pool.append(dict(r,query_ko=ko))
 pool.sort(key=lambda r:hashlib.sha256(('astra-real-token-v1:'+r['vessel_id']).encode()).digest())
 manifest={'method':'real train bilingual equal-token-position lexicon, >=2 ship support, >=75% unique-vessel agreement; no KoCat or Kanana query access','aligned_real_name_pairs':len(pairs),'supported_tokens':len(lex),'eligible_unseen_unique_english_names':len(pool),'selection':'fixed ID SHA256 independent of model predictions','evidence':'TRAIN_TOKEN_COMPOSITION_SYNTHETIC; NOT human independent test; no target-entity overlap with supervised train','lexicon_source_can_be_shared_with_baseline':True}
 (OUT/'TOKEN_LEXICON.json').write_text(json.dumps(lex,ensure_ascii=False,indent=2));(OUT/'COMPOSITIONAL_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));(OUT/'COMPOSITIONAL_POOL.json').write_text(json.dumps(pool,ensure_ascii=False,indent=2));print(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps([{k:r[k] for k in ('vessel_id','vessel_name','query_ko')} for r in pool[:15]],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
