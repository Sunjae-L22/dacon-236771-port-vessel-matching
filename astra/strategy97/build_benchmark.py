"""Independent train/catalog-only stress protocol, fixed outer fold2.

This intentionally does not import any existing name/tonnage augmentation helper.
All learned corruption inventories and label transitions use outer-fit rows only.
"""
from pathlib import Path
import sys,json,hashlib,re,unicodedata,collections,math
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'shared/src'));import cv
SEED=970219;FOLD=2
CHO='ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ'
JUNG='ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'
JONG=['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
QUERY_COLUMNS=['sample_id','vessel_name','call_sign','gross_tonnage','international_gross_tonnage','flag_name','vessel_type','arrival_year']

def stable_int(*parts):return int.from_bytes(hashlib.sha256('|'.join(map(str,(SEED,*parts))).encode()).digest()[:8],'little')
def rng_for(sample,variant):return np.random.default_rng(stable_int(sample,variant))
def is_ko(s):return isinstance(s,str) and any('\uac00'<=c<='\ud7a3' for c in unicodedata.normalize('NFC',s))
def positive(x):
    try:return math.isfinite(float(x)) and float(x)>0
    except (TypeError,ValueError):return False
def equal(a,b):
    if pd.isna(a) and pd.isna(b):return True
    return a==b

def representative(frame):
    result=[]
    for target,g in frame.groupby('target_vessel_id',sort=True):
        result.append(min(g.to_dict('records'),key=lambda r:stable_int('representative',target,r['sample_id'])))
    return result

def inventories(fit):
    components=[collections.Counter(),collections.Counter(),collections.Counter()];ascii_chars=collections.Counter()
    for name in fit.vessel_name:
        for ch in unicodedata.normalize('NFC',str(name)):
            if '\uac00'<=ch<='\ud7a3':
                code=ord(ch)-0xac00
                for j,value in enumerate((code//588,(code%588)//28,code%28)):components[j][value]+=1
            elif ch.isascii() and ch.isalpha():ascii_chars[ch.upper()]+=1
    transitions={}
    for field in ['flag_name','vessel_type']:
        counts=collections.defaultdict(collections.Counter)
        for target,g in fit.groupby('target_vessel_id',sort=False):
            values=sorted(set(str(x) for x in g[field].dropna()))
            for a in values:
                for b in values:
                    if a!=b:counts[a][b]+=1/max(len(values)-1,1)
        transitions[field]={a:dict(sorted(b.items())) for a,b in sorted(counts.items())}
    return components,ascii_chars,transitions

def jamo_edit(name,rng,components,count):
    chars=list(unicodedata.normalize('NFC',name));locations=[i for i,c in enumerate(chars) if '\uac00'<=c<='\ud7a3']
    if not locations:return name
    chosen=rng.choice(locations,size=min(count,len(locations)),replace=False)
    for i in chosen:
        code=ord(chars[i])-0xac00;parts=[code//588,(code%588)//28,code%28];component=int(rng.integers(3));options=sorted(set(components[component])-{parts[component]})
        if not options:continue
        weights=np.array([components[component][v] for v in options],float);weights/=weights.sum();parts[component]=int(rng.choice(options,p=weights));chars[i]=chr(0xac00+588*parts[0]+28*parts[1]+parts[2])
    return ''.join(chars)

def compatibility_jamo(name):
    result=[]
    for char in unicodedata.normalize('NFC',name):
        if '\uac00'<=char<='\ud7a3':
            code=ord(char)-0xac00;result.extend([CHO[code//588],JUNG[(code%588)//28],JONG[code%28]])
        else:result.append(char)
    return ''.join(result)

def name_change(name,mode,rng,components,donors,ascii_chars):
    name=unicodedata.normalize('NFC',str(name));words=name.split();chars=list(name)
    if mode=='nfd':return unicodedata.normalize('NFD',name)
    if mode=='compat_jamo':return compatibility_jamo(name)
    if mode=='jamo1':return jamo_edit(name,rng,components,1)
    if mode=='jamo2':return jamo_edit(name,rng,components,2)
    if mode=='delete_syllable':
        positions=[i for i,c in enumerate(chars) if '\uac00'<=c<='\ud7a3']
        if len(positions)>1:del chars[int(rng.choice(positions))]
        return ''.join(chars)
    if mode=='swap_adjacent':
        positions=[i for i in range(len(chars)-1) if chars[i]!=' ' and chars[i+1]!=' ' and chars[i]!=chars[i+1]]
        if positions:i=int(rng.choice(positions));chars[i],chars[i+1]=chars[i+1],chars[i]
        return ''.join(chars)
    if mode.startswith('prefix'):
        fraction=float(mode[6:])/100;return name[:max(2,int(math.ceil(len(name)*fraction)))].rstrip()
    if mode=='suffix50':return name[-max(2,int(math.ceil(len(name)*.5))):].lstrip()
    if mode=='first_token':return words[0] if words else name
    if mode=='last_token':return words[-1] if words else name
    if mode=='drop_first':return ' '.join(words[1:]) if len(words)>1 else name
    if mode=='drop_middle':
        if len(words)>=3:del words[int(rng.integers(1,len(words)-1))]
        return ' '.join(words)
    if mode=='join_spaces':return ''.join(words)
    if mode=='permute_tokens':
        if len(words)>1:
            order=np.arange(len(words));rng.shuffle(order)
            if np.array_equal(order,np.arange(len(words))):order=np.roll(order,1)
            return ' '.join(words[i] for i in order)
        return name
    if mode=='rename_owner':
        if len(words)<2:return name
        options=[x.split()[0] for x in donors if x.split() and x.split()[0]!=words[0]]
        return str(options[int(rng.integers(len(options)))])+' '+' '.join(words[1:]) if options else name
    if mode=='rename_full':
        options=[x for x in donors if x!=name];return options[int(rng.integers(len(options)))] if options else name
    if mode=='ascii_edit2':
        alphabet=sorted(ascii_chars);weights=np.array([ascii_chars[c] for c in alphabet],float);weights/=weights.sum()
        for _ in range(2):
            positions=[i for i,c in enumerate(chars) if c.isascii() and c.isalnum()]
            if not positions:break
            i=int(rng.choice(positions));op=int(rng.integers(3))
            if op==0 and len(chars)>2:del chars[i]
            elif op==1:chars.insert(i,str(rng.choice(alphabet,p=weights)))
            else:
                choices=[c for c in alphabet if c!=chars[i].upper()];chars[i]=choices[int(rng.integers(len(choices)))]
        return ''.join(chars)
    return name

def scale_value(row,key,factor):
    if positive(row.get(key)):row[key]=float(row[key])*float(factor)
def factor(rng):
    return float(np.exp(rng.uniform(np.log(.65),np.log(.85)))) if rng.random()<.5 else float(np.exp(rng.uniform(np.log(1.15),np.log(1.45))))
def tonnage_change(row,mode,rng):
    gt='gross_tonnage';igt='international_gross_tonnage'
    if mode=='gt_only':scale_value(row,gt,factor(rng))
    elif mode=='igt_only':scale_value(row,igt,factor(rng))
    elif mode=='independent':scale_value(row,gt,factor(rng));scale_value(row,igt,factor(rng))
    elif mode=='common':
        f=factor(rng);scale_value(row,gt,f);scale_value(row,igt,f)
    elif mode=='opposite':
        low=float(rng.uniform(.65,.85));high=float(rng.uniform(1.15,1.45));a,b=(low,high) if rng.random()<.5 else (high,low);scale_value(row,gt,a);scale_value(row,igt,b)
    elif mode=='drop_gt':row[gt]=np.nan
    elif mode=='drop_igt':row[igt]=np.nan

def label_change(row,field,rng,transitions):
    options=transitions[field].get(str(row[field]),{})
    if options:
        labels=sorted(options);weights=np.array([options[x] for x in labels],float);weights/=weights.sum();row[field]=str(rng.choice(labels,p=weights))

def main():
    train,catalog=cv.load();fold=cv.folds(train);fit=train[fold!=FOLD].copy();held=train[fold==FOLD].copy();assert not set(fit.target_vessel_id)&set(held.target_vessel_id)
    components,ascii_chars,transitions=inventories(fit);donors=sorted(set(fit.loc[fit.vessel_name.map(is_ko),'vessel_name'].astype(str)));ko=representative(held[held.vessel_name.map(is_ko)])
    english=sorted(representative(held),key=lambda row:stable_int('english_cohort',row['target_vessel_id']))[:128];catalog_name=dict(zip(catalog.vessel_id,catalog.vessel_name));out=[]
    specs=[]
    def spec(name,family,namemode='',tonnage='',label='',cohort='ko_stress',priority='core'):
        specs.append(dict(variant=name,family=family,name=namemode,tonnage=tonnage,label=label,cohort=cohort,priority=priority))
    spec('ko_nfd','unicode','nfd',priority='invariance');spec('ko_compat_jamo','unicode','compat_jamo',priority='diagnostic')
    for name in ['jamo1','jamo2','delete_syllable','swap_adjacent']:spec('ko_'+name,'local_name_damage',name)
    for name in ['prefix35','prefix70','suffix50','first_token','last_token','drop_middle']:spec('ko_'+name,'partial_name',name)
    for name in ['join_spaces','permute_tokens']:spec('ko_'+name,'name_format',name)
    spec('ko_rename_owner','rename','rename_owner');spec('ko_rename_full','rename','rename_full',priority='diagnostic')
    for mode in ['gt_only','igt_only','independent','common','opposite','drop_gt','drop_igt']:spec('ko_clean_'+mode,'tonnage_channel',tonnage=mode)
    for mode in ['gt_only','igt_only','independent','common','opposite']:spec('ko_prefix70_'+mode,'joint_name_tonnage','prefix70',mode)
    for field in ['flag_name','vessel_type']:
        spec('ko_clean_transition_'+field,'metadata_transition',label=field,priority='diagnostic')
        spec('ko_prefix70_transition_'+field,'joint_name_metadata','prefix70',label=field,priority='diagnostic')
    for mode in ['clean','ascii_edit2','prefix40','drop_first','last_token','permute_tokens']:spec('catalog_en_'+mode,'catalog_english',namemode=mode,cohort='catalog_en',priority='auxiliary')
    spec('catalog_en_prefix40_independent','catalog_english','prefix40','independent',cohort='catalog_en',priority='auxiliary');spec('catalog_en_gt_only','catalog_english',tonnage='gt_only',cohort='catalog_en',priority='auxiliary')
    def append(base,row,variant,family,cohort,priority):
        source_language='ko' if is_ko(base['vessel_name']) else 'en';row['call_sign']='UNSEEN';row['query_id']=f"B97_F2::{variant}::{base['sample_id']}";row['source_sample_id']=base['sample_id'];row['original_vessel_name']=base['vessel_name'];row['variant']=variant;row['family']=family;row['cohort']=cohort;row['priority']=priority;row['source_language']=source_language;row['effective_name_change']=str(row['vessel_name'])!=str(base['vessel_name']);row['effective_tonnage_change']=any(not equal(row[k],base[k]) for k in ['gross_tonnage','international_gross_tonnage']);row['effective_label_change']=any(not equal(row[k],base[k]) for k in ['flag_name','vessel_type']);row['any_effective_change']=row['effective_name_change'] or row['effective_tonnage_change'] or row['effective_label_change'];out.append(row)
    for row in held.to_dict('records'):append(row,dict(row),'native_all_rows','native_anchor','native','anchor')
    for s in specs:
        cohort=ko if s['cohort']=='ko_stress' else english
        for original in cohort:
            base=dict(original);row=dict(original);rng=rng_for(original['sample_id'],s['variant'])
            if s['cohort']=='catalog_en':base['vessel_name']=str(catalog_name[original['target_vessel_id']]);row['vessel_name']=base['vessel_name']
            row['vessel_name']=name_change(row['vessel_name'],s['name'],rng,components,donors,ascii_chars);tonnage_change(row,s['tonnage'],rng)
            if s['label']:label_change(row,s['label'],rng,transitions)
            append(base,row,s['variant'],s['family'],s['cohort'],s['priority'])
    data=pd.DataFrame(out);assert data.query_id.is_unique;assert set(data.target_vessel_id)<=set(held.target_vessel_id);assert not set(data.target_vessel_id)&set(fit.target_vessel_id)
    path=OUT/'benchmark_fold2.csv';data.to_csv(path,index=False)
    coverage=data.groupby(['cohort','family','variant'],sort=False).agg(rows=('query_id','size'),vessels=('target_vessel_id','nunique'),changed=('any_effective_change','sum'),name_changed=('effective_name_change','sum'),tonnage_changed=('effective_tonnage_change','sum'),label_changed=('effective_label_change','sum')).reset_index();coverage.to_csv(OUT/'coverage.csv',index=False)
    learned=dict(jamo_component_counts=[dict(sorted(c.items())) for c in components],ascii_character_counts=dict(sorted(ascii_chars.items())),label_transition_counts=transitions,donor_names=donors);(OUT/'fit_only_corruption_inventory.json').write_text(json.dumps(learned,ensure_ascii=False,indent=2))
    protocol=dict(seed=SEED,outer_fold=FOLD,fit_vessels=fit.target_vessel_id.nunique(),heldout_vessels=held.target_vessel_id.nunique(),fit_rows=len(fit),heldout_rows=len(held),korean_stress_vessels=len(ko),english_catalog_vessels=len(english),query_rows=len(data),query_id_column='query_id',allowed_query_columns=QUERY_COLUMNS,target_column='target_vessel_id',call_sign='UNSEEN for every query; no row grouping permitted at prediction time',corruption_source='All label transitions, character inventories, and Korean donor names use outer-fit rows only; English candidate names come from allowed catalog.',tonnage_noise='Independent fixed-seed log-uniform errors in [.65,.85] or [1.15,1.45]; separate GT-only, IGT-only, independent, common-factor, opposite-sign, one-column missing channels. These are stress scenarios, not claims about hidden-test prevalence.',representative_selection='One Korean row per heldout vessel chosen by fixed SHA256 hash, not first-row ordering or model results.',specs=specs,reporting_contract=['Native Korean and native English are separate anchors; report row and vessel accuracy.', 'Report each variant separately with all-row accuracy, changed-only accuracy, target candidate recall, top20 recall, conditional reranking accuracy, and vessel-bootstrap uncertainty.', 'Never pool catalog-English auxiliary rows into Korean accuracy.', 'Do not average unchanged transforms into the changed-only result; report unsupported label-transition coverage.', 'No overall equal-weight stress average is a Public-score estimate or sole adoption gate.', 'NFD is a representation-invariance check, not extra evidence of score improvement.', 'Compatibility jamo, full rename, and empirical label transitions are diagnostic scenarios with unknown hidden-test prevalence.', 'Preserve original native accuracy; changes must be compared to the matched79 recipe, including dictionary, LLM coverage and seed count.', 'Freeze method choices before reading fold2 scores; this project has used fold2 vessels in previous experiments, so this is a new benchmark on reused heldout vessels, not a pristine global holdout.'],train_sha256=hashlib.sha256((ROOT/'data/open/train.csv').read_bytes()).hexdigest(),catalog_sha256=hashlib.sha256((ROOT/'data/open/vessel_catalog.csv').read_bytes()).hexdigest(),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),benchmark_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2));print(json.dumps({k:protocol[k] for k in ['seed','outer_fold','fit_vessels','heldout_vessels','korean_stress_vessels','english_catalog_vessels','query_rows','benchmark_sha256']},indent=2));print(coverage.to_string(index=False))

if __name__=='__main__':main()
