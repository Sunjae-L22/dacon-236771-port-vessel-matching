"""Fixed catalog-derived diagnostic cohort; labels stay in a separate CSV.
SYNTHETIC, TABLE-SEEN/STYLE-BIASED. Not Public-like accuracy or true holdout.
"""
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True


def norm(s): return ''.join(c.lower() for c in str(s) if c.isalnum())
def read_csv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def digest(s): return hashlib.sha256(s.encode()).hexdigest()


def build(root,out):
    root,out=Path(root),Path(out)
    paths={
        'catalog':root/'data/open/vessel_catalog.csv',
        'train':root/'data/open/train.csv',
        'table':root/'astra/arena_20261101/baseline/model/ko/model/catalog_ko.json',
        'excluded':root/'astra/arena_20261101/baseline/model/ko/model/train_vessels.json',
        'fewshot_source':root/'astra/arena_20261101/baseline/model/ko/model/src/llm_translit.py'}
    catalog=read_csv(paths['catalog']); train=read_csv(paths['train'])
    ko=json.loads(paths['table'].read_text())
    excluded=set(json.loads(paths['excluded'].read_text())) | {r['target_vessel_id'] for r in train}
    parsed=ast.parse(paths['fewshot_source'].read_text())
    pairs=None
    for n in parsed.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FEW_SHOT' for t in n.targets):
            pairs=ast.literal_eval(n.value)
    if pairs is None: raise ValueError('Fewshot provenance missing')
    few_ko={norm(p[0]) for p in pairs}; few_en={norm(p[1]) for p in pairs}
    eligible=[]
    for r in catalog:
        name=ko.get(r['vessel_name'],'')
        # Length limit is cohort design, not inference threshold or learned rule.
        compact=''.join(name.split())
        if r['vessel_id'] in excluded or norm(r['vessel_name']) in few_en or norm(name) in few_ko: continue
        if not 4<=len(compact)<=20 or not all('\uac00'<=c<='\ud7a3' for c in compact): continue
        eligible.append((digest('neural-rescue-stress-v1:'+r['vessel_id']),r,compact))
    selected=sorted(eligible)[:64]
    if len(selected)!=64: raise ValueError('Insufficient eligible unseen vessels')
    queries=[]; truths=[]
    for h,r,name in selected:
        n_edits=1+(int(h[:2],16)%2)
        noisy=name
        for step in range(n_edits):
            x=int(h[2+step*4:6+step*4],16)
            pos=x%(len(noisy)-1)
            op=(x//len(noisy))%3
            if op==0: noisy=noisy[:pos]+noisy[pos+1:]
            elif op==1: noisy=noisy[:pos]+noisy[pos]+noisy[pos:]
            elif noisy[pos]!=noisy[pos+1]: noisy=noisy[:pos]+noisy[pos+1]+noisy[pos]+noisy[pos+2:]
            else: noisy=noisy[:pos]+noisy[pos+1:]
        if noisy==name: noisy=name[:-1]
        for scenario,qname,gt in [('clean',name,r['gross_tonnage']),('noise',noisy,r['gross_tonnage']),
                                  ('missing_gt',name,''),('noise_missing_gt',noisy,'')]:
            sid='SYN_'+h[:16]+'_'+scenario
            queries.append({'sample_id':sid,'vessel_name':qname,'call_sign':'',
                            'gross_tonnage':gt,'international_gross_tonnage':'',
                            'vessel_type':'','flag_name':''})
            truths.append({'sample_id':sid,'target_vessel_id':r['vessel_id'],'scenario':scenario,
                           'planned_name_edits':n_edits if 'noise' in scenario else 0,
                           'label_scope':'SYNTHETIC,TABLE-SEEN/STYLE-BIASED'})
    out.mkdir(parents=True,exist_ok=True)
    for filename,rows in [('queries.csv',queries),('truth.csv',truths)]:
        if (out/filename).exists(): raise FileExistsError(out/filename)
        with (out/filename).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    manifest={'rows':len(queries),'vessels':len(selected),'eligible_catalog_vessels':len(eligible),
              'excluded_train_union':len(excluded),'fewshot_pairs':len(pairs),'status':'SYNTHETIC,TABLE-SEEN/STYLE-BIASED',
              'selection':'fixed SHA256,64 vessels; all Hangul 4..20 syllables; no model scores or labels used for selection',
              'metadata':'type/flag/callsign unavailable; international GT unavailable; GT official catalog or missing',
              'inference_inputs_contain_labels':False,'production_train_exclusions_changed':False,
              'source_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}}
    (out/'cohort_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    return manifest


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();print(json.dumps(build(args.root,args.out),ensure_ascii=False,indent=2))
