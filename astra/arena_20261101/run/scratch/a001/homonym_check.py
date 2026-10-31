import os,sys
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json,csv,unicodedata,hashlib,fcntl
from collections import defaultdict
from pathlib import Path
from probe import choose
HERE=Path(__file__).resolve().parent
ROOT=Path('/Users/sunjae/Desktop/항만')
os.environ['TMPDIR']=str(HERE)
with (ROOT/'astra/arena_20261101/CPU.lock').open('r') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    names=defaultdict(list)
    with (ROOT/'data/open/vessel_catalog.csv').open(encoding='utf-8-sig') as f:
        for c in csv.DictReader(f):
            key=' '.join(unicodedata.normalize('NFC',c['vessel_name']).upper().split())
            names[key].append(c['vessel_id'])
    ambiguous={vid for ids in names.values() if len(ids)>1 for vid in ids}
    out={'definition':'Same catalog vessel_name after NFC, upper case, whitespace collapse', 'folds':{}}
    for fold in (2,4):
        rs=[json.loads(s) for s in (HERE/f'fold{fold}_evaluated.jsonl').read_text().splitlines()]
        hs=[r for r in rs if r['truth'] in ambiguous]
        out['folds'][str(fold)]={'homonym_target_rows':len(hs),'homonym_target_vessels':len({r['truth'] for r in hs}),'homonym_target_changes':sum(r['base']!=r['new'] for r in hs)}
    row={'vessel_name':'세임','gross_tonnage':100,'international_gross_tonnage':100}
    pred=choose(row,'A',1,['A','B'],{'A':200.,'B':100.},{100.:['B'],200.:['A']})
    assert pred=='B'
    out['synthetic_counterexample']={'catalog':[{'id':'A','name':'SAME','GT':200},{'id':'B','name':'SAME','GT':100}], 'input':row,'base':'A','gap':1,'top5':['A','B'],'prediction':pred,'assumed_truth':'A','harm':True,'status':'mechanics only, not observed held-out harm'}
    out['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (HERE/'homonym_check.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print(json.dumps(out,ensure_ascii=False,indent=2))
