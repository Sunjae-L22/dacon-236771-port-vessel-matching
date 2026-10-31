"""Astra: immutable110 inference components and fold-safe contexts."""
from pathlib import Path
import os,sys,zipfile,json,pickle,hashlib
os.environ['KO_DICT']='1';os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
SRC=OUT/'src';SRC.mkdir(exist_ok=True)
with zipfile.ZipFile(ROOT/'submissions/110_submit.zip') as z:
    for n in ['opus_lib.py','ko_translit_argon.py','fuzzy_ext.py','ksh_hangul.py']:
        p=SRC/n;b=z.read('model/'+n)
        if p.exists():assert p.read_bytes()==b
        else:p.write_bytes(b)
# incumbent97 pins the same Korean library to79; verify before sharing its module.
sys.path.insert(0,str(ROOT/'astra'))
import incumbent97 as I
assert (SRC/'opus_lib.py').read_bytes()==(I.SRC/'opus_lib.py').read_bytes()
sys.path.insert(0,str(SRC))
from fuzzy_ext import FuzzyExt
import numpy as np,pandas as pd,lightgbm as lgb
B=I.B;SEEDS=I.SEEDS

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str));tmp.replace(p)

def context(scope):
    if scope.startswith('fold'):
        f=int(scope[4:]);tr,cat,fit,va,ix,llm,base=I.context(f)
        cache=base/'features';plan=json.loads((base/'plan.json').read_text());key='X'
    else:
        tr,cat=I.cv.load();fit=tr;va=None
        B.KA._UNIVERSAL_KO_SUBWORDS=I.UNIVERSAL
        ix=B.Index(cat,fit)
        llm=B.llm_korean_rows(fit,ROOT/'shared/experiments/pinned/build70/llm_korean_scored.csv')
        ix.ko=B.KA.KoMatcher(cat,pd.concat([fit,llm],ignore_index=True))
        base=ROOT/'astra/super98/full_train';cache=base/'features84';plan=json.loads((base/'plan.json').read_text());key='X84'
        with zipfile.ZipFile(ROOT/'submissions/110_submit.zip') as z:stats=pickle.loads(z.read('model/stats.pkl'))
        pairs=pd.concat([fit.loc[fit.vessel_name.map(B.is_hangul)],llm])[['vessel_name','target_vessel_id']].to_dict('records')
        assert pairs==stats['ko_pairs']
    return fit,va,cat,ix,FuzzyExt(ix,cat),base,cache,plan,key

def edit(name,rng,op=None):
    pos=[i for i,c in enumerate(name) if '가'<=c<='힣']
    if len(pos)<2:return name
    op=op or rng.choice(['delete','duplicate','swap'],p=[.35,.35,.30])
    if op=='swap':
        pairs=[i for i in pos if i+1<len(name) and '가'<=name[i+1]<='힣' and name[i]!=name[i+1]]
        if pairs:
            i=int(rng.choice(pairs));return name[:i]+name[i+1]+name[i]+name[i+2:]
        op='duplicate'
    i=int(rng.choice(pos))
    return name[:i]+name[i+1:] if op=='delete' else name[:i+1]+name[i]+name[i+1:]

def augment(fit,seed):
    rng=np.random.default_rng(113000+seed);out=[]
    for copy in range(3):
        for row in fit[fit.vessel_name.map(B.is_hangul)].to_dict('records'):
            r=dict(row)
            for _ in range(1 if rng.random()<.7 else 2):r['vessel_name']=edit(r['vessel_name'],rng)
            if rng.random()<.5:r['gross_tonnage']=r['international_gross_tonnage']=np.nan
            out.append(r)
    return out

def baseline_paths(base,cache,plan,seed):
    paths=[]
    for name in plan['seeds'][str(seed)]['order']:
        n=len(json.loads((base/'blocks'/f'{name}.json').read_text()))
        for lo in range(0,n,500):
            p=cache/name/f'{lo:06d}.npz';assert p.exists(),p;paths.append(p)
    return paths

def assert_cache_samples(base,cache,plan,key,ix):
    checked=0
    # Recompute first query of each distinct block; inspect actual first60 values and labels.
    names=list(dict.fromkeys(n for seed in SEEDS for n in plan['seeds'][str(seed)]['order']))
    for name in names:
        p=cache/name/'000000.npz'
        if not p.exists():continue
        row=json.loads((base/'blocks'/f'{name}.json').read_text())[0]
        idx,F=B.query_features(ix,row,B.query_candidates(ix,row));t=ix.id2i[row['target_vessel_id']]
        with np.load(p) as z:
            n=int(z['groups'][0]);assert n==len(idx)
            assert np.array_equal(z[key][:n,:60],F),name
            assert np.array_equal(z['y'][:n],np.array([i==t for i in idx])),name
        checked+=1
    return checked
