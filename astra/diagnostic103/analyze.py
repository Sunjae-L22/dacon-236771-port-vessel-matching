"""Read-only analysis of existing OOF and matched six-model fold2 caches."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from conditions import CONDITIONS, predicates

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FEATURES = json.loads((ROOT/'opus/stage2/stage1_features.json').read_text())
CAT = pd.read_csv(ROOT/'data/open/vessel_catalog.csv')
NAMES = CAT.vessel_name.fillna('').to_numpy()

def record(source, fold, meta, idx, F, scores, target):
    assert len(idx) >= 1
    second = 1 if len(idx) > 1 else 0
    gap = float(scores[0] - scores[1]) if second else 99.
    r = dict(source=source, fold=fold, variant=meta.get('variant', ''),
             vessel=meta.get('vessel', meta.get('target_vessel_id')),
             hard=gap < 4, gap=gap, correct=bool(idx[0] == target),
             second_correct=bool(second and idx[second] == target),
             top20=bool(target in idx[:20]))
    r.update(predicates(meta['vessel_name'], NAMES[idx[0]], NAMES[idx[second]], F[0],
                        F[second] if second else None, FEATURES))
    return r

def main():
    rows, inputs = [], []
    for fold in range(5):
        p = ROOT/f'opus/stage2/oof/fold{fold}.npz'
        z = np.load(p)
        meta = pd.read_csv(p.with_name(p.stem+'_queries.csv')).to_dict('records')
        q, cand, s, F, target = (z[k] for k in ('q','cand','s1','X','target'))
        starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
        ends = np.r_[starts[1:], len(q)]
        assert len(starts) == len(meta) == len(target)
        assert np.array_equal(q[starts], np.arange(len(meta)))
        assert F.shape[1] == len(FEATURES)
        for j, (lo, hi) in enumerate(zip(starts, ends)):
            assert np.all(s[lo:hi-1] >= s[lo+1:hi])
            assert CAT.vessel_id.iloc[target[j]] == meta[j]['vessel']
            rows.append(record('oof_1model',fold,meta[j],cand[lo:hi],F[lo:hi],s[lo:hi],target[j]))
        inputs.append(p)
    base = ROOT/'astra/incumbent97/fold2'
    p = next(base.glob('eval_benchmark_*.npz'))
    z = np.load(p)
    s = np.load(base/'benchmark_6models_all_scores.npy')
    meta = json.loads((base/'eval_benchmark_queries.json').read_text())
    assert len(s) == len(z['idx']) and len(meta) == len(z['groups'])
    X, ids, allowed, targets = (z[k] for k in ('X','idx','allowed','target'))
    lo = 0
    for j, n in enumerate(z['groups']):
        pool = np.arange(lo, lo+n)
        valid = pool[allowed[pool]]
        if not len(valid): valid = pool
        rank = np.array(sorted(valid, key=lambda i: (-float(s[i]), int(ids[i]))))[:20]
        if meta[j].get('source_language') == 'ko':
            assert CAT.vessel_id.iloc[targets[j]] == meta[j]['target_vessel_id']
            rows.append(record('fold2_6models',2,meta[j],ids[rank],X[rank],s[rank],targets[j]))
        lo += n
    inputs += [p, base/'benchmark_6models_all_scores.npy']
    df = pd.DataFrame(rows)
    df.to_csv(OUT/'row_diagnostics.csv', index=False)
    summary = []
    for source, group in df.groupby('source'):
        for scope, g in [('all',group), ('native',group[group.variant.str.startswith(('native','orig'))]),
                         ('synthetic',group[~group.variant.str.startswith(('native','orig'))])]:
            hard = g[g.hard]
            for cond in ('ALL_HARD',)+CONDITIONS:
                d = hard if cond == 'ALL_HARD' else hard[hard[cond]]
                summary.append(dict(source=source,scope=scope,condition=cond,total=len(g),hard_total=len(hard),
                    n=len(d),fraction_all=len(d)/len(g) if len(g) else None,
                    fraction_hard=len(d)/len(hard) if len(hard) else None,
                    correct=int(d.correct.sum()),errors=int((~d.correct).sum()),
                    accuracy=float(d.correct.mean()),top20_recall=float(d.top20.mean()),
                    top2_rescue=int(d.second_correct.sum()),top2_damage=int(d.correct.sum())))
    pd.DataFrame(summary).to_csv(OUT/'condition_summary.csv',index=False)
    detail=[]
    for keys,g in df.groupby(['source','fold','variant']):
        for cond in ('ALL_HARD',)+CONDITIONS:
            d=g[g.hard] if cond=='ALL_HARD' else g[g.hard & g[cond]]
            detail.append(dict(source=keys[0],fold=keys[1],variant=keys[2],condition=cond,n=len(d),
                               correct=int(d.correct.sum()),accuracy=float(d.correct.mean())))
    pd.DataFrame(detail).to_csv(OUT/'by_fold_variant.csv',index=False)
    hard=df[(df.source=='oof_1model') & df.hard]
    overlap=pd.DataFrame({a:{b:int((hard[a]&hard[b]).sum()) for b in CONDITIONS} for a in CONDITIONS})
    overlap.to_csv(OUT/'condition_overlap.csv')
    calibration=[]
    for source,g in df.groupby('source'):
        easy=g[~g.hard]
        native=g[g.variant.str.startswith(('native','orig'))]
        calibration.append(dict(source=source,n=len(g),easy_n=len(easy),easy_correct=int(easy.correct.sum()),
              easy_precision=float(easy.correct.mean()),native_n=len(native),native_easy_fraction=float((~native.hard).mean())))
    manifest=dict(calibration=calibration,inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        limits=['OOF seed0 one model, not production six-model ensemble',
                'fold2 six-model benchmark is a different synthetic query set, not independent five-fold validation',
                'raw Roman length is a proxy, not syllable/transliteration length',
                'mask scores identify correct counts only, not cohort size or error counts'])
    (OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print(pd.DataFrame(summary).query("scope == 'all'").to_string(index=False))
    print(json.dumps(calibration,indent=2))

if __name__=='__main__': main()
