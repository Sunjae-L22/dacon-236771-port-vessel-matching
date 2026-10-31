# author: opus
"""Stress test: corrupt fold-0 validation rows and see where v4 breaks."""
import sys, unicodedata
from pathlib import Path
import lightgbm as lgb
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from cv import folds, load
import opus_lib as L
from opus_cv import PARAMS, CACHE

def nogt(r): r = dict(r); r["gross_tonnage"] = np.nan; r["international_gross_tonnage"] = np.nan; return r
def nospace(r): r = dict(r); r["vessel_name"] = r["vessel_name"].replace(" ", ""); return r
def nfd(r): r = dict(r); r["vessel_name"] = unicodedata.normalize("NFD", r["vessel_name"]); return r
def noflag(r): r = dict(r); r["flag_name"] = np.nan; return r
def gtnoise(r):
    r = dict(r)
    for c in ("gross_tonnage", "international_gross_tonnage"):
        if np.isfinite(L.positive(r[c])): r[c] = float(r[c]) * 1.15
    return r
VARIANTS = {"orig": lambda r: r, "nogt": nogt, "nospace": nospace, "nfd": nfd, "noflag": noflag, "gt+15%": gtnoise}

def main(tag="v5", k=0):
    train, catalog = load(); f = folds(train)
    z = np.load(CACHE / f"{tag}_fold{k}.npz", allow_pickle=True)
    m = lgb.train(PARAMS, lgb.Dataset(z["Xtr"], z["ytr"], group=z["gtr"], feature_name=L.FEATURES), 600)
    ix = L.Index(catalog, train[f != k])
    va = train[f == k]
    for lang, sub in (("ko", va[va.vessel_name.map(L.is_hangul)]), ("en", va[~va.vessel_name.map(L.is_hangul)].iloc[:400])):
        for name, fn in VARIANTS.items():
            ok = rec = 0
            for row in sub.to_dict("records"):
                row = fn(row); t = ix.id2i[row["target_vessel_id"]]
                idx, F = L.query_features(ix, row, L.query_candidates(ix, row))
                rec += t in idx; ok += idx[int(np.argmax(m.predict(F)))] == t
            print(f"{lang} {name:8s} n={len(sub)} acc={ok/len(sub):.3f} recall={rec/len(sub):.3f}", flush=True)

if __name__ == "__main__":
    main(*sys.argv[1:2])
