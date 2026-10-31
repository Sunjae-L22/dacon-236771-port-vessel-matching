# author: opus
"""Harsher corruptions. Fold-0 v5 model, Korean + English validation rows."""
import random, re, sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from cv import folds, load
import opus_lib as L
from opus_cv import PARAMS, CACHE, fold_data

R = random.Random(0)
def _name(fn):
    def g(r): r = dict(r); r["vessel_name"] = fn(str(r["vessel_name"])); return r
    return g
def typo(s, n):
    s = list(s)
    for _ in range(n):
        if len(s) < 3: break
        i = R.randrange(len(s)); op = R.random()
        if op < .33: del s[i]
        elif op < .66: s.insert(i, R.choice("AEIOURNSTL") if s[i].isascii() else s[i])
        elif i + 1 < len(s): s[i], s[i+1] = s[i+1], s[i]
    return "".join(s)
def drop_word(s):
    w = s.split(); return " ".join(w[:-1]) if len(w) > 1 else s
def first_word(s):
    w = s.split(); return w[0] if len(w) > 1 else s
def trunc(s): return s[:max(3, int(len(s) * .6))]
def nogt(r): r = dict(r); r["gross_tonnage"] = r["international_gross_tonnage"] = np.nan; return r
def gtx(m):
    def g(r):
        r = dict(r)
        for c in ("gross_tonnage", "international_gross_tonnage"):
            if np.isfinite(L.positive(r[c])): r[c] = float(r[c]) * m
        return r
    return g
def all_missing(r): r = nogt(r); r["flag_name"] = np.nan; r["call_sign"] = None; return r
V = {
    "orig": lambda r: r, "typo1": _name(lambda s: typo(s, 1)), "typo2": _name(lambda s: typo(s, 2)),
    "typo3": _name(lambda s: typo(s, 3)), "drop_last_word": _name(drop_word), "first_word_only": _name(first_word),
    "drop_first_word": _name(lambda s: " ".join(s.split()[1:]) if len(s.split()) > 1 else s),
    "trunc60%": _name(trunc), "lower": _name(str.lower), "MV_prefix": _name(lambda s: "M/V " + s),
    "nogt": nogt, "gt*1.3": gtx(1.3), "gt*0.7": gtx(0.7), "gt*0.5": gtx(0.5), "allmeta_missing": all_missing,
    "typo2+nogt": lambda r: nogt(_name(lambda s: typo(s, 2))(r)),
}
if __name__ == "__main__":
    train, catalog = load(); f = folds(train); k = 0
    tag = sys.argv[1] if len(sys.argv) > 1 else "v5"
    Xtr, ytr, gtr, *_ = fold_data(train, catalog, f, k, tag)
    m = lgb.train(PARAMS, lgb.Dataset(Xtr, ytr, group=gtr, feature_name=L.FEATURES), 600)
    ix = L.Index(catalog, train[f != k]); va = train[f == k]
    sets = {"ko": va[va.vessel_name.map(L.is_hangul)], "en": va[~va.vessel_name.map(L.is_hangul)].sample(300, random_state=0)}
    for name, fn in V.items():
        out = []
        for lang, sub in sets.items():
            ok = rec = 0
            for row in sub.to_dict("records"):
                row = fn(row); t = ix.id2i[row["target_vessel_id"]]
                idx, F = L.query_features(ix, row, L.query_candidates(ix, row))
                rec += t in idx; ok += idx[int(np.argmax(m.predict(F)))] == t
            out.append(f"{lang} acc={ok/len(sub):.3f} rec={rec/len(sub):.3f}")
        print(f"{name:16s} " + " | ".join(out), flush=True)
