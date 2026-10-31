# author: opus
"""Generator-like corruptions (char dup/delete/swap, renames) on vessel-split folds."""
import random
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cv import folds, load  # noqa: E402
import opus_lib as L  # noqa: E402
from opus_cv import PARAMS, fold_data  # noqa: E402


def gen_edit(s, n, rng):
    """The train generator's noise: duplicate, delete or swap characters (not spaces)."""
    s = list(s)
    for _ in range(n):
        pos = [i for i, ch in enumerate(s) if ch != " "]
        if len(pos) < 3:
            break
        i = rng.choice(pos)
        op = rng.random()
        if op < 0.4:
            s.insert(i, s[i])
        elif op < 0.7:
            del s[i]
        elif i + 1 < len(s) and s[i + 1] != " ":
            s[i], s[i + 1] = s[i + 1], s[i]
    return "".join(s)


def variants(pool_en, pool_ko, seed=0):
    rng = random.Random(seed)

    def name_op(fn):
        def g(r):
            r = dict(r)
            r["vessel_name"] = fn(str(r["vessel_name"]))
            return r
        return g

    def rename(r):
        r = dict(r)
        pool = pool_ko if L.is_hangul(r["vessel_name"]) else pool_en
        r["vessel_name"] = rng.choice(pool)
        return r

    def rename_brand(r):
        r = dict(r)
        pool = pool_ko if L.is_hangul(r["vessel_name"]) else pool_en
        w = str(r["vessel_name"]).split()
        r["vessel_name"] = (rng.choice(pool).split()[0] + " " + w[-1]) if len(w) > 1 else rng.choice(pool)
        return r

    def rename_new(r):
        r = dict(r)
        pool = pool_ko if L.is_hangul(r["vessel_name"]) else pool_en
        r["vessel_name"] = rng.choice(pool).split()[0] + " " + rng.choice(pool).split()[-1]
        return r

    def nogt(r):
        r = dict(r)
        r["gross_tonnage"] = r["international_gross_tonnage"] = np.nan
        return r

    return {
        "orig": lambda r: r,
        "edit1": name_op(lambda s: gen_edit(s, 1, rng)),
        "edit2": name_op(lambda s: gen_edit(s, 2, rng)),
        "edit3": name_op(lambda s: gen_edit(s, 3, rng)),
        "edit4": name_op(lambda s: gen_edit(s, 4, rng)),
        "rename": rename,
        "rename_brand": rename_brand,
        "rename_new": rename_new,
        "edit2+nogt": lambda r: nogt(name_op(lambda s: gen_edit(s, 2, rng))(r)),
        "MV": name_op(lambda s: "M/V " + s),
        "nospace": name_op(lambda s: s.replace(" ", "")),
    }


def main(tag, k=0, n_en=400):
    train, catalog = load()
    f = folds(train)
    Xtr, ytr, gtr, *_ = fold_data(train, catalog, f, k, tag)
    m = lgb.train(PARAMS, lgb.Dataset(Xtr, ytr, group=gtr, feature_name=L.FEATURES), 600)
    tr, va = train[f != k], train[f == k]
    ix = L.Index(catalog, tr)
    pool_en = [n.upper() for n in catalog["vessel_name"].sample(5000, random_state=1)]
    pool_ko = sorted(set(tr.loc[tr.vessel_name.map(L.is_hangul), "vessel_name"]))
    sets = {"ko": va[va.vessel_name.map(L.is_hangul)],
            "en": va[~va.vessel_name.map(L.is_hangul)].sample(n_en, random_state=0)}
    for name, fn in variants(pool_en, pool_ko).items():
        out = []
        for lang, sub in sets.items():
            ok = rec = 0
            for row in sub.to_dict("records"):
                row = fn(row)
                t = ix.id2i[row["target_vessel_id"]]
                idx, F = L.query_features(ix, row, L.query_candidates(ix, row))
                rec += t in idx
                ok += idx[int(np.argmax(m.predict(F)))] == t
            out.append(f"{lang} acc={ok / len(sub):.3f} rec={rec / len(sub):.3f}")
        print(f"{tag} {name:11s} " + " | ".join(out), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
