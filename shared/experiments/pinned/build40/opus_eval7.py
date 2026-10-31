# author: opus
"""Fold-0 check of alias features under vessel-split and row-split, with name corruptions."""
import sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from cv import folds, folds_rows, load
import opus_lib as L
from opus_cv import PARAMS, CACHE
from opus_stress2 import V

ALIAS = [L.FEATURES.index(c) for c in ("alias_exact", "alias_sim", "alias_n", "alias_amb")]
train, catalog = load()
for split, f in (("vessel", folds(train)), ("row", folds_rows(train))):
    k = 0
    tr, va = train[f != k], train[f == k]
    p = CACHE / f"v7_{split}_fold0.npz"
    ix = L.Index(catalog, tr)
    if p.exists():
        z = np.load(p); X, y, g = z["X"], z["y"], z["g"]
    else:
        X, y, g, _ = L.build(ix, pd_concat := __import__("pandas").concat(
            [L.augment_korean(tr, k), L.augment_corrupt(tr, k)], ignore_index=True), True)
        np.savez(p, X=X, y=y, g=g)
    models = {"with_alias": lgb.train(PARAMS, lgb.Dataset(X, y, group=g, feature_name=L.FEATURES), 600)}
    X0 = X.copy(); X0[:, ALIAS] = -1
    models["no_alias"] = lgb.train(PARAMS, lgb.Dataset(X0, y, group=g, feature_name=L.FEATURES), 600)
    sets = {"ko": va[va.vessel_name.map(L.is_hangul)], "en": va[~va.vessel_name.map(L.is_hangul)].sample(300, random_state=0)}
    for vname in ("orig", "typo2", "drop_last_word", "trunc60%", "typo2+nogt"):
        out = []
        for lang, sub in sets.items():
            ok = {m: 0 for m in models}
            for row in sub.to_dict("records"):
                row = V[vname](row); row["call_sign"] = None  # never rely on the call-sign override
                t = ix.id2i[row["target_vessel_id"]]
                idx, F = L.query_features(ix, row, L.query_candidates(ix, row))
                for mn, m in models.items():
                    FF = F if mn == "with_alias" else np.where(np.isin(np.arange(F.shape[1]), ALIAS), -1, F)
                    ok[mn] += idx[int(np.argmax(m.predict(FF)))] == t
            out.append(f"{lang} " + " ".join(f"{mn}={v/len(sub):.3f}" for mn, v in ok.items()))
        print(f"{split:6s} {vname:15s} " + " | ".join(out), flush=True)
