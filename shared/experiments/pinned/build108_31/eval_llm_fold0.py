# author: opus
"""Fold 0: v14 recipe with vs without LLM Korean rows, on identical corrupted Korean validation rows."""
import sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from cv import folds, load
import opus_lib as L
from opus_cv import PARAMS, fold_data
import opus_stress2 as S2, opus_stress3 as S3, opus_stress4 as S4

tr, ca = load(); f = folds(tr); k = 0
trf, va = tr[f != k], tr[f == k]
llm = L.llm_korean_rows(trf, Path(__file__).parent / "llm_korean_scored.csv")   # train-fold vessels only
print("LLM rows (train fold):", len(llm), flush=True)

base = L.Index(ca, trf)
Xb, yb, gb, *_ = fold_data(tr, ca, f, k, "v14")                                  # same code, cached
mb = lgb.train(PARAMS, lgb.Dataset(Xb, yb, group=gb, feature_name=L.FEATURES), 600)

ixl = L.Index(ca, trf); ixl.ko = L.KA.KoMatcher(ca, pd.concat([trf, llm], ignore_index=True))
Xl, yl, gl, _ = L.build(ixl, pd.concat([L.augment_korean(trf, k), L.augment_corrupt(trf, k), L.augment_rename(trf, k), llm],
                                       ignore_index=True), True)
ml = lgb.train(PARAMS, lgb.Dataset(Xl, yl, group=gl, feature_name=L.FEATURES), 600)
print("trained both", flush=True)

pool_en = [n.upper() for n in ca.vessel_name.sample(5000, random_state=1)]
pool_ko = sorted(set(trf.loc[trf.vessel_name.map(L.is_hangul), "vessel_name"]))
V3 = S3.variants(pool_en, pool_ko, seed=0)
tests = {"orig": lambda r, c: r, "edit1": lambda r, c: V3["edit1"](r), "edit2": lambda r, c: V3["edit2"](r),
         "jamo1": S4.V["jamo1"], "reorder": S4.V["reorder"], "rename": lambda r, c: V3["rename"](r),
         "rename_brand": lambda r, c: V3["rename_brand"](r), "rename_new": lambda r, c: V3["rename_new"](r),
         "drop_last": lambda r, c: S2.V["drop_last_word"](r), "trunc60": lambda r, c: S2.V["trunc60%"](r)}
rows = va[va.vessel_name.map(L.is_hangul)].to_dict("records")
for tn, fn in tests.items():
    ok = {"base": 0, "llm": 0}
    for r in rows:
        q = fn(dict(r), base.names_raw[base.id2i[r["target_vessel_id"]]])
        t = base.id2i[r["target_vessel_id"]]
        for name, ix, m in (("base", base, mb), ("llm", ixl, ml)):
            idx, F = L.query_features(ix, q, L.query_candidates(ix, q))
            s = m.predict(F)
            order = sorted(range(len(idx)), key=lambda j: (-s[j], idx[j]))
            best = next((idx[j] for j in order if idx[j] not in ix.train_targets_set), idx[order[0]]) if hasattr(ix, "train_targets_set") else idx[order[0]]
            ok[name] += best == t
    print(f"{tn:13s} base {ok['base'] / len(rows):.3f}  llm {ok['llm'] / len(rows):.3f}", flush=True)
