# author: opus
"""5-fold grouped CV for the opus ranker. Caches features per fold."""
import os
import shutil
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cv import folds, load, score  # noqa: E402
import opus_lib as L  # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "experiments" / "cache"
AUG_SEED = int(os.environ.get("AUG_SEED", "0")) * 1000  # different augmentation draws for seed ensembles
CACHE.mkdir(parents=True, exist_ok=True)

PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[1], learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambdarank_truncation_level=10, verbose=-1, num_threads=8)


def fold_data(train, catalog, f, k, tag):
    p = CACHE / f"{tag}_fold{k}.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        return [z[n] for n in ("Xtr", "ytr", "gtr", "Xva", "yva", "gva")] + [list(z["cva"])]
    snap = CACHE.parent / "snapshots" / tag
    if not snap.exists():  # keep the code that produced this cache, so old tags stay reproducible
        snap.mkdir(parents=True)
        for name in ("opus_lib.py", "ko_translit_argon.py", "opus_cv.py"):
            shutil.copy(Path(__file__).parent / name, snap / name)
    tr, va = train[f != k], train[f == k]
    ix = L.Index(catalog, tr)
    t = time.time()
    Xtr, ytr, gtr, _ = L.build(ix, pd.concat([L.augment_korean(tr, k + AUG_SEED), L.augment_corrupt(tr, k + AUG_SEED), L.augment_rename(tr, k + AUG_SEED)], ignore_index=True), True)
    Xva, yva, gva, cva = L.build(ix, va, True)
    print(f"fold {k} features {time.time() - t:.0f}s", flush=True)
    np.savez(p, Xtr=Xtr, ytr=ytr, gtr=gtr, Xva=Xva, yva=yva, gva=gva, cva=np.array(cva, dtype=object))
    return Xtr, ytr, gtr, Xva, yva, gva, cva


def main(tag="v1", rounds=600):
    train, catalog = load()
    f = folds(train)
    ids = catalog["vessel_id"].astype(str).to_numpy()
    pred = np.empty(len(train), dtype=object)
    recall = []
    for k in range(5):
        Xtr, ytr, gtr, Xva, yva, gva, cva = fold_data(train, catalog, f, k, tag)
        m = lgb.train(PARAMS, lgb.Dataset(Xtr, ytr, group=gtr, feature_name=L.FEATURES), rounds)
        s = m.predict(Xva)
        va_idx = np.flatnonzero(f == k)
        off = 0
        for qi, g in enumerate(gva):
            pred[va_idx[qi]] = ids[cva[qi][int(np.argmax(s[off:off + g]))]]
            recall.append(yva[off:off + g].any())
            off += g
    sc = score(train["vessel_name"], train["target_vessel_id"], pred)
    ko = train["vessel_name"].map(L.is_hangul).to_numpy()
    rec = np.array(recall)  # order differs from train but only means are used per group below
    print(tag, sc, "candidate recall", rec.mean())
    imp = pd.Series(m.feature_importance("gain"), index=L.FEATURES).sort_values(ascending=False)
    print(imp.head(20).round(0).to_string())
    oof = train[["sample_id", "vessel_name", "target_vessel_id"]].assign(pred=pred, fold=f)
    oof.to_csv(CACHE / f"oof_{tag}.csv", index=False)
    wrong = oof[(oof.pred != oof.target_vessel_id) & ko]
    print("hangul misses", len(wrong))
    return sc


if __name__ == "__main__":
    main(*sys.argv[1:2])
