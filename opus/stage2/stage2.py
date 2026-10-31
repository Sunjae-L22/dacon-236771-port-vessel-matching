# author: opus
"""Stage-2 Korean top-k reranker trained on stage-1 out-of-fold candidates.

Each candidate in a query's top-K gets its 60 stage-1 features plus query-relative views that a
pointwise-tree stage 1 cannot form: the stage-1 score margin to the leader, the feature minus the
leader's value, and the feature's rank percentile inside the top-K. Evaluation is vessel-grouped
cross-validation over the OOF folds, reported per variant against stage-1 top-1.

usage: python3 opus/stage2/stage2.py            (CV report)
       python3 opus/stage2/stage2.py --save DIR (also train on everything and save models)
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_lib import relative  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "oof"
TOPK = 20
FEATURES = json.loads((HERE / "stage1_features.json").read_text())
PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[1], learning_rate=0.03, num_leaves=15,
              min_data_in_leaf=40, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, lambdarank_truncation_level=TOPK, verbose=-1, num_threads=4, seed=0)
ROUNDS = 400


NAMES = (["s1", "s1_margin", "lead_gap", "s1_soft", "s1_pos", "k"] + FEATURES
         + [f + "_dlead" for f in FEATURES] + [f + "_rk" for f in FEATURES])


def load():
    parts = []
    qoff = 0
    for f in sorted(OUT.glob("fold*.npz")):
        z = np.load(f)
        m = pd.read_csv(f.with_name(f.stem + "_queries.csv"))
        m["fold"] = int(f.stem[4:])
        q = z["q"] + qoff
        parts.append((z["X"], z["s1"], z["cand"], q, z["target"][z["q"]], m))
        qoff += len(m)
    X, s1, cand, q, tgt = (np.concatenate([p[i] for p in parts]) for i in range(5))
    meta = pd.concat([p[5] for p in parts], ignore_index=True)
    return X, s1, cand, q, (cand == tgt).astype(np.int8), meta


def pick(score, q, cand):
    """Top candidate index per query (ties -> lower catalog index, as production)."""
    order = np.lexsort((cand, -score, q))
    first = np.r_[True, q[order][1:] != q[order][:-1]]
    return order[first]


def main(save=None):
    X, s1, cand, q, y, meta = load()
    F = relative(X, s1, q)
    groups = np.bincount(q)
    vessel_fold = meta["fold"].to_numpy()
    folds = sorted(set(vessel_fold))
    if len(folds) == 1:  # dev mode on a single OOF fold: split its vessels 4 ways
        v = meta["vessel"].astype("category").cat.codes.to_numpy()
        vessel_fold = v % 4
        folds = [0, 1, 2, 3]
    qfold = vessel_fold
    rowfold = qfold[q]
    s2 = np.zeros(len(q))
    for k in folds:
        tr, te = rowfold != k, rowfold == k
        m = lgb.train(PARAMS, lgb.Dataset(F[tr], y[tr], group=groups[qfold != k]), ROUNDS)
        s2[te] = m.predict(F[te])
    np.save(HERE / "s2_oof.npy", s2)
    for lam in (0.25, 0.5, 1.0, 2.0):  # stage-1 + lam * stage-2 blends
        wb = pick(s1 + lam * s2, q, cand)
        ok = pd.Series(y[wb] == 1)
        print("blend lam=%.2f all %.4f  by fold %s" % (lam, ok.mean(),
              ok.groupby(meta["fold"].to_numpy()).mean().round(4).to_dict()))
    w1 = pick(s1, q, cand)
    w2 = pick(s2, q, cand)
    meta["s1_ok"] = y[w1] == 1
    meta["s2_ok"] = y[w2] == 1
    meta["v"] = meta.variant.str.replace(r"\d$", "", regex=True)
    rep = meta.groupby("v")[["s1_ok", "s2_ok"]].agg(["mean", "sum"])
    rep["n"] = meta.groupby("v").size()
    pd.set_option("display.width", 200)
    print(rep.round(3).to_string())
    print("ALL n=%d  s1 %.4f  s2 %.4f  fixed %d  broken %d" % (
        len(meta), meta.s1_ok.mean(), meta.s2_ok.mean(),
        (~meta.s1_ok & meta.s2_ok).sum(), (meta.s1_ok & ~meta.s2_ok).sum()))
    by_fold = meta.groupby("fold")[["s1_ok", "s2_ok"]].mean()
    print(by_fold.round(4).to_string())
    if save:
        save = Path(save)
        save.mkdir(parents=True, exist_ok=True)
        for seed in (0, 1, 2):  # small bag; CV used seed 0 alone
            m = lgb.train(dict(PARAMS, seed=seed), lgb.Dataset(F, y, group=groups, feature_name=NAMES), ROUNDS)
            m.save_model(str(save / f"stage2_{seed}.txt"))
        print("saved", save)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save")
    main(p.parse_args().save)
