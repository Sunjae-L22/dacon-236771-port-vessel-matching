# author: opus
"""Reuse cached v4 fold features to test Korean-specific training schemes."""
import sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from cv import folds, load, score
import opus_lib as L
from opus_cv import PARAMS, CACHE

KO = L.FEATURES.index("is_ko")


def groups_mask(X, g):
    starts = np.r_[0, np.cumsum(g)[:-1]]
    return X[starts, KO] == 1, starts


def fit(X, y, g, w=None, rounds=600):
    return lgb.train(PARAMS, lgb.Dataset(X, y, group=g, feature_name=L.FEATURES), rounds)


def subset(X, y, g, keep):
    starts = np.r_[0, np.cumsum(g)[:-1]]
    rows = np.concatenate([np.arange(s, s + n) for s, n, k in zip(starts, g, keep) if k])
    return X[rows], y[rows], g[keep]


def repeat_ko(X, y, g, times):
    kog, _ = groups_mask(X, g)
    Xk, yk, gk = subset(X, y, g, kog)
    return np.vstack([X] + [Xk] * times), np.concatenate([y] + [yk] * times), np.concatenate([g] + [gk] * times)


def main(mode):
    train, catalog = load()
    f = folds(train)
    ids = catalog["vessel_id"].astype(str).to_numpy()
    pred = np.empty(len(train), dtype=object)
    for k in range(5):
        z = np.load(CACHE / f"v4_fold{k}.npz", allow_pickle=True)
        Xtr, ytr, gtr, Xva, gva, cva = z["Xtr"], z["ytr"], z["gtr"], z["Xva"], z["gva"], list(z["cva"])
        base = fit(Xtr, ytr, gtr)
        if mode == "sep":
            kog, _ = groups_mask(Xtr, gtr)
            kom = fit(*subset(Xtr, ytr, gtr, kog), rounds=300)
        elif mode.startswith("rep"):
            kom = fit(*repeat_ko(Xtr, ytr, gtr, int(mode[3:])))
        else:
            kom = base
        va_ko, _ = groups_mask(Xva, gva)
        sb, sk = base.predict(Xva), kom.predict(Xva)
        va_idx = np.flatnonzero(f == k)
        off = 0
        for qi, n in enumerate(gva):
            s = (sk if va_ko[qi] else sb)[off:off + n]
            pred[va_idx[qi]] = ids[cva[qi][int(np.argmax(s))]]
            off += n
    print(mode, score(train["vessel_name"], train["target_vessel_id"], pred), flush=True)


if __name__ == "__main__":
    for m in sys.argv[1:]:
        main(m)
