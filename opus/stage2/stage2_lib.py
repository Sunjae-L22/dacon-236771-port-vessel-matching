# author: opus
"""Stage-2 feature builder shared by training (stage2.py) and the submission script."""
import numpy as np
import pandas as pd


def relative(X, s1, q):
    """X: (n, 60) candidates sorted by stage-1 score inside each query q (leader first)."""
    first = np.r_[True, q[1:] != q[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(q)), 0))
    lead = start
    pos = np.arange(len(q)) - start
    n = np.bincount(q)[q]
    s_lead = s1[lead]
    second = np.where(n > 1, s1[np.minimum(lead + 1, len(q) - 1)], s1[lead])
    ex = np.exp(s1 - s_lead)
    soft = ex / np.bincount(q, weights=ex)[q]
    cols = [s1, s1 - s_lead, s_lead - second, soft, pos, n]
    diff = X - X[lead]
    rk = np.empty_like(X)
    df = pd.DataFrame(X)
    df["q"] = q
    rk[:] = df.groupby("q").rank(pct=True).to_numpy()
    return np.column_stack(cols + [X, diff, rk]).astype(np.float32)
