# author: opus
"""Stage-1 accuracy / top-k recall per fold and variant from oof/fold*.npz."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "oof"
for f in sorted(OUT.glob("fold*.npz")):
    z = np.load(f); m = pd.read_csv(f.with_name(f.stem + "_queries.csv"))
    q, cand, tgt = z["q"], z["cand"], z["target"]
    hit = cand == tgt[q]
    first = np.r_[True, q[1:] != q[:-1]]
    pos = np.arange(len(q)) - np.maximum.accumulate(np.where(first, np.arange(len(q)), 0))
    m["top1"] = np.bincount(q, weights=hit & (pos == 0), minlength=len(m)) > 0
    m["topk"] = np.bincount(q, weights=hit, minlength=len(m)) > 0
    print(f.stem, len(m), "top1 %.3f topk %.3f" % (m.top1.mean(), m.topk.mean()),
          "fixable(topk&!top1)", int((m.topk & ~m.top1).sum()), "errors", int((~m.top1).sum()))
    if "-v" in sys.argv:
        print(m.groupby(m.variant.str.replace(r"\d$", "", regex=True))[["top1", "topk"]].mean().round(3).to_string())
