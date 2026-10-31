# author: opus
"""Summarise regime_fp_fold*.json against the Public Korean fingerprint (acc .925, hard .30, hard acc .76, 105 -5.1%)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
TARGET = dict(acc=0.925, hard=0.30, hard_acc=0.76, d105=-27 / 532)
res = {}
for p in sorted(HERE.glob((sys.argv[1] if len(sys.argv) > 1 else "regime_fp") + "_fold*.json")):
    for k, v in json.loads(p.read_text()).items():
        res.setdefault(k, []).extend(v)
print(f"{'regime':16s}  {'acc':>6s} {'hard':>6s} {'hardacc':>7s} {'d105':>7s}   dist")
for k, v in res.items():
    a = np.array(v)
    m = dict(acc=a[:, 0].mean(), hard=a[:, 1].mean(), hard_acc=a[a[:, 1] == 1, 0].mean(), d105=(a[:, 2] - a[:, 0]).mean())
    d = sum(abs(m[x] - TARGET[x]) / s for x, s in (("acc", .02), ("hard", .05), ("hard_acc", .05), ("d105", .01)))
    print(f"{k:16s}  {m['acc']:.4f} {m['hard']:.3f} {m['hard_acc']:.3f}   {m['d105']*100:+.2f}%   {d:.2f}   n={len(a)}")
