# author: opus
"""Fast fold-0 comparison of augmentation variants under stress inputs."""
import sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from cv import folds, load
import opus_lib as L
from opus_cv import PARAMS
from opus_stress import VARIANTS

train, catalog = load(); f = folds(train); k = 0
tr, va = train[f != k], train[f == k]
ix = L.Index(catalog, tr)
vk = va[va.vessel_name.map(L.is_hangul)]
for noisy in (False, True):
    X, y, g, _ = L.build(ix, L.augment_korean(tr, k, noisy_copy=noisy), True)
    m = lgb.train(PARAMS, lgb.Dataset(X, y, group=g, feature_name=L.FEATURES), 600)
    out = []
    for name in ("orig", "nogt", "nfd", "gt+15%"):
        ok = 0
        for row in vk.to_dict("records"):
            row = VARIANTS[name](row); t = ix.id2i[row["target_vessel_id"]]
            idx, F = L.query_features(ix, row, L.query_candidates(ix, row))
            ok += idx[int(np.argmax(m.predict(F)))] == t
        out.append(f"{name}={ok/len(vk):.3f}")
    print("noisy_copy" if noisy else "nogt_only ", " ".join(out), flush=True)
