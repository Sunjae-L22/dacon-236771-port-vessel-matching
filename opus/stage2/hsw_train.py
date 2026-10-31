# author: opus
"""Reproduce HanSunwook's 1031_08 Korean model (github HanSunwook/vessel-analysis-project, Public Korean 495) from
its code: pipeline.fit on all train rows with code/best_1031_08 settings, augmented Hangul rows weighted 1.0
instead of augment.AUG_WEIGHT (= tools/train_full.py HANGUL_AUG_WEIGHT=1.0, the "가중치 1.0" of 1031_08).
Only its Korean predictions are used (build_vote.py), so its English half need not match their hybrid.

usage: python3 hsw_train.py CODE_DIR OUT_MODEL.pkl
"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CODE, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, CODE)
import augment  # noqa: E402
import matcher  # noqa: E402
import pipeline  # noqa: E402

_pair_weights = pipeline.pair_weights


def pair_weights(rows, lens, X):
    w = _pair_weights(rows, lens, X)
    aug = np.repeat(augment.is_aug(rows["sample_id"]), lens)
    hang = X[:, 0] > 0
    return np.where(aug & hang, w / augment.AUG_WEIGHT * 1.0, w)


pipeline.pair_weights = pair_weights

t0 = time.time()
train = pipeline.read_records(ROOT / "data/open/train.csv")
cat = matcher.Catalog(pd.read_csv(ROOT / "data/open/vessel_catalog.csv", encoding="utf-8-sig"))
print(f"catalog {time.time() - t0:.0f}s", flush=True)
bundle = pipeline.fit(train, cat)
joblib.dump(bundle, OUT, compress=3)
print(f"saved {OUT} {time.time() - t0:.0f}s", flush=True)
