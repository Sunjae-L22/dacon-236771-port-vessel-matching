# author: Sol
"""Untrained prototype for two per-query tonnage-window name features.

This module is not wired into any pinned build or submission. Scores must be
the existing forward ko_combined_score, one candidate pool for one query.
Outside/empty windows yield NaN; a singleton has diff=0 and margin=NaN.
Ties use the exact input values (no new tolerance hyperparameter).
"""
import numpy as np

FEATURE_NAMES = ("ko_gt_win_diff_max", "ko_gt_win_margin")


def window_features(comb, best_gap, threshold=0.12):
    comb = np.asarray(comb, dtype=np.float32)
    gap = np.asarray(best_gap, dtype=np.float32)
    if comb.ndim != 1 or gap.shape != comb.shape:
        raise ValueError("comb and best_gap must be equal-length one-dimensional arrays")
    out = np.full((len(comb), 2), np.nan, dtype=np.float32)
    inside = np.isfinite(comb) & np.isfinite(gap) & (gap <= threshold)
    values = comb[inside]
    if not len(values):
        return out
    top = values.max()
    out[inside, 0] = values - top
    if len(values) > 1:
        second = np.partition(values, -2)[-2]
        competitors = np.where(values == top, second, top)
        out[inside, 1] = values - competitors
    return out
