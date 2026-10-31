"""v35: v33 matcher (train vessels stay in the candidate set, skipped only at the final pick) + borrowed-name
features (pipe20) + train-presence features.

Evaluation vessels are never train.csv vessels. Keeping the train vessels as candidates lets the ranker see
when the record name fits a train vessel best (a name taken from another ship / a renamed vessel); these
features make that explicit and also give every candidate its gaps to the best *eligible* (non-train)
candidate, so the ranker can compare eligible candidates directly:
  c_is_train        candidate is a train.csv vessel (never the answer)
  q_train_frac      share of train vessels among the candidates
  q_trbest_<s>      best <s> over train candidates minus best <s> over eligible candidates
  c_gap_elig_<s>    candidate's <s> minus the best <s> over eligible candidates
for <s> in name similarity, retrieval score, token match (hangul), transliteration PMI (hangul).
"""
from __future__ import annotations

import numpy as np

from pipe20 import *  # noqa: F401,F403
from pipe20 import Matcher20, FEATURES7 as _F20

TP_FEATURES = ["c_is_train", "q_train_frac",
               "q_trbest_name", "c_gap_elig_name", "q_trbest_ret", "c_gap_elig_ret",
               "q_trbest_tok", "c_gap_elig_tok", "q_trbest_pmi", "c_gap_elig_pmi"]
FEATURES7 = list(_F20) + TP_FEATURES
_I = {f: i for i, f in enumerate(_F20)}


class Matcher21(Matcher20):
    def __init__(self, *a, train_mask=None, **kw):
        super().__init__(*a, **kw)
        self.train_mask = None if train_mask is None else np.asarray(train_mask, bool)

    def features(self, row: dict, q=None):
        cands, X = super().features(row, q)
        n = len(cands)
        tp = np.full((n, len(TP_FEATURES)), np.nan, np.float32)
        tr = self.train_mask[cands] if self.train_mask is not None else np.zeros(n, bool)
        el = ~tr
        tp[:, 0] = tr
        tp[:, 1] = tr.mean() if n else 0.0
        sims = [X[:, _I["best_name"]] - X[:, _I["name_gap_best"]], X[:, _I["ret_score"]],
                X[:, _I["tk_mean_q"]], X[:, _I["t_pmi_rel"]]]
        for j, s in enumerate(sims):
            if np.all(np.isnan(s)) or not el.any():
                continue
            s = np.nan_to_num(s, nan=-1.0)
            eb = s[el].max()
            tp[:, 3 + 2 * j] = s - eb
            if tr.any():
                tp[:, 2 + 2 * j] = s[tr].max() - eb
        return cands, np.hstack([X, tp]).astype(np.float32)
