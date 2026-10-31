"""v34: v33 + exclusion-aware retrieval.

Evaluation vessels are disjoint from train.csv vessels, so the train vessels are masked out of the
catalog *before* every top-k retrieval channel instead of only at the final pick. The freed candidate
slots go to vessels that can actually be the answer, and every candidate-set feature (ranks, gaps to
the best, ties, exact-tonnage counts, pool-relative transliteration scores) is computed among them.
Training items get the same treatment: a fold-k item masks the train vessels of the other folds.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
from rapidfuzz import fuzz, process

from pipe19 import *  # noqa: F401,F403
from pipe19 import Matcher19, jamo, FEATURES7 as _F19
from pipe11 import Matcher11
import vm

TOK_RETR_K = 30
# borrowed-name signal: the record name is (nearly) the name another vessel already carries in train.csv.
# Evaluation vessels are not train vessels, so such a name points away from the name channel (a renamed
# vessel or a name taken from another ship); the ranker can then lean on tonnage / type / flag.
TN_FEATURES = ["q_train_exact", "q_train_near", "q_train_near_tok"]
FEATURES7 = list(_F19) + TN_FEATURES


def name_key(s) -> str:
    s = unicodedata.normalize("NFC", str(s)).upper()
    return re.sub(r"[^0-9A-Z가-힣]", "", s)


def fine_key(s) -> str:
    """jamo-level key for hangul (finer edit distance), plain key otherwise."""
    k = name_key(s)
    return unicodedata.normalize("NFD", k) if vm.is_hangul(k) else k


class Matcher20(Matcher19):
    def __init__(self, *a, excluded=None, train_names=None, **kw):
        super().__init__(*a, **kw)
        self.set_excluded(excluded)
        names = sorted({str(n) for n in (train_names or []) if isinstance(n, str)})
        self.tn_exact = {name_key(n) for n in names}
        self.tn_fine = {True: [], False: []}
        self.tn_tok = {True: [], False: []}
        for n in names:
            h = vm.is_hangul(n)
            self.tn_fine[h].append(fine_key(n))
            self.tn_tok[h].append(" ".join(fine_key(t) for t in n.split()))
        self._tn_cache = {}

    def train_name_feats(self, name: str):
        if name in self._tn_cache:
            return self._tn_cache[name]
        h = vm.is_hangul(name)
        k = name_key(name)
        exact = float(bool(k) and k in self.tn_exact)
        near = tok = 0.0
        if k and self.tn_fine[h]:
            r = process.extractOne(fine_key(name), self.tn_fine[h], scorer=fuzz.ratio)
            near = r[1] / 100.0 if r else 0.0
            r = process.extractOne(" ".join(fine_key(t) for t in name.split()), self.tn_tok[h], scorer=fuzz.token_set_ratio)
            tok = r[1] / 100.0 if r else 0.0
        out = (exact, near, tok)
        self._tn_cache[name] = out
        return out

    def features(self, row: dict, q=None):
        q = q or self.query_info(row)
        cands, X = super().features(row, q)
        tn = np.tile(np.array(self.train_name_feats(q["name"]), np.float32), (len(cands), 1))
        return cands, np.hstack([X, tn]).astype(np.float32)

    def set_excluded(self, excluded):
        self.excl = None if excluded is None else np.asarray(excluded, bool)

    def _top(self, arr, k):
        if self.excl is not None and len(arr) == self.ix.N:
            arr = np.where(self.excl, -1e18, arr)
        return Matcher11._top(arr, k)

    def token_candidates(self, q, tscore):
        qt = [jamo(t) for t in vm.ko_split(q["name"])[0] if sum("가" <= ch <= "힣" for ch in t) >= 2]
        if not qt:
            return np.array([], dtype=np.int64)
        score = np.zeros(self.ix.N, np.float32)
        for a in qt:
            for key, sc, idx in process.extract(a, self.tok_keys, scorer=fuzz.ratio, score_cutoff=78, limit=40):
                v = self.tok_vessels[idx]
                score[v] = np.maximum(score[v], sc / 100.0)
        if self.excl is not None:
            score[self.excl] = 0.0
        if not score.any():
            return np.array([], dtype=np.int64)
        comb = score + 0.6 * tscore
        k = min(TOK_RETR_K, int((score > 0).sum()))
        idx = np.argpartition(-comb, k - 1)[:k] if k > 0 else np.array([], dtype=np.int64)
        return idx[score[idx] > 0]

    def candidates(self, q, row):
        out = super().candidates(q, row)
        if self.excl is None:
            return out
        cands = out[0]
        keep = ~self.excl[cands]
        if keep.any():
            cands = cands[keep]
        return (cands,) + tuple(out[1:])
