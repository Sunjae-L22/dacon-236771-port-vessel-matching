"""v33: v31 + rarity-weighted token matching (hangul and english).

A matched candidate word counts by its inverse document frequency over the catalog: matching a rare
word (LOMBARD) is strong evidence, matching a fleet/common word (GLOVIS, ACE, KMTC) is weak."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
from rapidfuzz import fuzz

from pipe17 import *  # noqa: F401,F403
from pipe17 import Matcher17, FEATURES7 as _F17, jamo
import vm

IDF_FEATURES = ["idf_matched", "idf_max", "idf_unmatched", "idf_frac", "idf_rank"]
FEATURES7 = list(_F17) + IDF_FEATURES
MATCH = 0.8


class Matcher19(Matcher17):
    def __init__(self, *a, word_dict=None, **kw):
        super().__init__(*a, word_dict=word_dict, **kw)
        self._word_dict = word_dict or {}
        names = self.ix.cat["vessel_name"].astype(str).tolist()
        words = [vm.en_split(n)[0] for n in names]
        df = Counter(w for ws in words for w in set(ws))
        N = len(names)
        self.idf = {w: math.log(N / c) for w, c in df.items()}
        self.cwords = words

    def _idf_feats(self, cands, sims_per_cand):
        n = len(cands)
        out = np.full((n, len(IDF_FEATURES)), np.nan, np.float32)
        for r, (ws, best_c) in enumerate(sims_per_cand):
            if not ws:
                continue
            idf = np.array([self.idf.get(w, 0.0) for w in ws], np.float32)
            m = best_c >= MATCH
            tot = float(idf.sum()) or 1.0
            out[r, 0] = float(idf[m].sum()); out[r, 1] = float(idf[m].max()) if m.any() else 0.0
            out[r, 2] = float(idf[~m].sum()); out[r, 3] = float(idf[m].sum()) / tot
        v = np.nan_to_num(out[:, 0], nan=-1)
        out[:, 4] = (v[None, :] > v[:, None]).sum(axis=1)
        return out

    def features(self, row: dict, q=None):
        q = q or self.query_info(row)
        cands, X = super().features(row, q)
        sims = []
        if q["hang"]:
            qt = [jamo(t) for t in vm.ko_split(q["name"])[0] if any("가" <= ch <= "힣" for ch in t)]
            for i in cands:
                ws, best = [], []
                if qt:
                    for w in self.cwords[i]:
                        sp = self._spellings(w)
                        if not sp:
                            continue
                        ws.append(w); best.append(max(fuzz.ratio(a, s) for a in qt for s in sp) / 100.0)
                sims.append((ws, np.array(best, np.float32)))
        else:
            qt = vm.en_split(q["name"])[0]
            for i in cands:
                ws = self.cwords[i]
                best = np.array([max((fuzz.ratio(a, w) for a in qt), default=0) / 100.0 for w in ws], np.float32) if qt else np.zeros(len(ws), np.float32)
                sims.append((ws, best))
        return cands, np.hstack([X, self._idf_feats(cands, sims)]).astype(np.float32)

    def _spellings(self, w):
        if not hasattr(self, "_sp_cache"):
            self._sp_cache = {}
        if w not in self._sp_cache:
            import json, os
            if not hasattr(self, "_tok_ko"):
                self._tok_ko = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tok_ko.json"), encoding="utf-8"))
            sp = set()
            if self._tok_ko.get(w):
                sp.add(self._tok_ko[w])
            sp.update(self._word_dict.get(w, ()))
            self._sp_cache[w] = sorted(jamo(x) for x in sp)
        return self._sp_cache[w]
