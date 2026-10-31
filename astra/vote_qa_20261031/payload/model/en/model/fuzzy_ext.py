# author: opus
"""Wide edit-distance candidates for Korean queries, ported from the user's v15-v18 (ksh1004 repo).

Added on top of 93's candidates without retraining (rank features of new candidates = their position):
  - Indel (fuzz.ratio) search of the query consonant key over every catalog ckey_acr, top 500
  - typo reversals of the query (adjacent syllables swapped, doubled syllable dropped) whose key exactly
    equals a catalog key (closest tonnage, 10)
  - the user's phonetic key (ksh_hangul.hangul_key) against english_key of every catalog name, top 500
Uses only the row's own fields and fixed catalog indexes, so it is row independent.
Fold check (opus/stage2/fuzzy_probe.py, 5 folds, 1191 Korean rows): clean 0, syllable typos + no tonnage
+24, syllable typos + tonnage +-15% +7, jamo typos + no tonnage -1.
"""
from collections import defaultdict
import unicodedata

import numpy as np
from rapidfuzz import fuzz, process

import opus_lib as L
from ksh_hangul import english_key, hangul_key

K_FUZZY = 500


def typo_reversals(name):
    out = []
    for i in range(len(name) - 1):
        a, b = name[i], name[i + 1]
        if a.isspace() or b.isspace():
            continue
        out.append(name[:i] + name[i + 1:] if a == b else name[:i] + b + a + name[i + 2:])
    return list(dict.fromkeys(out))


class FuzzyExt:
    def __init__(self, ix, catalog):
        self.ix = ix
        self.exact = defaultdict(list)
        for i, k in enumerate(ix.ckey_acr):
            if k:
                self.exact[k].append(i)
        self.kkeys = [english_key(v).replace(" ", "") for v in catalog["vessel_name"]]
        self.gt = catalog["gross_tonnage"].to_numpy(float)

    def _closest(self, hits, row, k=10):
        hits = np.asarray(hits, int)
        refs = [v for v in (L.positive(row.get("gross_tonnage")), L.positive(row.get("international_gross_tonnage")))
                if np.isfinite(v)]
        if len(hits) <= k or not refs:
            return hits[:k]
        gap = np.min([np.abs(self.gt[hits] - r) for r in refs], axis=0)
        return hits[np.lexsort((hits, np.where(np.isnan(gap), np.inf, gap)))[:k]]

    def add(self, cand, row):
        name = unicodedata.normalize("NFC", str(row.get("vessel_name") or ""))

        def put(i, r):
            e = cand.setdefault(int(i), [999, 999, 999])
            e[0] = min(e[0], r)
        _, ck, _ = L.ko_keys(name)
        if ck:
            for r, (_, _, i) in enumerate(process.extract(ck, self.ix.ckey_acr, scorer=fuzz.ratio, limit=K_FUZZY)):
                put(i, r)
        for v in typo_reversals(name):
            vk = L.ko_keys(v)[1]
            if vk and vk != ck and vk in self.exact:
                for i in self._closest(self.exact[vk], row):
                    put(i, 0)
        hk = hangul_key(name).replace(" ", "")
        if hk:
            for r, (_, _, i) in enumerate(process.extract(hk, self.kkeys, scorer=fuzz.ratio, limit=K_FUZZY)):
                put(i, r)
        return cand
