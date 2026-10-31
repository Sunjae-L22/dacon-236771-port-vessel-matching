"""v42: v35/v38 matcher + typo-robust ('fuzzy') candidate retrieval for hangul names.

The evaluation hangul names carry the organizers' synthetic typos (syllable deleted, duplicated, or swapped with its
neighbour), often without tonnage. Such a typo bends the learned transliteration score (PMI) of the true vessel, so it
can fall out of the short candidate lists although its consonant skeleton / romanisation is still close. This adds,
for hangul rows only, the catalog-wide top-K by
  * skeleton edit similarity (the 'first' score that builds the retrieval pool),
  * romanised-name similarity (fuzz.ratio, i.e. normalised Indel distance),
  * transliteration PMI (a longer list),
  * partial skeleton similarity,
  * skeleton similarity of typo-reversed names (adjacent syllables swapped back, a doubled syllable removed),
to the candidate set. Features are computed exactly as before; the rankers are unchanged. Row-independent: only the
row's own name is used.
"""
from __future__ import annotations

import unicodedata

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from pipe21 import *  # noqa: F401,F403
from pipe21 import Matcher21
import vm

FZ_FIRST = 500
FZ_ROM = 300
FZ_PMI = 200
FZ_SKP = 100
FZ_TYPO = 50
FZ_TYPO_MAX_VARIANTS = 12


def typo_reversals(name: str) -> list[str]:
    """Names that undo one organizer-style syllable typo: swap two adjacent syllables back, or drop one copy of a
    doubled syllable (within a word)."""
    out = []
    words = name.split(" ")
    for wi, w in enumerate(words):
        syl = list(w)
        for i in range(len(syl) - 1):
            if not ("가" <= syl[i] <= "힣" and "가" <= syl[i + 1] <= "힣"):
                continue
            if syl[i] == syl[i + 1]:
                v = syl[:i] + syl[i + 1:]
            else:
                v = syl[:i] + [syl[i + 1], syl[i]] + syl[i + 2:]
            out.append(" ".join(words[:wi] + ["".join(v)] + words[wi + 1:]))
    seen, res = set(), []
    for v in out:
        if v not in seen and v != name:
            seen.add(v)
            res.append(v)
    return res[:FZ_TYPO_MAX_VARIANTS]


class Matcher23(Matcher21):
    def candidates(self, q, row):
        out = super().candidates(q, row)
        if not q["hang"]:
            return out
        cands, nsim, ret, gap, tscore, tinfo = out[0], out[1], out[2], out[3], out[4], out[5]
        ix = self.ix
        sk = ix.skel_sim_all(q["sk"])
        vk = process.cdist([v for v in q["vk"] if v] or [""], ix.vkey, scorer=Levenshtein.normalized_similarity,
                           dtype=np.float32, workers=-1).max(axis=0)
        first = sk + vk
        add = set(self._top(first, FZ_FIRST).tolist())
        if tinfo is not None:
            add.update(self._top(tinfo["romsim"] + 0.01 * first, FZ_ROM).tolist())
            add.update(self._top(tinfo["pmi"], FZ_PMI).tolist())
            add.update(self._top(tinfo["skp"] + 0.01 * first, FZ_SKP).tolist())
        name = unicodedata.normalize("NFC", row.get("vessel_name")) if isinstance(row.get("vessel_name"), str) else ""
        alts = []
        for v in typo_reversals(name):
            alts += [a[0] for a in vm.ko_variants(v)[0][:2]]
        if alts:
            add.update(self._top(ix.skel_sim_all(alts) + 0.01 * first, FZ_TYPO).tolist())
        sel = set(cands.tolist()) | add
        cands = np.array(sorted(sel, key=lambda i: (-ret[i], i)))
        return (cands,) + tuple(out[1:])
