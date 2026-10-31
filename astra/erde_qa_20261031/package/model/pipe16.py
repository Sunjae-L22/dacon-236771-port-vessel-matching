"""v30: v18 matcher + token-level reverse channel for hangul queries.

Every English token of a candidate's name is spelled in hangul once (standard-orthography engine,
cached in tok_ko.json); query hangul tokens are compared with candidate tokens at the jamo level.
This gives a sharp "this fragment matches that word" signal for truncated / partly dropped names.
"""
from __future__ import annotations

import json
import os
import unicodedata

import numpy as np
from rapidfuzz import fuzz

from pipe11 import *  # noqa: F401,F403
from pipe11 import FEATURES7 as _F7, Matcher11
import vm

NEW_FEATURES = ["tk_max", "tk_mean_q", "tk_min_q", "tk_nq_match", "tk_cov_c", "tk_whole", "tk_whole_part",
                "tk_mean_q_rank", "tk_whole_rank", "tk_mean_q_gap"]
FEATURES7 = list(_F7) + NEW_FEATURES


def jamo(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if not ch.isspace())


class Matcher16(Matcher11):
    def __init__(self, *a, tok_path: str = None, **kw):
        super().__init__(*a, **kw)
        path = tok_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "tok_ko.json")
        tok_ko = json.load(open(path, encoding="utf-8"))
        self.ctoks = []
        for n in self.ix.cat["vessel_name"].astype(str):
            ws = vm.en_split(n)[0]
            self.ctoks.append([jamo(tok_ko.get(w, "")) for w in ws if tok_ko.get(w)])
        self.cwhole = ["".join(t) for t in self.ctoks]

    def features(self, row: dict, q=None):
        q = q or self.query_info(row)
        cands, X = super().features(row, q)
        n = len(cands)
        extra = np.full((n, len(NEW_FEATURES)), np.nan, np.float32)
        if q["hang"] and n:
            qtoks = [jamo(t) for t in vm.ko_split(q["name"])[0] if any("가" <= ch <= "힣" for ch in t)]
            qwhole = "".join(qtoks)
            if qtoks:
                for r, i in enumerate(cands):
                    ct = self.ctoks[i]
                    if not ct:
                        continue
                    sim = np.array([[fuzz.ratio(a, b) / 100.0 for b in ct] for a in qtoks], np.float32)
                    best_q = sim.max(axis=1); best_c = sim.max(axis=0)
                    extra[r, 0] = sim.max(); extra[r, 1] = best_q.mean(); extra[r, 2] = best_q.min()
                    extra[r, 3] = float((best_q >= 0.75).sum()); extra[r, 4] = float((best_c >= 0.75).mean())
                    extra[r, 5] = fuzz.ratio(qwhole, self.cwhole[i]) / 100.0
                    extra[r, 6] = fuzz.partial_ratio(qwhole, self.cwhole[i]) / 100.0 if qwhole and self.cwhole[i] else 0.0
                m = np.nan_to_num(extra[:, 1], nan=-1); w = np.nan_to_num(extra[:, 5], nan=-1)
                extra[:, 7] = (m[None, :] > m[:, None]).sum(axis=1)
                extra[:, 8] = (w[None, :] > w[:, None]).sum(axis=1)
                extra[:, 9] = m - m.max()
        return cands, np.hstack([X, extra]).astype(np.float32)
