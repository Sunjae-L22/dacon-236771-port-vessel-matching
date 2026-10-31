"""v31: v30 + (a) several hangul spellings per English token (standard engine + spellings observed in the
training pairs) and (b) token-based candidate retrieval over the whole catalog."""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
from rapidfuzz import fuzz, process

from pipe16 import *  # noqa: F401,F403
from pipe16 import Matcher16, NEW_FEATURES, jamo
import vm

TOK_RETR_K = 30


def build_word_dict(train, catalog) -> dict:
    cname = dict(zip(catalog["vessel_id"], catalog["vessel_name"].astype(str)))
    out = defaultdict(set)
    h = train[train["vessel_name"].map(vm.is_hangul)].drop_duplicates("vessel_name")
    for ko, tid in zip(h["vessel_name"], h["target_vessel_id"]):
        en = cname.get(tid)
        if not en:
            continue
        kt = [t for t in ko.split() if any("가" <= ch <= "힣" for ch in t)]
        et = vm.en_split(en)[0]
        if len(kt) == len(et) and kt:
            for a, b in zip(et, kt):
                out[a].add(b)
    return out


class Matcher17(Matcher16):
    def __init__(self, *a, word_dict=None, **kw):
        super().__init__(*a, **kw)
        word_dict = word_dict or {}
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tok_ko.json")
        tok_ko = json.load(open(path, encoding="utf-8"))
        self.ctoks_multi = []
        index = defaultdict(set)
        for i, n in enumerate(self.ix.cat["vessel_name"].astype(str)):
            toks = []
            for w in vm.en_split(n)[0]:
                sp = set()
                if tok_ko.get(w):
                    sp.add(jamo(tok_ko[w]))
                for k in word_dict.get(w, ()):
                    sp.add(jamo(k))
                if sp:
                    toks.append(sorted(sp))
                    for s in sp:
                        if len(s) >= 4:
                            index[s].add(i)
            self.ctoks_multi.append(toks)
        self.tok_keys = list(index.keys())
        self.tok_vessels = [np.fromiter(index[k], dtype=np.int64) for k in self.tok_keys]

    # -- token retrieval -----------------------------------------------------
    def token_candidates(self, q, tscore):
        qt = [jamo(t) for t in vm.ko_split(q["name"])[0] if sum("가" <= ch <= "힣" for ch in t) >= 2]
        if not qt:
            return np.array([], dtype=np.int64)
        score = np.zeros(self.ix.N, np.float32)
        for a in qt:
            for key, sc, idx in process.extract(a, self.tok_keys, scorer=fuzz.ratio, score_cutoff=78, limit=40):
                v = self.tok_vessels[idx]
                score[v] = np.maximum(score[v], sc / 100.0)
        if not score.any():
            return np.array([], dtype=np.int64)
        comb = score + 0.6 * tscore
        k = min(TOK_RETR_K, int((score > 0).sum()))
        idx = np.argpartition(-comb, k - 1)[:k] if k > 0 else np.array([], dtype=np.int64)
        return idx[score[idx] > 0]

    def candidates(self, q, row):
        out = super().candidates(q, row)
        if not q["hang"]:
            return out
        cands, nsim, ret, gap, tscore = out[0], out[1], out[2], out[3], out[4]
        extra = self.token_candidates(q, tscore)
        if len(extra):
            sel = set(cands.tolist()) | set(extra.tolist())
            cands = np.array(sorted(sel, key=lambda i: (-ret[i], i)))
        return (cands,) + tuple(out[1:])

    # -- token features with several spellings --------------------------------
    def features(self, row: dict, q=None):
        q = q or self.query_info(row)
        cands, X = Matcher11_features(self, row, q)
        n = len(cands)
        extra = np.full((n, len(NEW_FEATURES)), np.nan, np.float32)
        if q["hang"] and n:
            qtoks = [jamo(t) for t in vm.ko_split(q["name"])[0] if any("가" <= ch <= "힣" for ch in t)]
            qwhole = "".join(qtoks)
            if qtoks:
                for r, i in enumerate(cands):
                    ct = self.ctoks_multi[i]
                    if not ct:
                        continue
                    sim = np.array([[max(fuzz.ratio(a, s) for s in sp) / 100.0 for sp in ct] for a in qtoks], np.float32)
                    best_q = sim.max(axis=1); best_c = sim.max(axis=0)
                    cw = "".join(sp[0] for sp in ct)
                    extra[r, 0] = sim.max(); extra[r, 1] = best_q.mean(); extra[r, 2] = best_q.min()
                    extra[r, 3] = float((best_q >= 0.75).sum()); extra[r, 4] = float((best_c >= 0.75).mean())
                    extra[r, 5] = fuzz.ratio(qwhole, cw) / 100.0
                    extra[r, 6] = fuzz.partial_ratio(qwhole, cw) / 100.0 if qwhole and cw else 0.0
                m = np.nan_to_num(extra[:, 1], nan=-1); w = np.nan_to_num(extra[:, 5], nan=-1)
                extra[:, 7] = (m[None, :] > m[:, None]).sum(axis=1)
                extra[:, 8] = (w[None, :] > w[:, None]).sum(axis=1)
                extra[:, 9] = m - m.max()
        return cands, np.hstack([X, extra]).astype(np.float32)


def Matcher11_features(self, row, q):
    from pipe11 import Matcher11
    return Matcher11.features(self, row, q)
