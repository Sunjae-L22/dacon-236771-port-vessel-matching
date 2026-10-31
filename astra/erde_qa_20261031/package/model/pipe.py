"""Candidate retrieval + pairwise features for vessel matching.

Row independence: `Matcher.process_row` only looks at one query row plus the
immutable catalog index and statistics built from the official train.csv.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import DamerauLevenshtein, JaroWinkler, Levenshtein

import vm

TON_TOL = 0.06


def rel_gap(a, b):
    if a is None or b is None or not np.isfinite(b) or b <= 0:
        return np.nan
    return abs(a - b) / max(a, b)


class TrainStats:
    """Statistics learned only from official train rows (fold-aware in CV)."""

    def __init__(self, train: pd.DataFrame, catalog: pd.DataFrame):
        c = catalog.set_index("vessel_id")
        m = train.join(c[["vessel_type", "flag_name"]], on="target_vessel_id", rsuffix="_cat")
        self.type_pair = self._cond(m["vessel_type"], m["vessel_type_cat"])
        self.flag_pair = self._cond(m["flag_name"].fillna("__NA__"), m["flag_name_cat"])
        # prior of catalog types among all matched rows
        vc = m["vessel_type_cat"].value_counts()
        self.cat_type_prior = (vc / vc.sum()).to_dict()
        # memorised maps (only fire when a test row shares a vessel with train)
        self.callsign_map = dict(zip(train["call_sign"], train["target_vessel_id"]))
        self.name_map = defaultdict(set)
        for n, t in zip(train["vessel_name"], train["target_vessel_id"]):
            self.name_map[vm.raw_compact(n)].add(t)

    @staticmethod
    def _cond(a: pd.Series, b: pd.Series):
        ct = pd.crosstab(a, b)
        tot = ct.sum(axis=1)
        out = {}
        for qa in ct.index:
            row = ct.loc[qa]
            nz = row[row > 0]
            out[qa] = {k: (v + 0.0) / (tot[qa] + 5.0) for k, v in nz.items()}
            out[qa]["__TOT__"] = float(tot[qa])
        return out

    def type_prob(self, qt, ct):
        d = self.type_pair.get(qt)
        if d is None:
            return np.nan
        return d.get(ct, 0.0)

    def flag_prob(self, qf, cf):
        d = self.flag_pair.get(qf if isinstance(qf, str) else "__NA__")
        if d is None:
            return np.nan
        return d.get(cf, 0.0)


class CatalogIndex:
    def __init__(self, catalog: pd.DataFrame):
        self.cat = catalog.reset_index(drop=True)
        self.ids = self.cat["vessel_id"].astype(str).to_numpy()
        self.id2i = {v: i for i, v in enumerate(self.ids)}
        names = self.cat["vessel_name"].tolist()
        self.compact = [vm.en_compact(n) for n in names]
        self.rawc = [vm.raw_compact(n) for n in names]
        self.nums = [tuple(vm.en_split(n)[1]) for n in names]
        self.tokens = [" ".join(vm.en_split(n)[0]) for n in names]
        sk_flat, owner = [], []
        self.sk = []
        for i, n in enumerate(names):
            alts = vm.en_skeletons(n)
            self.sk.append(alts)
            for a in alts:
                sk_flat.append(a); owner.append(i)
        self.sk_flat = sk_flat
        self.sk_owner = np.asarray(owner)
        self.vkey = ["".join(vm.en_skeleton_word(w, True) for w in vm.en_split(n)[0]) for n in names]
        self.gt = pd.to_numeric(self.cat["gross_tonnage"], errors="coerce").to_numpy(float)
        self.dwt = pd.to_numeric(self.cat["deadweight_tonnage"], errors="coerce").to_numpy(float)
        self.ctype = self.cat["vessel_type"].astype(str).to_numpy()
        self.cflag = self.cat["flag_name"].astype(str).to_numpy()
        self.N = len(self.cat)

    # -- vectorised similarity of one query against the whole catalog ----------
    def name_sim_all(self, q: str) -> np.ndarray:
        if not q:
            return np.zeros(self.N, np.float32)
        return process.cdist([q], self.compact, scorer=fuzz.ratio, dtype=np.uint8, workers=-1)[0].astype(np.float32) / 100.0

    def skel_sim_all(self, alts: list[str]) -> np.ndarray:
        alts = [a for a in alts if a]
        if not alts:
            return np.zeros(self.N, np.float32)
        mat = process.cdist(alts, self.sk_flat, scorer=Levenshtein.normalized_similarity,
                            dtype=np.float32, workers=-1)
        best = mat.max(axis=0)
        out = np.zeros(self.N, np.float32)
        np.maximum.at(out, self.sk_owner, best)
        return out

    def ton_score_all(self, gts: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """relative gap (min over query tonnage fields) and a 0..1 score."""
        gap = np.full(self.N, np.inf)
        with np.errstate(invalid="ignore", divide="ignore"):
            for g in gts:
                gg = np.abs(self.gt - g) / np.maximum(self.gt, g)
                gap = np.fmin(gap, gg)
        score = np.where(gap == 0, 1.0, np.clip(1 - gap / TON_TOL, 0, 1) * 0.7)
        score = np.nan_to_num(score, nan=0.0)
        return gap, score


FEATURES = [
    "is_hangul", "q_len", "c_len",
    "r_ratio", "r_raw_ratio", "r_partial", "r_tsort", "r_tset", "r_jw", "r_dl_dist", "r_lev_dist",
    "exact_compact", "exact_raw", "prefix3", "first_tok_eq",
    "sk_sim", "sk_dist", "sk_len_q", "sk_len_c", "vk_sim", "vk_partial",
    "num_q", "num_c", "num_match", "num_mismatch",
    "gap_gt", "gap_igt", "gap_min", "exact_gt", "exact_igt", "q_ton_missing", "c_gt_missing",
    "log_ratio_gt", "c_log_gt", "c_dwt_gt_ratio",
    "type_prob", "type_tot", "flag_prob", "q_flag_missing", "cat_type_prior",
    "ret_score", "ret_rank", "name_rank", "name_gap_best", "ton_rank",
    "n_exact_ton_in_cands", "n_cands",
    "train_name_hit", "train_cs_hit",
]


class Matcher:
    def __init__(self, index: CatalogIndex, stats: TrainStats, k_comb=40, k_name=15, k_ton=10):
        self.ix = index
        self.st = stats
        self.k_comb, self.k_name, self.k_ton = k_comb, k_name, k_ton

    def query_info(self, row: dict):
        name = row.get("vessel_name")
        name = name if isinstance(name, str) else ""
        hang = vm.is_hangul(name)
        gts = [x for x in (vm.pos_float(row.get("gross_tonnage")), vm.pos_float(row.get("international_gross_tonnage"))) if x]
        if hang:
            alts, nums = vm.ko_variants(name)
            sk = [a[0] for a in alts]
            vk = [a[1] for a in alts]
            # latin chunks inside a hangul name
            comp = vm.en_compact(name)
        else:
            sk = vm.en_skeletons(name)
            w, nums = vm.en_split(name)
            vk = ["".join(vm.en_skeleton_word(x, True) for x in w)]
            comp = vm.en_compact(name)
        return dict(name=name, hang=hang, gts=gts, sk=sk, vk=vk, nums=tuple(nums), comp=comp,
                    raw=vm.raw_compact(name), tokens=" ".join(vm.en_split(name)[0]))

    def candidates(self, q):
        ix = self.ix
        gap, tscore = ix.ton_score_all(q["gts"]) if q["gts"] else (np.full(ix.N, np.inf), np.zeros(ix.N))
        if q["hang"]:
            nsim = ix.skel_sim_all(q["sk"])
        else:
            nsim = ix.name_sim_all(q["comp"])
        ret = nsim + 0.6 * tscore
        sel = set()
        def top(arr, k):
            if k <= 0:
                return []
            k = min(k, len(arr) - 1)
            idx = np.argpartition(-arr, k)[:k]
            return idx[np.argsort(-arr[idx])]
        for i in top(ret, self.k_comb):
            sel.add(int(i))
        for i in top(nsim, self.k_name):
            sel.add(int(i))
        if q["gts"]:
            for i in top(tscore + 0.01 * nsim, self.k_ton):
                sel.add(int(i))
        cands = np.array(sorted(sel, key=lambda i: -ret[i]))
        return cands, nsim, ret, gap, tscore

    def features(self, row: dict, q=None):
        q = q or self.query_info(row)
        cands, nsim, ret, gap, tscore = self.candidates(q)
        ix, st = self.ix, self.st
        qt = row.get("vessel_type")
        qf = row.get("flag_name")
        qgt = vm.pos_float(row.get("gross_tonnage"))
        qigt = vm.pos_float(row.get("international_gross_tonnage"))
        name_hits = st.name_map.get(q["raw"], set()) if q["raw"] else set()
        cs_hit = st.callsign_map.get(row.get("call_sign"))
        cn = nsim[cands]
        name_order = np.argsort(-cn)
        name_rank = np.empty(len(cands)); name_rank[name_order] = np.arange(len(cands))
        tg = gap[cands]
        ton_order = np.argsort(np.nan_to_num(tg, nan=9, posinf=9))
        ton_rank = np.empty(len(cands)); ton_rank[ton_order] = np.arange(len(cands))
        n_exact = int(np.sum(tg == 0))
        best_name = cn.max() if len(cn) else 0
        rows = []
        qcomp, qraw = q["comp"], q["raw"]
        for r, i in enumerate(cands):
            ccomp = ix.compact[i]
            cgt = ix.gt[i]
            g1 = rel_gap(qgt, cgt); g2 = rel_gap(qigt, cgt)
            if q["hang"]:
                sk_best, sk_d = 0.0, 99
                for a in q["sk"]:
                    for b in ix.sk[i]:
                        s = Levenshtein.normalized_similarity(a, b)
                        if s > sk_best:
                            sk_best = s; sk_d = Levenshtein.distance(a, b)
                vk_sim = max(Levenshtein.normalized_similarity(a, ix.vkey[i]) for a in q["vk"])
                vk_part = max(fuzz.partial_ratio(a, ix.vkey[i]) for a in q["vk"]) / 100
                sk_lq = min(len(a) for a in q["sk"])
            else:
                sk_best = max(Levenshtein.normalized_similarity(a, b) for a in q["sk"] for b in ix.sk[i])
                sk_d = min(Levenshtein.distance(a, b) for a in q["sk"] for b in ix.sk[i])
                vk_sim = Levenshtein.normalized_similarity(q["vk"][0], ix.vkey[i])
                vk_part = fuzz.partial_ratio(q["vk"][0], ix.vkey[i]) / 100
                sk_lq = len(q["sk"][0]) if q["sk"] else 0
            cnums = ix.nums[i]
            tp = st.type_prob(qt, ix.ctype[i])
            ttot = st.type_pair.get(qt, {}).get("__TOT__", 0.0)
            rows.append([
                float(q["hang"]), len(qcomp), len(ccomp),
                fuzz.ratio(qcomp, ccomp) / 100, fuzz.ratio(qraw, ix.rawc[i]) / 100,
                fuzz.partial_ratio(qcomp, ccomp) / 100 if qcomp and ccomp else 0.0,
                fuzz.token_sort_ratio(q["tokens"], ix.tokens[i]) / 100,
                fuzz.token_set_ratio(q["tokens"], ix.tokens[i]) / 100,
                JaroWinkler.similarity(qcomp, ccomp),
                DamerauLevenshtein.distance(qcomp, ccomp), Levenshtein.distance(qcomp, ccomp),
                float(bool(qcomp) and qcomp == ccomp), float(bool(qraw) and qraw == ix.rawc[i]),
                float(qcomp[:3] == ccomp[:3]) if qcomp else 0.0,
                float(q["tokens"].split(" ")[0] == ix.tokens[i].split(" ")[0]) if q["tokens"] else 0.0,
                sk_best, sk_d, sk_lq, min(len(b) for b in ix.sk[i]) if ix.sk[i] else 0, vk_sim, vk_part,
                len(q["nums"]), len(cnums), float(len(q["nums"]) > 0 and set(q["nums"]) == set(cnums)),
                float(set(q["nums"]) != set(cnums)),
                g1, g2, np.nanmin([g1, g2]) if not (np.isnan(g1) and np.isnan(g2)) else np.nan,
                float(g1 == 0), float(g2 == 0), float(qgt is None and qigt is None), float(not np.isfinite(cgt)),
                math.log((qgt or qigt) / cgt) if (qgt or qigt) and np.isfinite(cgt) and cgt > 0 else np.nan,
                math.log1p(cgt) if np.isfinite(cgt) else np.nan,
                ix.dwt[i] / cgt if np.isfinite(cgt) and cgt > 0 and np.isfinite(ix.dwt[i]) else np.nan,
                tp, ttot, st.flag_prob(qf, ix.cflag[i]), float(not isinstance(qf, str)),
                st.cat_type_prior.get(ix.ctype[i], 0.0),
                ret[i], r, name_rank[r], best_name - cn[r], ton_rank[r],
                n_exact, len(cands),
                float(ix.ids[i] in name_hits), float(cs_hit == ix.ids[i]),
            ])
        return cands, np.asarray(rows, np.float32)
