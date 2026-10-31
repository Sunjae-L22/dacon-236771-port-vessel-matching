"""v7 matcher: v6 + NFC input normalisation, reversed hangul word order,
english initial-form channel (G. SPIRIT).

(was) v6 matcher (v5 + hangul-side word dropping, softer english word-skip, token-set channel).

v3 (learned transliteration incl. word-skipping local score, partial retrieval,
tie-aware ranks, token overlap) + a structural retrieval channel (tonnage x
type x flag) + row-level tie features, with a fused transliteration DP.

Row independence: `Matcher4.features(row)` only uses that row, the immutable
catalog index and statistics/parameters fitted on the official train.csv.
"""
from __future__ import annotations

import math
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import DamerauLevenshtein, JaroWinkler, Levenshtein

import vm
import rom
import translit as T
from pipe import CatalogIndex, TrainStats, rel_gap  # noqa: F401

PMI_LAMBDA = 0.5
POOL = 3000
POOL_PARTIAL = 1000
EN_PARTIAL_TRIGGER = 0.85

FEATURES7 = [
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
    "t_pmi_rel", "t_pmi_rank", "t_lp_rel", "t_lp_per_unit", "t_lm", "t_n_units", "t_en_len",
    "t_margin", "t_pmi_minus_2nd",
    "t_loc_rel", "t_loc_rank", "t_loc_minus_glob", "t_loc_per_unit",
    "last_tok_eq", "tok_jacc", "tok_contain", "sk_partial", "r_partial_tok",
    # v4
    "best_name", "n_name_ties", "n_exact_name", "struct_score", "struct_rank",
    "type_rank_in_ties", "ton_rank_in_ties",
    # v5
    "rom_sim", "rom_partial", "rom_rank", "rom_gap_best",
    # v6
    "t_red_rel", "t_red_minus_full", "t_red_rank", "q_n_tokens", "en_tset_rank",
    # v7
    "init_sim", "q_has_initial",
]


def fit_transducer(td: T.Transducer, train: pd.DataFrame, ix: CatalogIndex, iters: int = 6):
    td.P = td.P0.copy()
    h = train[train["vessel_name"].map(vm.is_hangul)].drop_duplicates("target_vessel_id")
    pairs = []
    for name, tid in zip(h["vessel_name"], h["target_vessel_id"]):
        if tid in ix.id2i:
            pairs.append(([T.text_units(a) for a in T.hangul_text_variants(name)], ix.id2i[tid]))
    td.fit(pairs, iters=iters)
    return td


class Matcher7:
    def __init__(self, ix: CatalogIndex, stats: TrainStats, td: T.Transducer, lmv: np.ndarray,
                 k_comb=40, k_name=15, k_ton=10, k_pmi=30, k_struct=15):
        self.ix, self.st, self.td, self.lmv = ix, stats, td, lmv
        self.k_comb, self.k_name, self.k_ton, self.k_pmi, self.k_struct = k_comb, k_name, k_ton, k_pmi, k_struct
        L = np.array([len(a) for a in ix.sk_flat])
        self.sk_len_ok = np.clip(L / 4.0, 0, 1).astype(np.float32)
        self.type_list, self.type_codes = np.unique(ix.ctype, return_inverse=True)
        self.flag_list, self.flag_codes = np.unique(ix.cflag, return_inverse=True)
        self._type_cache, self._flag_cache = {}, {}
        self.init_comp = []
        for n in ix.cat["vessel_name"].astype(str):
            w = vm.en_split(n)[0]
            self.init_comp.append(w[0][:1] + "".join(w[1:]) if len(w) >= 2 else "")

    # -- vectorised type / flag compatibility -----------------------------------
    def type_arr(self, qt):
        if qt not in self._type_cache:
            d = self.st.type_pair.get(qt)
            v = np.zeros(len(self.type_list)) if d is None else np.array([d.get(t, 0.0) for t in self.type_list])
            self._type_cache[qt] = v
        return self._type_cache[qt][self.type_codes]

    def flag_arr(self, qf):
        key = qf if isinstance(qf, str) else "__NA__"
        if key not in self._flag_cache:
            d = self.st.flag_pair.get(key)
            v = np.zeros(len(self.flag_list)) if d is None else np.array([d.get(f, 0.0) for f in self.flag_list])
            self._flag_cache[key] = v
        return self._flag_cache[key][self.flag_codes]

    # ------------------------------------------------------------------
    def query_info(self, row: dict):
        name = row.get("vessel_name")
        name = unicodedata.normalize("NFC", name) if isinstance(name, str) else ""
        hang = vm.is_hangul(name)
        gts = [x for x in (vm.pos_float(row.get("gross_tonnage")),
                           vm.pos_float(row.get("international_gross_tonnage"))) if x]
        if hang:
            alts, nums = vm.ko_variants(name)
            sk = [a[0] for a in alts]
            vk = [a[1] for a in alts]
            units = [T.text_units(a) for a in T.hangul_text_variants(name)]
            units = [u for u in units if u]
            romv = rom.variants(name)
            toks = vm.ko_split(name)[0]
            units_red = []
            if len(toks) >= 2:
                for red in (" ".join(toks[1:]), " ".join(toks[:-1]), " ".join(reversed(toks))):
                    for a in T.hangul_text_variants(red)[:2]:
                        u = T.text_units(a)
                        if u:
                            units_red.append(u)
                    sk += [a[0] for a in vm.ko_variants(red)[0][:2]]
        else:
            romv = []
            units_red = []
            sk = vm.en_skeletons(name)
            w, nums = vm.en_split(name)
            vk = ["".join(vm.en_skeleton_word(x, True) for x in w)]
            units = []
        return dict(name=name, hang=hang, gts=gts, sk=sk, vk=vk, nums=tuple(nums),
                    comp=vm.en_compact(name), raw=vm.raw_compact(name),
                    tokens=" ".join(vm.en_split(name)[0]), units=units, romv=romv, units_red=units_red,
                    n_tokens=len(vm.ko_split(name)[0]) if hang else len(vm.en_split(name)[0]))

    @staticmethod
    def _top(arr, k):
        k = min(k, len(arr) - 1)
        if k <= 0:
            return np.arange(len(arr))
        idx = np.argpartition(-arr, k)[:k]
        return idx[np.argsort(-arr[idx], kind="stable")]

    def partial_skel_all(self, alts):
        alts = [a for a in alts if a]
        if not alts:
            return np.zeros(self.ix.N, np.float32)
        mat = process.cdist(alts, self.ix.sk_flat, scorer=fuzz.partial_ratio, dtype=np.uint8, workers=-1)
        best = mat.max(axis=0).astype(np.float32) / 100.0
        out = np.zeros(self.ix.N, np.float32)
        np.maximum.at(out, self.ix.sk_owner, best * self.sk_len_ok)
        return out

    def candidates(self, q, row):
        ix = self.ix
        if q["gts"]:
            gap, tscore = ix.ton_score_all(q["gts"])
        else:
            gap, tscore = np.full(ix.N, np.inf), np.zeros(ix.N)
        tarr = self.type_arr(row.get("vessel_type"))
        farr = self.flag_arr(row.get("flag_name"))
        struct = tscore + 0.25 * tarr + 0.15 * farr
        tinfo = None
        if q["hang"]:
            sk = ix.skel_sim_all(q["sk"])
            vk = process.cdist([v for v in q["vk"] if v] or [""], ix.vkey,
                               scorer=Levenshtein.normalized_similarity, dtype=np.float32, workers=-1).max(axis=0)
            first = sk + vk
            skp = self.partial_skel_all(q["sk"])
            if q["romv"]:
                romsim = process.cdist(q["romv"], ix.compact, scorer=fuzz.ratio, dtype=np.uint8,
                                       workers=-1).max(axis=0).astype(np.float32) / 100.0
            else:
                romsim = np.zeros(ix.N, np.float32)
            pool = set(self._top(first, POOL).tolist())
            pool.update(self._top(skp + 0.01 * first, POOL_PARTIAL).tolist())
            pool.update(self._top(romsim + 0.01 * first, 400).tolist())
            if q["gts"]:
                pool.update(self._top(tscore + 0.01 * first, 800).tolist())
                pool.update(self._top(struct + 0.01 * first, 150).tolist())
            pool = np.array(sorted(pool))
            lp = np.full(len(pool), -1e9)
            lpl = np.full(len(pool), -1e9)
            for u in q["units"]:
                g, l = self.td.logprob_both(u, pool)
                lp = np.maximum(lp, g)
                lpl = np.maximum(lpl, l)
            lpr = np.full(len(pool), -1e9)
            for u in q["units_red"]:
                g, l = self.td.logprob_both(u, pool)
                lpr = np.maximum(lpr, np.maximum(g, l))
            pmi = lp - PMI_LAMBDA * self.lmv[pool]
            pmil = lpl - PMI_LAMBDA * self.lmv[pool]
            pmir = lpr - PMI_LAMBDA * self.lmv[pool] - 2.0   # hangul word dropped: small fixed cost
            pmir_full = np.full(ix.N, -1e9); pmir_full[pool] = pmir
            pmi_full = np.full(ix.N, -1e9); pmi_full[pool] = pmi
            lp_full = np.full(ix.N, -1e9); lp_full[pool] = lp
            pmil_full = np.full(ix.N, -1e9); pmil_full[pool] = pmil
            pmax = pmi.max()
            nsim = np.maximum(np.clip(1 + (pmi_full - pmax) / 10.0, 0, 1),
                              np.clip(1 + (pmil_full - pmax) / 10.0 - 0.1, 0, 1))
            nsim = np.maximum(nsim, np.clip(1 + (pmir_full - pmax) / 10.0 - 0.15, 0, 1)).astype(np.float32)
            ret = nsim + 0.6 * tscore
            sel = set(self._top(ret, self.k_comb).tolist())
            sel.update(self._top(pmi_full, self.k_pmi).tolist())
            sel.update(self._top(pmil_full, 15).tolist())
            sel.update(self._top(first, 10).tolist())
            if q["gts"]:
                loc_s = np.clip(1 + (pmil_full - pmil.max()) / 10.0, 0, 1)
                sel.update(self._top(loc_s + 0.6 * tscore, 15).tolist())
                sel.update(self._top(romsim + 0.6 * tscore, 10).tolist())
                if q["units_red"]:
                    red_s = np.clip(1 + (pmir_full - pmir.max()) / 10.0, 0, 1)
                    sel.update(self._top(red_s + 0.6 * tscore, 10).tolist())
            if q["units_red"]:
                sel.update(self._top(pmir_full, 10).tolist())
            sel.update(self._top(romsim, 8).tolist())
            order_pmi = np.sort(pmi)[::-1]
            tinfo = dict(pmi=pmi_full, lp=lp_full, pmax=pmax, lmax=lp.max(),
                         second=order_pmi[1] if len(order_pmi) > 1 else order_pmi[0],
                         pool_sorted=order_pmi, pmil=pmil_full, plmax=pmil.max(),
                         pool_sorted_l=np.sort(pmil)[::-1], skp=skp, romsim=romsim,
                         pmir=pmir_full, prmax=pmir.max(), pool_sorted_r=np.sort(pmir)[::-1])
            tset = None
        else:
            nsim = ix.name_sim_all(q["comp"])
            ret = nsim + 0.6 * tscore
            sel = set(self._top(ret, self.k_comb).tolist())
            sel.update(self._top(nsim, self.k_name).tolist())
            if q["comp"] and q["gts"]:
                near = np.flatnonzero(tscore > 0)
                if len(near):
                    prn = process.cdist([q["comp"]], [ix.compact[i] for i in near], scorer=fuzz.partial_ratio,
                                        dtype=np.uint8, workers=-1)[0].astype(np.float32) / 100.0
                    pr_full = np.zeros(ix.N, np.float32)
                    pr_full[near] = prn
                    sel.update(self._top(pr_full + 0.6 * tscore, 15).tolist())
            tset = None
            wq = q["tokens"].split()
            if len(wq) >= 2 and len(wq[0]) == 1:
                isim = process.cdist([q["comp"]], self.init_comp, scorer=fuzz.ratio, dtype=np.uint8,
                                     workers=-1)[0].astype(np.float32) / 100.0
                sel.update(self._top(isim + 0.6 * tscore, 15).tolist())
                sel.update(self._top(isim, 8).tolist())
            if q["comp"] and nsim.max() < EN_PARTIAL_TRIGGER:
                pr = process.cdist([q["comp"]], ix.compact, scorer=fuzz.partial_ratio,
                                   dtype=np.uint8, workers=-1)[0].astype(np.float32) / 100.0
                sel.update(self._top(pr + 0.6 * tscore, 15).tolist())
                sel.update(self._top(pr + 0.01 * nsim, 10).tolist())
                if q["n_tokens"] >= 2:
                    tset = process.cdist([q["tokens"]], ix.tokens, scorer=fuzz.token_set_ratio,
                                         dtype=np.uint8, workers=-1)[0].astype(np.float32) / 100.0
                    sel.update(self._top(tset + 0.6 * tscore, 15).tolist())
                    sel.update(self._top(tset + 0.01 * nsim, 10).tolist())
        if q["gts"]:
            sel.update(self._top(tscore + 0.01 * nsim, self.k_ton).tolist())
            sel.update(self._top(struct + 0.01 * nsim, self.k_struct).tolist())
        cands = np.array(sorted(sel, key=lambda i: (-ret[i], i)))
        return cands, nsim, ret, gap, tscore, tinfo, struct, tarr, tset

    @staticmethod
    def _tok_feats(qt: str, ct: str):
        a = qt.split() if qt else []
        b = ct.split() if ct else []
        if not a or not b:
            return [0.0, 0.0, 0.0]
        sa, sb = set(a), set(b)
        inter = len(sa & sb)
        return [float(a[-1] == b[-1]), inter / len(sa | sb), float(sa <= sb or sb <= sa)]

    # ------------------------------------------------------------------
    def features(self, row: dict, q=None):
        q = q or self.query_info(row)
        cands, nsim, ret, gap, tscore, tinfo, struct, tarr, tset = self.candidates(q, row)
        qcomp_ = q["comp"]
        if tset is not None:
            tsv = tset[cands]
            tset_rank = (tsv[None, :] > tsv[:, None]).sum(axis=1).astype(float)
        else:
            tset_rank = np.full(len(cands), np.nan)
        ix, st = self.ix, self.st
        qt = row.get("vessel_type")
        qf = row.get("flag_name")
        qgt = vm.pos_float(row.get("gross_tonnage"))
        qigt = vm.pos_float(row.get("international_gross_tonnage"))
        cn = nsim[cands]
        name_rank = (cn[None, :] > cn[:, None]).sum(axis=1).astype(float)
        tg = gap[cands]
        tgz = np.nan_to_num(tg, nan=9, posinf=9)
        ton_rank = (tgz[None, :] < tgz[:, None]).sum(axis=1).astype(float)
        cr = ret[cands]
        ret_rank = (cr[None, :] > cr[:, None]).sum(axis=1).astype(float)
        cs = struct[cands]
        struct_rank = (cs[None, :] > cs[:, None]).sum(axis=1).astype(float)
        n_exact = int(np.sum(tg == 0))
        best_name = float(cn.max()) if len(cn) else 0.0
        ties = cn >= best_name - 0.005
        n_ties = int(ties.sum())
        ct = tarr[cands]
        # ranks within the name-tie group (same-name vessels): by type compat and tonnage
        type_rank_t = np.where(ties, (np.where(ties, ct, -1)[None, :] > ct[:, None]).sum(axis=1), -1).astype(float)
        ton_rank_t = np.where(ties, (np.where(ties, tgz, 99)[None, :] < tgz[:, None]).sum(axis=1), -1).astype(float)
        exact_names = sum(1 for i in cands if q["raw"] and ix.rawc[i] == q["raw"])
        if tinfo is not None:
            rs = tinfo["romsim"][cands]
            rom_rank = (rs[None, :] > rs[:, None]).sum(axis=1).astype(float)
            rom_best = float(rs.max()) if len(rs) else 0.0
            ps = tinfo["pool_sorted"]
            n_units = min(len(u) for u in q["units"]) if q["units"] else 0
            margin = float(tinfo["pmax"] - tinfo["second"])
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
                p = tinfo["pmi"][i]
                lpv = tinfo["lp"][i]
                pl = tinfo["pmil"][i]
                tfe = [p - tinfo["pmax"], float(np.searchsorted(-ps, -p, side="left")),
                       lpv - tinfo["lmax"], lpv / max(n_units, 1), self.lmv[i], n_units,
                       len(self.td.letters[i]), margin, p - tinfo["second"],
                       pl - tinfo["plmax"], float(np.searchsorted(-tinfo["pool_sorted_l"], -pl, side="left")),
                       pl - p, (pl + PMI_LAMBDA * self.lmv[i]) / max(n_units, 1)]
                skp_i = float(tinfo["skp"][i])
                rfe = [float(rs[r]), max((fuzz.partial_ratio(v, ccomp) for v in q["romv"]), default=0) / 100.0,
                       rom_rank[r], rom_best - float(rs[r])]
                pr_ = tinfo["pmir"][i]
                v6 = [pr_ - tinfo["prmax"] if q["units_red"] else np.nan,
                      pr_ - max(p, pl) if q["units_red"] else np.nan,
                      float(np.searchsorted(-tinfo["pool_sorted_r"], -pr_, side="left")) if q["units_red"] else np.nan,
                      q["n_tokens"], np.nan]
            else:
                rfe = [np.nan] * 4
                v6 = [np.nan, np.nan, np.nan, q["n_tokens"], tset_rank[r]]
                sk_best = max((Levenshtein.normalized_similarity(a, b) for a in q["sk"] for b in ix.sk[i]), default=0.0)
                sk_d = min((Levenshtein.distance(a, b) for a in q["sk"] for b in ix.sk[i]), default=99)
                vk_sim = Levenshtein.normalized_similarity(q["vk"][0], ix.vkey[i])
                vk_part = fuzz.partial_ratio(q["vk"][0], ix.vkey[i]) / 100
                sk_lq = len(q["sk"][0]) if q["sk"] else 0
                tfe = [np.nan] * 13
                skp_i = np.nan
            cnums = ix.nums[i]
            tp = st.type_prob(qt, ix.ctype[i])
            ttot = st.type_pair.get(qt, {}).get("__TOT__", 0.0)
            gmin = np.nanmin([g1, g2]) if not (np.isnan(g1) and np.isnan(g2)) else np.nan
            qg = qgt or qigt
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
                g1, g2, gmin,
                float(g1 == 0), float(g2 == 0), float(qgt is None and qigt is None), float(not np.isfinite(cgt)),
                math.log(qg / cgt) if qg and np.isfinite(cgt) and cgt > 0 else np.nan,
                math.log1p(cgt) if np.isfinite(cgt) else np.nan,
                ix.dwt[i] / cgt if np.isfinite(cgt) and cgt > 0 and np.isfinite(ix.dwt[i]) else np.nan,
                tp, ttot, st.flag_prob(qf, ix.cflag[i]), float(not isinstance(qf, str)),
                st.cat_type_prior.get(ix.ctype[i], 0.0),
                ret[i], ret_rank[r], name_rank[r], best_name - cn[r], ton_rank[r],
                n_exact, len(cands),
            ] + tfe + self._tok_feats(q["tokens"], ix.tokens[i]) + [
                skp_i, fuzz.partial_token_set_ratio(q["tokens"], ix.tokens[i]) / 100 if q["tokens"] and ix.tokens[i] else 0.0,
                best_name, n_ties, exact_names, struct[i], struct_rank[r], type_rank_t[r], ton_rank_t[r],
            ] + rfe + v6 + [
                fuzz.ratio(qcomp, self.init_comp[i]) / 100 if qcomp and self.init_comp[i] else 0.0,
                float(len(q["tokens"].split()) >= 2 and len(q["tokens"].split()[0]) == 1)])
        return cands, np.asarray(rows, np.float32)
