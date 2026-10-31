# author: opus
"""Candidate retrieval + pairwise features for vessel matching.

Every query row is processed alone. Lookups use only the catalog and train.csv
(official data), so predictions never depend on other test rows.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import os
import re
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

import ko_translit_argon as KA

# Train-alias memory hurt the public score (05: 0.9033 vs 04: 0.9126), so it stays off.
USE_ALIAS = False

# Argon's hand-written dictionary was built from all of train.csv, which leaks
# into CV. Keep it off so CV and production use only fold-learned pairs.
if os.environ.get("KO_DICT", "0") != "1":
    KA._UNIVERSAL_KO_SUBWORDS = []

# ---------------------------------------------------------------- text utils

def is_hangul(s) -> bool:
    return isinstance(s, str) and any("가" <= c <= "힣" for c in s)


def norm(s) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s).upper()
    return "".join(ch for ch in s if ch.isalnum())


def norm_tokens(s) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s).upper()
    s = re.sub(r"[^0-9A-Z가-힣]+", " ", s)
    return " ".join(s.split())


def digits(s) -> str:
    return "".join(re.findall(r"\d+", s)) if isinstance(s, str) else ""


_CHO = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"]
_JUNG = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i"]
_JONG = ["", "k", "k", "k", "n", "n", "n", "t", "l", "l", "l", "l", "l", "l", "l", "l", "m", "p", "p", "s", "s", "ng", "t", "t", "k", "t", "p", "t"]


def romanize(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3:
            o -= 0xAC00
            out.append(_CHO[o // 588] + _JUNG[(o % 588) // 28] + _JONG[o % 28])
        else:
            out.append(ch.lower())
    return "".join(out)


def ko_clean(s: str) -> str:
    """Strip Korean ship-name affixes: trailing 호, 제N prefix kept as digits."""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"제\s*(\d+)", r" \1 ", s)
    s = re.sub(r"호(?=\s|$|\d)", " ", s)
    return " ".join(s.split())


_LETTER_KO = {
    "A": "ei", "B": "bi", "C": "ssi", "D": "di", "E": "i", "F": "epeu", "G": "ji", "H": "eichi",
    "I": "ai", "J": "jei", "K": "kei", "L": "el", "M": "em", "N": "en", "O": "o", "P": "pi",
    "Q": "kyu", "R": "al", "S": "eseu", "T": "ti", "U": "yu", "V": "beui", "W": "deobeulyu",
    "X": "ekseu", "Y": "wai", "Z": "jeteu",
}


def acronym_expand(name: str) -> str:
    """'TY Iris' -> 'tiwai iris' (Korean letter-name reading of acronym tokens)."""
    toks = []
    for t in re.findall(r"[A-Za-z0-9]+", name or ""):
        if t.isupper() and t.isalpha() and len(t) <= 5 and not re.search(r"[AEIOU]{1}[^AEIOU]+[AEIOU]", t):
            toks.append("".join(_LETTER_KO[c] for c in t))
        else:
            toks.append(t.lower())
    return " ".join(toks)


def phon_key(s: str, ko: bool = False) -> str:
    """Crude cross-script consonant skeleton for latin text (romanized Korean or English)."""
    s = s.lower()
    s = re.sub(r"\bno\.?\s*(?=\d)", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    if ko:
        rules = [("ch", "J"), ("ng", "N")]
    else:
        s = re.sub(r"r(?![aeiouy])", "", s)  # non-rhotic r disappears in Korean loanwords
        rules = [
            ("tch", "J"), ("ch", "J"), ("sh", "S"), ("ph", "P"), ("th", "S"), ("ck", "K"), ("ng", "N"), ("gh", ""),
            ("x", "KS"), ("qu", "K"), ("q", "K"), ("c(?=[eiy])", "S"), ("c", "K"), ("g(?=[eiy])", "J"),
        ]
    for a, b in rules:
        s = re.sub(a, b, s)
    table = str.maketrans({"b": "P", "p": "P", "f": "P", "v": "P", "d": "T", "t": "T", "g": "K", "k": "K",
                           "s": "S", "z": "S", "j": "J", "l": "L", "r": "L", "m": "M", "n": "N",
                           "h": "", "w": "", "y": "", "a": "", "e": "", "i": "", "o": "", "u": ""})
    s = s.translate(table)
    return re.sub(r"(.)\1+", r"\1", s)


def vowel_key(s: str) -> str:
    """Skeleton keeping vowel classes, for finer comparison."""
    s = s.lower()
    s = re.sub(r"\bno\.?\s*(?=\d)", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    s = re.sub(r"r(?![aeiouy])", "", s)
    s = re.sub("eo|eu|ou|oo", "U", s)
    s = re.sub("ae|ai|ay|ei|ey", "E", s)
    return phon_key(s) if False else re.sub(r"(.)\1+", r"\1", s.translate(str.maketrans({
        "b": "P", "p": "P", "f": "P", "v": "P", "d": "T", "t": "T", "g": "K", "k": "K", "c": "K", "q": "K",
        "s": "S", "z": "S", "x": "S", "j": "J", "l": "L", "r": "L", "m": "M", "n": "N", "h": "", "w": "",
        "y": "", "a": "A", "e": "E", "i": "I", "o": "O", "u": "U"})))


def ko_keys(ko_name: str) -> tuple[str, str, str]:
    """(romanized, consonant key, vowel key) for a Korean query name."""
    c = ko_clean(ko_name)
    r = romanize(c)
    r = re.sub(r"[^a-z0-9 ]", "", r)
    return r, phon_key(r, ko=True), vowel_key(r)


# ---------------------------------------------------------------- index

def positive(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) and v > 0 else np.nan


def rel_gap(a, b):
    if not (np.isfinite(a) and np.isfinite(b)):
        return 2.0
    return min(abs(a - b) / max(a, b, 1.0), 2.0)


class Index:
    def __init__(self, catalog: pd.DataFrame, train: pd.DataFrame | None = None):
        cat = catalog.reset_index(drop=True)
        self.cat = cat
        self.ids = cat["vessel_id"].astype(str).to_numpy()
        self.id2i = {v: i for i, v in enumerate(self.ids)}
        self.names_raw = cat["vessel_name"].fillna("").astype(str).tolist()
        self.names = [norm(v) for v in self.names_raw]
        self.names_tok = [norm_tokens(v) for v in self.names_raw]
        self.digits = [digits(v) for v in self.names_raw]
        self.ckey = [phon_key(v) for v in self.names_raw]
        self.ckey_acr = [phon_key(acronym_expand(v)) for v in self.names_raw]
        self.vkey = [vowel_key(acronym_expand(v)) for v in self.names_raw]
        self.gt = cat["gross_tonnage"].map(positive).to_numpy(float)
        self.dwt = cat["deadweight_tonnage"].map(positive).to_numpy(float)
        self.ctype = cat["vessel_type"].fillna("").astype(str).to_numpy()
        self.cflag = cat["flag_name"].fillna("").astype(str).to_numpy()
        self.type_codes, self.ctype_code = np.unique(self.ctype, return_inverse=True)
        self.flag_codes, self.cflag_code = np.unique(self.cflag, return_inverse=True)
        name_count = Counter(self.names)
        self.name_freq = np.array([name_count[n] for n in self.names], float)
        self.exact = defaultdict(list)
        for i, n in enumerate(self.names):
            self.exact[n].append(i)
        # "P. HOPE" style records: first letter of the owner word + the remaining words
        self.initial_key = defaultdict(list)
        for i, t in enumerate(self.names_tok):
            w = t.split()
            if len(w) > 1 and w[0][:1].isalpha():
                self.initial_key[w[0][0] + "|" + " ".join(w[1:])].append(i)
        self.sorted_names = sorted((n, i) for i, n in enumerate(self.names) if n)
        self.sorted_name_keys = [n for n, _ in self.sorted_names]
        ok = np.flatnonzero(np.isfinite(self.gt))
        o = np.argsort(self.gt[ok], kind="mergesort")
        self.gt_sorted_idx = ok[o]
        self.gt_sorted = self.gt[self.gt_sorted_idx]
        self.fit_train(train)

    # statistics learned from (a fold of) train.csv
    def fit_train(self, train: pd.DataFrame | None):
        self.cs_map = {}
        self.flag_p = defaultdict(Counter)
        self.type_p = defaultdict(Counter)
        self.ko_tok = defaultdict(Counter)
        self.ko = KA.KoMatcher(self.cat, train)
        self.fit_alias([] if train is None else
                       train[["sample_id", "vessel_name", "target_vessel_id"]].to_dict("records"))
        if train is None or len(train) == 0:
            return
        for r in train.itertuples(index=False):
            ci = self.id2i.get(r.target_vessel_id)
            if ci is None:
                continue
            self.cs_map[r.call_sign] = ci
            self.flag_p[str(r.flag_name)][self.cflag[ci]] += 1
            self.type_p[str(r.vessel_type)][self.ctype[ci]] += 1

    def fit_alias(self, records):
        """Remember how train.csv spelled each vessel (sample_id kept so a training
        row can exclude itself)."""
        self.alias_by_vessel = defaultdict(list)
        self.alias_by_name = defaultdict(list)
        self.alias_names = []
        if not USE_ALIAS:
            return
        for r in records:
            ci = self.id2i.get(r["target_vessel_id"])
            if ci is None:
                continue
            n = norm(unicodedata.normalize("NFC", str(r["vessel_name"])))
            if not n:
                continue
            self.alias_by_vessel[ci].append((r["sample_id"], n))
            self.alias_by_name[n].append((ci, r["sample_id"]))
        self.alias_names = list(self.alias_by_name)

    def _prob_vec(self, table, key, codes):
        """P(catalog value | query value) for every catalog code, smoothed like _prob."""
        c = table.get(key)
        if not c:
            return np.full(len(codes), 0.5)
        tot = sum(c.values()) + 1.0
        return np.array([(c.get(v, 0) + 0.1) / tot for v in codes])

    def meta_candidates(self, gt, igt, vtype, flag, k=40, window=0.06):
        """Ships near the query tonnage ranked by type/flag agreement and tonnage gap:
        the only route to a renamed ship whose name matches nothing."""
        pool = set()
        for v in dict.fromkeys((gt, igt)):
            if np.isfinite(v):
                lo, hi = np.searchsorted(self.gt_sorted, [v * (1 - window), v * (1 + window)])
                pool.update(self.gt_sorted_idx[lo:hi].tolist())
        if not pool:
            return []
        pool = np.fromiter(pool, int)
        tp = self._prob_vec(self.type_p, vtype, self.type_codes)[self.ctype_code[pool]]
        fp = self._prob_vec(self.flag_p, flag, self.flag_codes)[self.cflag_code[pool]]
        g = self.gt[pool]
        gap = np.minimum(np.abs(g - gt) / g if np.isfinite(gt) else 1.0,
                         np.abs(g - igt) / g if np.isfinite(igt) else 1.0)
        score = np.log(tp) + 0.5 * np.log(fp) - 60.0 * gap
        return pool[np.argsort(-score, kind="stable")[:k]].tolist()

    def _prob(self, table, key, value):
        c = table.get(key)
        if not c:
            return -1.0
        return (c.get(value, 0) + 0.1) / (sum(c.values()) + 1.0)

    def prefix_lookup(self, qn, limit=200):
        if len(qn) < 3:
            return []
        import bisect
        lo = bisect.bisect_left(self.sorted_name_keys, qn)
        out = []
        while lo < len(self.sorted_names) and self.sorted_name_keys[lo].startswith(qn) and len(out) < limit:
            out.append(self.sorted_names[lo][1]); lo += 1
        return out

    def gt_exact_ids(self, v, limit=200):
        if not np.isfinite(v):
            return []
        lo, hi = np.searchsorted(self.gt_sorted, [v, v], side="left")[0], np.searchsorted(self.gt_sorted, v, side="right")
        return self.gt_sorted_idx[lo:min(hi, lo + limit)].tolist()

    def gt_neighbors(self, v, k):
        if not np.isfinite(v):
            return []
        p = int(np.searchsorted(self.gt_sorted, v))
        lo, hi = max(0, p - k), min(len(self.gt_sorted), p + k)
        idx = self.gt_sorted_idx[lo:hi]
        d = np.abs(self.gt[idx] - v)
        return idx[np.argsort(d, kind="mergesort")[:k]].tolist()


# ---------------------------------------------------------------- per-query

FEATURES = [
    "is_ko", "exact", "ratio", "pratio", "tsort", "tset", "jw", "lev_norm",
    "ck_ratio", "ck_acr_ratio", "vk_ratio", "ck_jw", "rom_ratio",
    "len_q", "len_c", "len_ratio", "digit_q", "digit_match", "digit_conflict",
    "gt_gap", "igt_gap", "best_gap", "gt_exact", "q_gt_missing", "c_gt_missing", "dwt_gap", "gt_exact_n",
    "flag_p", "type_p", "name_freq", "rank_name", "rank_gt", "name_best", "name_margin",
    "alias_exact", "alias_sim", "alias_n", "alias_amb",
] + list(KA.KoMatcher.FEATURE_NAMES) + ["ko_comb_best", "ko_comb_margin", "is_prefix", "ck_pratio", "len_diff", "ko_comb_rev", "initial_ok"]


def initial_parts(q: str):
    """('P', 'HOPE') for a record like 'P. HOPE' whose first word is a single initial."""
    w = norm_tokens(q).split()
    if len(w) > 1 and len(w[0]) == 1 and w[0].isalpha() and not is_hangul(q):
        return w[0], " ".join(w[1:])
    return None


def query_candidates(ix: Index, row: dict, k_name=30, k_gt=40, k_ko=40):
    """Return (candidate indices, per-candidate retrieval info)."""
    q = unicodedata.normalize("NFC", str(row.get("vessel_name") or ""))
    ko = is_hangul(q)
    qn = norm(q)
    gt = positive(row.get("gross_tonnage"))
    igt = positive(row.get("international_gross_tonnage"))
    cand: dict[int, list] = {}

    def add(i, src_rank=None, gt_rank=None, ko_rank=None):
        e = cand.setdefault(int(i), [999, 999, 999])
        if src_rank is not None:
            e[0] = min(e[0], src_rank)
        if gt_rank is not None:
            e[1] = min(e[1], gt_rank)
        if ko_rank is not None:
            e[2] = min(e[2], ko_rank)

    if ko:
        _, ck, _ = ko_keys(q)
        pool = set()
        for v in dict.fromkeys((gt, igt)):
            if np.isfinite(v):
                lo, hi = np.searchsorted(ix.gt_sorted, [v * 0.88, v * 1.12])
                pool.update(ix.gt_sorted_idx[lo:hi].tolist())
        if ck and pool:
            pool = list(pool)
            keys = [ix.ckey_acr[i] for i in pool]
            for scorer in (fuzz.ratio, fuzz.partial_ratio):
                s = process.cdist([ck], keys, scorer=scorer, workers=1)[0]
                for r, j in enumerate(np.argsort(-s, kind="stable")[:k_ko]):
                    add(pool[j], r)
        for r, (i, _) in enumerate(ix.ko.candidates(q, k=k_ko)):
            add(i, r)
        toks = q.split()
        if len(toks) > 1:  # word order swapped in the record
            for r, (i, _) in enumerate(ix.ko.candidates(" ".join(toks[::-1]), k=k_ko // 2)):
                add(i, r)
        if ck:
            res = process.extract(ck, ix.ckey_acr, scorer=fuzz.ratio, limit=k_ko)
            for r, (_, _, i) in enumerate(res):
                add(i, r)
            res = process.extract(ck, ix.ckey, scorer=fuzz.ratio, limit=k_ko // 2)
            for r, (_, _, i) in enumerate(res):
                add(i, r)
    elif qn:
        for i in ix.exact.get(qn, [])[:20]:
            add(i, 0)
        res = process.extract(qn, ix.names, scorer=fuzz.ratio, limit=k_name)
        for r, (_, _, i) in enumerate(res):
            add(i, r)
        res = process.extract(norm_tokens(q), ix.names_tok, scorer=fuzz.token_sort_ratio, limit=k_name // 2)
        for r, (_, _, i) in enumerate(res):
            add(i, r)
        pool = set()
        for v in dict.fromkeys((gt, igt)):
            if np.isfinite(v):
                lo, hi = np.searchsorted(ix.gt_sorted, [v * 0.88, v * 1.12])
                pool.update(ix.gt_sorted_idx[lo:hi].tolist())
        if pool:
            pool = list(pool)
            names = [ix.names[i] for i in pool]
            for scorer in (fuzz.ratio, fuzz.partial_ratio):
                s = process.cdist([qn], names, scorer=scorer, workers=1)[0]
                for r, j in enumerate(np.argsort(-s, kind="stable")[:k_name]):
                    add(pool[j], r)
            s = process.cdist([norm_tokens(q)], [ix.names_tok[i] for i in pool], scorer=fuzz.token_set_ratio, workers=1)[0]
            for r, j in enumerate(np.argsort(-s, kind="stable")[:k_name // 2]):
                add(pool[j], r)
        ini = initial_parts(q)
        if ini:
            for i in ix.initial_key.get(ini[0] + "|" + ini[1], [])[:50]:
                add(i, 0)
        # truncated / dropped-word names: catalog names that start with the query
        pre = [i for i in ix.prefix_lookup(qn)][:k_name]
        for i in pre:
            add(i, 1)
    sid = row.get("sample_id")
    if qn:
        for ci, s_ in ix.alias_by_name.get(qn, ()):
            if s_ != sid:
                add(ci, 0)
        for name, _, _ in process.extract(qn, ix.alias_names, scorer=fuzz.ratio, limit=10, score_cutoff=60):
            for ci, s_ in ix.alias_by_name[name]:
                if s_ != sid:
                    add(ci, 1)
    for v in dict.fromkeys((gt, igt)):
        for i in ix.gt_exact_ids(v):
            add(i, None, 0)
        for r, i in enumerate(ix.gt_neighbors(v, k_gt)):
            add(i, None, r)
    for r, i in enumerate(ix.meta_candidates(gt, igt, str(row.get("vessel_type")), str(row.get("flag_name")))):
        add(i, None, r)
    return cand


def query_features(ix: Index, row: dict, cand: dict) -> tuple[list[int], np.ndarray]:
    q = unicodedata.normalize("NFC", str(row.get("vessel_name") or ""))
    ko = is_hangul(q)
    idx = list(cand.keys())
    n = len(idx)
    F = np.zeros((n, len(FEATURES)), np.float32)
    col = {f: j for j, f in enumerate(FEATURES)}
    gt = positive(row.get("gross_tonnage"))
    igt = positive(row.get("international_gross_tonnage"))
    flag = str(row.get("flag_name"))
    vtype = str(row.get("vessel_type"))
    cs = ix.cs_map.get(row.get("call_sign"))
    if ko:
        rom, ck, vk = ko_keys(q)
        qd = digits(ko_clean(q))
        qcmp = re.sub(r"[^a-z0-9]", "", rom).upper()
    else:
        qcmp = norm(q)
        qd = digits(q)
        ck = phon_key(q)
        vk = vowel_key(q)
    names = [ix.names[i] for i in idx]
    toks = [ix.names_tok[i] for i in idx]
    F[:, col["ratio"]] = process.cdist([qcmp], names, scorer=fuzz.ratio)[0]
    F[:, col["pratio"]] = process.cdist([qcmp], names, scorer=fuzz.partial_ratio)[0]
    qt = norm_tokens(rom if ko else q)
    F[:, col["tsort"]] = process.cdist([qt], toks, scorer=fuzz.token_sort_ratio)[0]
    F[:, col["tset"]] = process.cdist([qt], toks, scorer=fuzz.token_set_ratio)[0]
    F[:, col["jw"]] = process.cdist([qcmp], names, scorer=JaroWinkler.normalized_similarity)[0]
    F[:, col["lev_norm"]] = process.cdist([qcmp], names, scorer=Levenshtein.normalized_similarity)[0]
    F[:, col["ck_ratio"]] = process.cdist([ck], [ix.ckey[i] for i in idx], scorer=fuzz.ratio)[0]
    F[:, col["ck_acr_ratio"]] = process.cdist([ck], [ix.ckey_acr[i] for i in idx], scorer=fuzz.ratio)[0]
    F[:, col["vk_ratio"]] = process.cdist([vk], [ix.vkey[i] for i in idx], scorer=fuzz.ratio)[0]
    F[:, col["ck_jw"]] = process.cdist([ck], [ix.ckey_acr[i] for i in idx], scorer=JaroWinkler.normalized_similarity)[0]
    F[:, col["rom_ratio"]] = process.cdist([norm(acronym_expand(q)) if not ko else qcmp],
                                           [norm(acronym_expand(ix.names_raw[i])) for i in idx], scorer=fuzz.ratio)[0]
    F[:, col["ck_pratio"]] = process.cdist([ck], [ix.ckey_acr[i] for i in idx], scorer=fuzz.partial_ratio)[0]
    F[:, col["is_ko"]] = ko
    F[:, col["gt_exact_n"]] = len(set(ix.gt_exact_ids(gt)) | set(ix.gt_exact_ids(igt)))
    F[:, col["len_q"]] = len(qcmp)
    F[:, col["digit_q"]] = len(qd)
    qn = norm(q)
    ini = initial_parts(q)
    sid = row.get("sample_id")
    alias_amb = len({ci for ci, s_ in ix.alias_by_name.get(qn, ()) if s_ != sid})
    best = F[:, col["ck_acr_ratio"] if ko else col["ratio"]]
    bmax = best.max() if n else 0
    second = np.partition(best, -2)[-2] if n > 1 else 0
    for j, i in enumerate(idx):
        cg = ix.gt[i]
        F[j, col["exact"]] = float(not ko and qn == ix.names[i])
        lc = len(ix.names[i])
        F[j, col["len_c"]] = lc
        F[j, col["len_ratio"]] = min(lc, len(qcmp)) / max(lc, len(qcmp), 1)
        cd = ix.digits[i]
        F[j, col["digit_match"]] = float(qd == cd)
        F[j, col["digit_conflict"]] = float(bool(qd) and bool(cd) and qd != cd)
        g1, g2 = rel_gap(gt, cg), rel_gap(igt, cg)
        F[j, col["gt_gap"]] = g1
        F[j, col["igt_gap"]] = g2
        F[j, col["best_gap"]] = min(g1, g2)
        F[j, col["gt_exact"]] = float(np.isfinite(cg) and (cg == gt or cg == igt))
        F[j, col["q_gt_missing"]] = float(not np.isfinite(gt) and not np.isfinite(igt))
        F[j, col["c_gt_missing"]] = float(not np.isfinite(cg))
        F[j, col["dwt_gap"]] = min(rel_gap(gt, ix.dwt[i]), rel_gap(igt, ix.dwt[i]))
        F[j, col["flag_p"]] = ix._prob(ix.flag_p, flag, ix.cflag[i])
        F[j, col["type_p"]] = ix._prob(ix.type_p, vtype, ix.ctype[i])
        F[j, col["name_freq"]] = ix.name_freq[i]
        al = [n for s_, n in ix.alias_by_vessel.get(i, ()) if s_ != sid]
        F[j, col["alias_n"]] = len(al)
        F[j, col["alias_exact"]] = sum(n == qn for n in al)
        F[j, col["alias_sim"]] = max((fuzz.ratio(qn, n) for n in al), default=-1)
        F[j, col["alias_amb"]] = alias_amb
        F[j, col["rank_name"]] = cand[i][0]
        F[j, col["rank_gt"]] = cand[i][1]
        F[j, col["is_prefix"]] = float(len(qcmp) >= 3 and ix.names[i].startswith(qcmp))
        if ini:
            cw = ix.names_tok[i].split()
            F[j, col["initial_ok"]] = float(len(cw) > 1 and cw[0][:1] == ini[0]
                                            and fuzz.ratio(" ".join(cw[1:]), ini[1]) >= 80)
        else:
            F[j, col["initial_ok"]] = -1
        F[j, col["len_diff"]] = lc - len(qcmp)
        F[j, col["name_best"]] = best[j] - bmax
        F[j, col["name_margin"]] = bmax - second
    if ko and n:
        K = ix.ko.score_batch(q, idx)
        j0 = col[KA.KoMatcher.FEATURE_NAMES[0]]
        F[:, j0:j0 + K.shape[1]] = K
        comb = K[:, -1]
        F[:, col["ko_comb_best"]] = comb - comb.max()
        F[:, col["ko_comb_margin"]] = comb.max() - (np.partition(comb, -2)[-2] if n > 1 else 0)
        toks = q.split()
        if len(toks) > 1:  # same name with the word order reversed
            F[:, col["ko_comb_rev"]] = ix.ko.score_batch(" ".join(toks[::-1]), idx)[:, -1]
        else:
            F[:, col["ko_comb_rev"]] = comb
    return idx, F


def build(ix: Index, df: pd.DataFrame, with_label: bool):
    rows = df.to_dict("records")
    Xs, groups, cands, labels = [], [], [], []
    for row in rows:
        cand = query_candidates(ix, row)
        idx, F = query_features(ix, row, cand)
        Xs.append(F)
        groups.append(len(idx))
        cands.append(idx)
        if with_label:
            t = ix.id2i.get(row["target_vessel_id"], -1)
            labels.append(np.array([int(i == t) for i in idx], np.int8))
    X = np.vstack(Xs)
    y = np.concatenate(labels) if with_label else None
    return X, y, np.array(groups), cands


def augment_korean(train: pd.DataFrame, seed: int = 0, noisy_copy: bool = True) -> pd.DataFrame:
    """Copies of Korean rows with tonnage removed or perturbed, so the ranker
    learns to rely on the name when tonnage is missing or off."""
    rng = np.random.default_rng(seed)
    ko = train[train["vessel_name"].map(is_hangul)]
    nogt = ko.assign(gross_tonnage=np.nan, international_gross_tonnage=np.nan)
    noisy = ko.copy()
    scale = rng.uniform(0.8, 1.25, len(ko))
    noisy["gross_tonnage"] = noisy["gross_tonnage"] * scale
    noisy["international_gross_tonnage"] = noisy["international_gross_tonnage"] * scale
    return pd.concat([train, nogt] + ([noisy] if noisy_copy else []), ignore_index=True)


_RNG_CHARS = "AEIOURNSTL"


def _typo(s, n, rng):
    s = list(s)
    for _ in range(n):
        if len(s) < 3:
            break
        i = int(rng.integers(len(s))); op = rng.random()
        if op < 0.33:
            del s[i]
        elif op < 0.66:
            s.insert(i, _RNG_CHARS[int(rng.integers(10))] if s[i].isascii() else s[i])
        elif i + 1 < len(s):
            s[i], s[i + 1] = s[i + 1], s[i]
    return "".join(s)


def _corrupt_name(s, rng):
    w = s.split()
    op = int(rng.integers(8))
    if op == 5 and len(w) > 1:
        return " ".join(w[1:])
    if op == 6 and len(w) > 1:  # word order swapped
        return " ".join(w[1:] + w[:1])
    if op == 7 and len(w) > 1 and w[0].isascii():  # owner prefix reduced to an initial
        return " ".join([w[0][0] + "."] + w[1:])
    if op == 0:
        return _typo(s, 1, rng)
    if op == 1:
        return _typo(s, 2, rng)
    if op == 2 and len(w) > 1:
        return " ".join(w[:-1])
    if op == 3 and len(w) > 1:
        return w[0]
    return s[:max(3, int(len(s) * rng.uniform(0.5, 0.8)))]


def augment_corrupt(train: pd.DataFrame, seed: int = 0, frac: float = 0.5) -> pd.DataFrame:
    """Name-corrupted copies (typos, dropped words, truncation) for both languages;
    30% of them also lose tonnage."""
    rng = np.random.default_rng(seed + 100)
    sub = train.sample(frac=frac, random_state=seed + 100).copy()
    sub["vessel_name"] = [_corrupt_name(str(v), rng) for v in sub["vessel_name"]]
    drop = rng.random(len(sub)) < 0.3
    sub.loc[drop, ["gross_tonnage", "international_gross_tonnage"]] = np.nan
    return sub


def augment_rename(train: pd.DataFrame, seed: int = 0, frac: float = 0.15) -> pd.DataFrame:
    """Renamed-ship copies; tonnage, type and flag are kept so the ranker learns to use them
    when the name points nowhere.
      brand (45%): owner prefix swapped, last word kept   (FAIRCHEM ENDURANCE -> GINGA ENDURANCE)
      new   (40%): first word of one ship + last word of another, a name nobody has
      other (15%): another ship's actual name"""
    rng = np.random.default_rng(seed + 200)
    sub = train.sample(frac=frac, random_state=seed + 200).copy()
    ko = train["vessel_name"].map(is_hangul).to_numpy()
    pools = {True: train.loc[ko, ["vessel_name", "target_vessel_id"]].to_numpy(),
             False: train.loc[~ko, ["vessel_name", "target_vessel_id"]].to_numpy()}

    def pick(pool, target):
        for _ in range(10):
            name, t = pool[int(rng.integers(len(pool)))]
            if t != target:
                return str(name)
        return str(name)

    out = []
    for name, target in zip(sub["vessel_name"], sub["target_vessel_id"]):
        pool = pools[is_hangul(name)]
        w = str(name).split()
        mode = rng.random()
        if mode < 0.45 and len(w) > 1:
            out.append(" ".join(pick(pool, target).split()[:-1] or [pick(pool, target).split()[0]]) + " " + w[-1])
        elif mode < 0.85:
            a, b = pick(pool, target).split(), pick(pool, target).split()
            out.append(a[0] + " " + b[-1] if a[0] != b[-1] else a[0])
        else:
            out.append(pick(pool, target))
    sub["vessel_name"] = out
    return sub


def augment_korean_short(train: pd.DataFrame, seed: int = 0, copies: int = 2) -> pd.DataFrame:
    """Two shortened copies of every Korean record, tonnage kept: the shape the ranker misses most
    when tonnage is present. Each copy keeps only the first word, only the last word, drops one
    word, or cuts the end (single-word names are always cut)."""
    rng = np.random.default_rng(seed + 300)
    ko = train[train["vessel_name"].map(is_hangul)]
    out = []
    for c in range(copies):
        names = []
        for name in ko["vessel_name"]:
            w = str(name).split()
            op = int(rng.integers(4)) if len(w) > 1 else 3
            if op == 0:
                new = w[0]
            elif op == 1:
                new = w[-1]
            elif op == 2:
                drop = int(rng.integers(len(w)))
                new = " ".join(w[:drop] + w[drop + 1:])
            else:
                s = str(name).replace(" ", "") if len(w) == 1 else str(name)
                new = s[:max(2, int(round(len(s) * rng.uniform(0.5, 0.8))))].rstrip()
            names.append(new)
        out.append(ko.assign(vessel_name=names))
    return pd.concat(out, ignore_index=True)
