"""Super98 Production Feature Extractor & Hangul Compatibility Jamo Composer (for 92_submit.zip).
100% deterministic, zero pyarrow dependency, stateless row-independent inference mode.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from functools import lru_cache
import re
import sys
import unicodedata
import numpy as np
from rapidfuzz import fuzz, process

FEATURE_NAMES = [
    "ko_v2_en2ko_sim",
    "ko_v2_lr_sim",
    "ko_v2_cons_sim",
    "ko_v2_tail_sim",
    "ko_v2_q_cov",
    "ko_short_tok_cov",
    "ko_v2_combined",
    "ko_v2_comb_gain",
    "trans_ratio",
    "trans_tsort",
    "first_tok_sim",
    "last_tok_or_suffix_sim",
    "unexplained_q_ratio",
    "gt_win_best",
    "gt_win_margin",
    "gt_win_zscore",
    "gt_win_broad_best",
    "gt_win_broad_margin",
    "gt_win_tail_best",
    "within_gt_rank",
    "meta_compat_score",
    "meta_compat_best",
    "igt_prefer_gap",
    "cat_type_unit_prior",
]
assert len(FEATURE_NAMES) == 24

CLEAN18_DROP_COLS = {
    "unexplained_q_ratio",
    "gt_win_zscore",
    "gt_win_broad_best",
    "gt_win_broad_margin",
    "within_gt_rank",
    "meta_compat_score",
}
CLEAN18_KEEP_IDX = [i for i, f in enumerate(FEATURE_NAMES) if f not in CLEAN18_DROP_COLS]
CLEAN18_KEEP_NAMES = [FEATURE_NAMES[i] for i in CLEAN18_KEEP_IDX]
assert len(CLEAN18_KEEP_IDX) == 18

_CHOSEONG = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
_JUNGSEONG = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]
_JONGSEONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
_CHO_MAP = {c: i for i, c in enumerate(_CHOSEONG)}
_JUNG_MAP = {c: i for i, c in enumerate(_JUNGSEONG)}
_JONG_MAP = {c: i for i, c in enumerate(_JONGSEONG) if c}
_COMPOUND_JONG = {
    ("ㄱ", "ㅅ"): "ㄳ", ("ㄴ", "ㅈ"): "ㄵ", ("ㄴ", "ㅎ"): "ㄶ",
    ("ㄹ", "ㄱ"): "ㄺ", ("ㄹ", "ㅁ"): "ㄻ", ("ㄹ", "ㅂ"): "ㄼ", ("ㄹ", "ㅅ"): "ㄽ", ("ㄹ", "ㅌ"): "ㄾ", ("ㄹ", "ㅍ"): "ㄿ", ("ㄹ", "ㅎ"): "ㅀ",
    ("ㅂ", "ㅅ"): "ㅄ",
}
_COMPOUND_JUNG = {
    ("ㅗ", "ㅏ"): "ㅘ", ("ㅗ", "ㅐ"): "ㅙ", ("ㅗ", "ㅣ"): "ㅚ",
    ("ㅜ", "ㅓ"): "ㅝ", ("ㅜ", "ㅔ"): "ㅞ", ("ㅜ", "ㅣ"): "ㅟ",
    ("ㅡ", "ㅣ"): "ㅢ",
}


def compose_compat_jamo(s: str) -> str:
    """Deterministically compose Hangul Compatibility Jamo (\u3131..\u3163) into standard Hangul syllables."""
    if not any("\u3131" <= c <= "\u3163" for c in s):
        return s
    chars = list(s)
    merged_v = []
    i = 0
    while i < len(chars):
        if i + 1 < len(chars) and (chars[i], chars[i + 1]) in _COMPOUND_JUNG:
            merged_v.append(_COMPOUND_JUNG[(chars[i], chars[i + 1])])
            i += 2
        else:
            merged_v.append(chars[i])
            i += 1
    chars = merged_v
    out = []
    n = len(chars)
    i = 0
    while i < n:
        c = chars[i]
        if c in _CHO_MAP and i + 1 < n and chars[i + 1] in _JUNG_MAP:
            L = _CHO_MAP[c]
            V = _JUNG_MAP[chars[i + 1]]
            T = 0
            i += 2
            if i < n and chars[i] in _JONG_MAP:
                if i + 1 < n and chars[i + 1] in _JUNG_MAP:
                    pass
                elif i + 1 < n and (chars[i], chars[i + 1]) in _COMPOUND_JONG and not (i + 2 < n and chars[i + 2] in _JUNG_MAP):
                    T = _JONG_MAP[_COMPOUND_JONG[(chars[i], chars[i + 1])]]
                    i += 2
                else:
                    T = _JONG_MAP[chars[i]]
                    i += 1
            out.append(chr(0xAC00 + (L * 21 + V) * 28 + T))
        else:
            out.append(c)
            i += 1
    return "".join(out)


@lru_cache(maxsize=131072)
def en_to_ko_v2(en_body: str, keep_r: bool = False, alt_th_ch: bool = False) -> str:
    s = en_body.lower()
    s = re.sub(r"\bkn", "n", s)
    s = re.sub(r"\bwr", "r", s)
    s = re.sub(r"\bgh", "g", s)
    s = re.sub(r"dge\b", "ji", s)
    s = re.sub(r"dge|dg(?=[eiy])", "je", s)
    s = re.sub(r"stle\b", "seul", s)
    s = re.sub(r"\bchem", "kem", s)
    s = re.sub(r"\bchr", "kl", s)
    s = re.sub(r"arch", "ak", s)
    s = re.sub(r"\bex", "iks", s)
    s = re.sub(r"\bdia", "daia", s)
    s = re.sub(r"woo", "u", s)
    s = re.sub(r"\bwu", "u", s)
    s = re.sub(r"hyun", "hyeon", s)
    s = re.sub(r"hyoung|hyung", "hyeong", s)
    s = re.sub(r"young|yung", "yeong", s)
    s = re.sub(r"tion\b", "syeon", s)
    s = re.sub(r"sion\b", "jeon", s)
    s = re.sub(r"liance\b", "laieonseu", s)
    s = re.sub(r"ture\b", "cheo", s)
    s = re.sub(r"shou", "seou", s)
    s = re.sub(r"zh", "j", s)
    s = re.sub(r"ph", "p", s)
    if alt_th_ch:
        s = re.sub(r"tch", "ch", s)
        s = re.sub(r"ch", "k", s)
        s = re.sub(r"th", "t", s)
    else:
        s = re.sub(r"tch|ch", "ch", s)
        s = re.sub(r"th", "s", s)
    s = re.sub(r"sh(?=[iu])", "sy", s)
    s = re.sub(r"sh", "s", s)
    s = re.sub(r"ck|qu", "k", s)
    s = re.sub(r"x(?=[iu])", "sy", s)
    s = re.sub(r"x", "ks", s)
    s = re.sub(r"q(?=[iu])", "ch", s)
    s = re.sub(r"c(?=[eiy])", "s", s)
    s = re.sub(r"c(?!h)", "k", s)
    s = re.sub(r"g(?=[eiy])", "j", s)
    s = re.sub(r"v", "b", s)
    s = re.sub(r"f", "p", s)
    s = re.sub(r"z", "j", s)
    s = re.sub(r"ear(?=[^aeiouy\s])", "eol", s)
    s = re.sub(r"eer\b|ier\b|ear\b", "ieo", s)
    if keep_r:
        s = re.sub(r"r(?![aeiouy])", "l", s)
    else:
        s = re.sub(r"(?:er|or|ar|ur|our)\b", "eo", s)
        s = re.sub(r"ar(?![aeiouy])", "a", s)
        s = re.sub(r"or(?![aeiouy])", "o", s)
        s = re.sub(r"(?:ir|ur|er)(?![aeiouy])", "eo", s)
    s = re.sub(r"ee|ea", "i", s)
    s = re.sub(r"oo|ou", "u", s)
    s = re.sub(r"ay\b|(?<![whktsmdlnpb\s])ai(?=[lnrt])", "ei", s)
    s = re.sub(r"igh", "ai", s)
    s = re.sub(r"gh\b|gh(?=[^aeiouy])", "", s)
    s = re.sub(r"oa|ow", "o", s)
    s = re.sub(r"au|aw", "o", s)
    s = re.sub(r"r", "l", s)
    s = re.sub(r"([bdfgjklmnprst])\1+", r"\1", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


@lru_cache(maxsize=131072)
def en_to_lr(en_body: str) -> str:
    s = en_body.lower()
    s = re.sub(r"\bkn", "n", s)
    s = re.sub(r"\bwr", "r", s)
    s = re.sub(r"\bgh", "g", s)
    s = re.sub(r"dge\b", "ji", s)
    s = re.sub(r"dge|dg(?=[eiy])", "je", s)
    s = re.sub(r"ph", "p", s)
    s = re.sub(r"tch|ch", "ch", s)
    s = re.sub(r"sh", "s", s)
    s = re.sub(r"th", "s", s)
    s = re.sub(r"ck|qu", "k", s)
    s = re.sub(r"c(?=[eiy])", "s", s)
    s = re.sub(r"c(?!h)", "k", s)
    s = re.sub(r"g(?=[eiy])", "j", s)
    s = re.sub(r"v", "b", s)
    s = re.sub(r"f", "p", s)
    s = re.sub(r"z", "j", s)
    s = re.sub(r"ear(?=[^aeiouy\s])", "eor", s)
    s = re.sub(r"(?:er|or|ar|ur|our)\b", "eo", s)
    s = re.sub(r"ar(?![aeiouy])", "a", s)
    s = re.sub(r"or(?![aeiouy])", "o", s)
    s = re.sub(r"(?:ir|ur|er)(?![aeiouy])", "eo", s)
    s = re.sub(r"ee|ea", "i", s)
    s = re.sub(r"oo|ou", "u", s)
    s = re.sub(r"igh", "ai", s)
    s = re.sub(r"gh\b|gh(?=[^aeiouy])", "", s)
    s = re.sub(r"oa|ow|au|aw", "o", s)
    s = re.sub(r"([bdfgjklmnprst])\1+", r"\1", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


@lru_cache(maxsize=65536)
def ko_rom_lr(rom_no_eu: str) -> str:
    s = re.sub(r"(?<=[aeiouy])l(?=[aeiouy])", "r", rom_no_eu)
    s = re.sub(r"l+", "l", s)
    return s


def _sim(a: str, b: str) -> float:
    return fuzz.ratio(a, b) / 100.0 if (a and b) else 0.0


def _qcov(q: str, c: str) -> float:
    if not q or not c:
        return 0.0
    pr = fuzz.partial_ratio(q, c) / 100.0
    if len(c) < len(q):
        return max(_sim(q, c), pr * (len(c) / len(q)))
    return pr


def _lcp_ratio(q: str, c: str) -> float:
    if not q or not c:
        return 0.0
    m = 0
    for a, b in zip(q, c):
        if a != b:
            break
        m += 1
    return m / len(q)


class SuperFeatureExtractor:
    """Precomputes catalog V2 projections and vessel/unit priors once."""

    def __init__(
        self,
        ix,
        cat_df,
        fit_df,
        base_feature_names: list[str],
        fit_vids: set[str] | None = None,
        type_unit_counts: dict[str, dict[str, int]] | None = None,
        stateless: bool = False,
    ):
        self.ix = ix
        self.km = ix.ko
        self.col = {f: i for i, f in enumerate(base_feature_names)}
        self.n_cat = len(cat_df)
        self.stateless = stateless

        ka_mod = sys.modules[ix.ko.__class__.__module__]
        self.ka = ka_mod

        c_v2_std = []
        c_v2_alt = []
        c_v2_lr = []
        c_v2_cons = []
        c_comp_degem = []
        c_tok_v2 = []
        c_first_tok_upper = []
        c_last_tok_upper = []

        cat_names = cat_df["vessel_name"].fillna("").astype(str).tolist()
        for i, raw in enumerate(cat_names):
            body, _ = ka_mod.extract_digits_and_clean_en(raw, join_hyphen=True)
            acr = ka_mod.expand_en_acronyms(body)
            s1 = en_to_ko_v2(acr, False, False)
            s2 = en_to_ko_v2(acr, True, True)
            slr = en_to_lr(acr)
            c_v2_std.append(s1)
            c_v2_alt.append(s2)
            c_v2_lr.append(slr)
            c_v2_cons.append(ka_mod.canonical_cons_key(s1, keep_h=True, soft_g=True))
            c_comp_degem.append(re.sub(r"([bdfgjklmnprst])\1+", r"\1", self.km.cat_en_compact[i]))
            toks = body.lower().split()
            c_tok_v2.append([
                (
                    t,
                    en_to_ko_v2(t, False, False),
                    en_to_ko_v2(t, True, True),
                    en_to_lr(t),
                    ka_mod.canonical_cons_key(t, keep_h=True, soft_g=True),
                )
                for t in toks
            ])
            utoks = ix.names_tok[i].split()
            c_first_tok_upper.append(utoks[0] if utoks else "")
            c_last_tok_upper.append(utoks[-1] if utoks else "")

        self.c_v2_std = c_v2_std
        self.c_v2_alt = c_v2_alt
        self.c_v2_lr = c_v2_lr
        self.c_v2_cons = c_v2_cons
        self.c_comp_degem = c_comp_degem
        self.c_tok_v2 = c_tok_v2
        self.c_first_tok_upper = c_first_tok_upper
        self.c_last_tok_upper = c_last_tok_upper

        c_units = cat_df["unit"].fillna("MISSING").astype(str).tolist()
        c_types = cat_df["vessel_type"].fillna("MISSING").astype(str).tolist()
        c_dwt = np.asarray(cat_df["deadweight_tonnage"], dtype=np.float64)
        c_sz = np.asarray(cat_df["size"], dtype=np.float64)
        c_dwt_miss = ~np.isfinite(c_dwt)
        c_sz_miss = ~np.isfinite(c_sz)
        self.c_units = c_units

        if fit_vids is not None:
            vids_set = {str(v) for v in fit_vids}
        elif fit_df is not None and len(fit_df):
            vids_set = set(fit_df["target_vessel_id"].astype(str))
        else:
            vids_set = set()
        self.fit_vids = vids_set
        in_fit = np.array([str(vid) in vids_set for vid in cat_df["vessel_id"]], dtype=np.float32)

        sector_keys = [
            f"{c_types[i]}|{c_units[i]}|{int(c_dwt_miss[i])}|{int(c_sz_miss[i])}"
            for i in range(self.n_cat)
        ]
        sec_tot = Counter()
        sec_fit = Counter()
        for i, sk in enumerate(sector_keys):
            sec_tot[sk] += 1
            if in_fit[i] > 0.5:
                sec_fit[sk] += 1

        base_rate = (float(in_fit.sum()) + 1.0) / (self.n_cat + 20.0)
        c_sector_log_prior = np.zeros(self.n_cat, dtype=np.float32)
        for i, sk in enumerate(sector_keys):
            num = sec_fit[sk] - in_fit[i] + 10.0 * base_rate
            den = sec_tot[sk] - 1.0 + 10.0
            p_sec = max(num / max(den, 1.0), 1e-4)
            c_sector_log_prior[i] = float(np.log(p_sec / base_rate))
        self.c_sector_log_prior = c_sector_log_prior

        self.type_unit_counts = defaultdict(Counter)
        self.unit_set = sorted(set(c_units))
        if type_unit_counts is not None:
            for k, d in type_unit_counts.items():
                self.type_unit_counts[str(k)] = Counter({str(uk): int(uv) for uk, uv in d.items()})
        elif fit_df is not None and len(fit_df):
            for r in fit_df.itertuples(index=False):
                ci = ix.id2i.get(str(r.target_vessel_id))
                if ci is not None:
                    self.type_unit_counts[str(r.vessel_type)][c_units[ci]] += 1

        self._ko_pair_cache: dict[tuple[str, int], tuple[float, float, float, float, float, float, float]] = {}
        self._qinfo_cache: dict[str, dict] = {}

    def export_stats(self) -> dict:
        return {
            "fit_vids": sorted(self.fit_vids),
            "type_unit_counts": {k: dict(v) for k, v in self.type_unit_counts.items()},
        }

    def _get_qinfo(self, q_name: str) -> dict:
        if not self.stateless:
            qi = self._qinfo_cache.get(q_name)
            if qi is not None:
                return qi
        qi = self.km.translate_ko(q_name)
        q_tr = re.sub(r"l+", "l", qi["trans"])
        q_rom = re.sub(r"l+", "l", qi["rom_no_eu"])
        qi["q_tr_degem"] = q_tr
        qi["q_rom_degem"] = q_rom
        qi["q_rom_ae"] = re.sub(r"ae", "a", q_rom)
        qi["q_lr"] = ko_rom_lr(qi["rom_no_eu"])
        qi["q_lr_ae"] = re.sub(r"ae", "a", qi["q_lr"])
        qi["q_cons_r"] = self.ka.canonical_cons_key(q_rom, keep_h=True, soft_g=True)
        qi["q_cons_t"] = self.ka.canonical_cons_key(q_tr or q_rom, keep_h=True, soft_g=True)
        rom_raw = re.sub(r"[^a-z0-9]", "", self.ka.romanize_hangul(q_name, drop_eu=False)).upper()
        qi["qcmp_trans"] = (qi["trans"] or rom_raw).upper()
        qi["qt_trans"] = (
            " ".join(t.upper() for t in qi["trans_tokens"])
            if qi["trans_tokens"]
            else rom_raw
        )
        if not self.stateless:
            self._qinfo_cache[q_name] = qi
        return qi

    def _score_ko_v2_single(self, q_name: str, qi: dict, idx: int, old_row: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
        if not self.stateless:
            ckey = (q_name, idx)
            cached = self._ko_pair_cache.get(ckey)
            if cached is not None:
                return cached

        col = self.col
        old_tr_sim = float(old_row[col["ko_trans_sim"]])
        old_tr_jw = float(old_row[col["ko_trans_jw"]])
        old_e2k = float(old_row[col["ko_en2ko_sim"]])
        old_cons = float(old_row[col["ko_cons_sim"]])
        old_cons_acr = float(old_row[col["ko_cons_acr_sim"]])
        old_phon = float(old_row[col["ko_phon_sim"]])
        old_pref = float(old_row[col["ko_prefix_sim"]])
        old_tail = float(old_row[col["ko_tail_sim"]])
        old_dig = float(old_row[col["ko_digit_match"]])

        q_tr = qi["q_tr_degem"]
        q_rom = qi["q_rom_degem"]
        q_rom_ae = qi["q_rom_ae"]
        q_lr = qi["q_lr"]
        q_lr_ae = qi["q_lr_ae"]
        q_cons_r = qi["q_cons_r"]
        q_cons_t = qi["q_cons_t"]

        c_comp = self.c_comp_degem[idx]
        cs1 = self.c_v2_std[idx]
        cs2 = self.c_v2_alt[idx]
        clr = self.c_v2_lr[idx]
        ccons = self.c_v2_cons[idx]
        ctoks = self.c_tok_v2[idx]

        lr_sim = max(_sim(q_lr, clr), _sim(q_lr_ae, clr))
        e2k_v2 = max(
            old_e2k,
            _sim(q_tr, cs1),
            _sim(q_tr, cs2),
            _sim(q_rom, cs1),
            _sim(q_rom, cs2),
            _sim(q_rom_ae, cs1),
            _sim(q_rom_ae, cs2),
            lr_sim,
        )
        cons_v2 = max(old_cons, old_cons_acr, _sim(q_cons_t, ccons), _sim(q_cons_r, ccons))

        tok_proj = qi["tok_projections"]
        tail_v2 = old_tail
        if len(tok_proj) >= 2 and len(ctoks) >= 2:
            qt_l, qr_l, _ = tok_proj[-1]
            qr_l_clean = re.sub(r"l+", "l", qr_l)
            qr_l_ae = re.sub(r"ae", "a", qr_l_clean)
            qr_lr = ko_rom_lr(qr_l)
            qc_rom = self.ka.canonical_cons_key(qr_l, keep_h=True, soft_g=True)
            ct_l, ce1_l, ce2_l, clr_l, cc_l = ctoks[-1]
            cons_w = 0.78 if min(len(qc_rom), len(cc_l)) <= 2 else 0.88
            t_sc = max(
                _sim(qt_l, ct_l),
                _sim(qt_l, ce1_l),
                _sim(qr_l_clean, ce1_l),
                _sim(qr_l_clean, ce2_l),
                _sim(qr_l_ae, ce1_l),
                _sim(qr_lr, clr_l),
                cons_w * _sim(qc_rom, cc_l),
            )
            if 1 <= len(qr_l) <= 4:
                pref = max(
                    _lcp_ratio(qt_l, ct_l),
                    _lcp_ratio(qr_l_clean, ce1_l),
                    _lcp_ratio(qr_l_clean, ce2_l),
                )
                t_sc = max(t_sc, 0.88 * pref)
            if qr_l and ct_l and (qr_l[0] in "aeiouwy") != (ct_l[0] in "aeiouwy"):
                t_sc *= 0.78
            tail_v2 = t_sc
        elif len(tok_proj) == 1 and len(ctoks) >= 2:
            ct0, ce1_0, ce2_0, clr_0, _ = ctoks[0]
            ct_rest = "".join(x[0] for x in ctoks[1:])
            ce1_rest = "".join(x[1] for x in ctoks[1:])
            ce2_rest = "".join(x[2] for x in ctoks[1:])
            clr_rest = "".join(x[3] for x in ctoks[1:])
            for q_s, c0_s, cr_s in [(qi["trans"], ct0, ct_rest), (q_rom, ce1_0, ce1_rest), (q_lr, clr_0, clr_rest)]:
                if len(c0_s) >= 2 and len(q_s) > len(c0_s) and _sim(q_s[:len(c0_s)], c0_s) >= 0.75:
                    q_rem = q_s[len(c0_s):]
                    rem_sc = max(_sim(q_rem, ct_rest), _sim(q_rem, ce1_rest), _sim(q_rem, ce2_rest), _sim(q_rem, clr_rest))
                    if q_rem and cr_s and (q_rem[0] in "aeiouwy") != (cr_s[0] in "aeiouwy"):
                        rem_sc *= 0.78
                    tail_v2 = max(tail_v2, rem_sc)

        short_cov = 0.0
        if tok_proj and ctoks:
            for qt_t, qr_t, _ in tok_proj:
                if not (1 <= len(qr_t) <= 3):
                    continue
                best_s = 0.0
                for ct_i, ce1_i, ce2_i, _, _ in ctoks:
                    if len(ct_i) < 2:
                        continue
                    if (qt_t and ct_i.startswith(qt_t)) or (qr_t and (ce1_i.startswith(qr_t) or ce2_i.startswith(qr_t))):
                        sc = 0.86
                        if bool(qr_t and qr_t[0] in "aeiouwy") != bool(ct_i and ct_i[0] in "aeiouwy"):
                            sc *= 0.85
                        if sc > best_s:
                            best_s = sc
                short_cov += best_s
            short_cov /= len(tok_proj)

        qcov_v2 = max(
            _qcov(q_tr, c_comp),
            _qcov(q_tr, cs1),
            _qcov(q_rom, cs1),
            _qcov(q_rom, cs2),
            _qcov(q_lr, clr),
        )

        dig_v2 = old_dig
        c_dig = self.km.cat_digits[idx]
        if not qi["digits"] and c_dig == "2" and qi["ko_nospace"].endswith("이") and len(qi["ko_nospace"]) >= 3:
            q_stripped = self._get_qinfo(qi["ko_nospace"][:-1])
            if max(_sim(q_stripped["q_tr_degem"], cs1), _sim(q_stripped["q_rom_ae"], cs1)) >= 0.65:
                dig_v2 = 1.0
                e2k_v2 = max(e2k_v2, _sim(q_stripped["q_rom_ae"], cs1))

        comb_v2 = (
            0.22 * max(old_tr_sim, e2k_v2)
            + 0.12 * old_tr_jw
            + 0.20 * e2k_v2
            + 0.12 * cons_v2
            + 0.10 * old_phon
            + 0.12 * tail_v2
            + 0.12 * qcov_v2
            + 0.04 * old_pref
            + 0.08 * dig_v2
        )
        res = (float(e2k_v2), float(lr_sim), float(cons_v2), float(tail_v2), float(qcov_v2), float(short_cov), float(comb_v2))
        if not self.stateless and len(self._ko_pair_cache) < 600000:
            self._ko_pair_cache[(q_name, idx)] = res
        return res

    def compute_query(self, row: dict, cand_indices: np.ndarray | list[int], F_base: np.ndarray) -> np.ndarray:
        n = len(cand_indices)
        out = np.zeros((n, 24), dtype=np.float32)
        if n == 0:
            return out

        col = self.col
        q = unicodedata.normalize("NFC", str(row.get("vessel_name") or ""))
        ko = bool(F_base[0, col["is_ko"]] > 0.5)

        if ko:
            qi = self._get_qinfo(q)
            qcmp_tr = qi["qcmp_trans"]
            qt_tr = qi["qt_trans"]
            names = [self.ix.names[int(i)] for i in cand_indices]
            toks = [self.ix.names_tok[int(i)] for i in cand_indices]
            out[:, 8] = process.cdist([qcmp_tr], names, scorer=fuzz.ratio)[0]
            out[:, 9] = process.cdist([qt_tr], toks, scorer=fuzz.token_sort_ratio)[0]

            tok_proj = qi["tok_projections"]
            q_first = tok_proj[0][0] if tok_proj else qi["q_tr_degem"]
            q_last = tok_proj[-1][0] if tok_proj else qi["q_tr_degem"]
            q_first_rom = tok_proj[0][1] if tok_proj else qi["q_rom_degem"]
            q_last_rom = tok_proj[-1][1] if tok_proj else qi["q_rom_degem"]
            len_q_eff = max(len(qi["trans"]), 1)

            for j, c_idx in enumerate(cand_indices):
                ci = int(c_idx)
                e2k_v2, lr_sim, cons_v2, tail_v2, qcov_v2, short_cov, comb_v2 = self._score_ko_v2_single(q, qi, ci, F_base[j])
                out[j, 0] = e2k_v2
                out[j, 1] = lr_sim
                out[j, 2] = cons_v2
                out[j, 3] = tail_v2
                out[j, 4] = qcov_v2
                out[j, 5] = short_cov
                out[j, 6] = comb_v2
                out[j, 7] = comb_v2 - F_base[j, col["ko_combined_score"]]

                ctoks = self.c_tok_v2[ci]
                if ctoks:
                    ct0, ce1_0, ce2_0, _, _ = ctoks[0]
                    ctL, ce1_L, ce2_L, _, _ = ctoks[-1]
                    out[j, 10] = max(_sim(q_first, ct0), _sim(q_first_rom, ce1_0), _sim(q_first_rom, ce2_0))
                    out[j, 11] = max(_sim(q_last, ctL), _sim(q_last_rom, ce1_L), _sim(q_last_rom, ce2_L), tail_v2)
                c_len = len(self.km.cat_en_compact[ci])
                out[j, 12] = max(0.0, float(len_q_eff - c_len)) / float(len_q_eff)
        else:
            out[:, 8] = F_base[:, col["ratio"]]
            out[:, 9] = F_base[:, col["tsort"]]
            w_toks = re.sub(r"[^A-Z0-9]+", " ", q.upper()).split()
            q_first_u = w_toks[0] if w_toks else ""
            q_last_u = w_toks[-1] if w_toks else ""
            qcmp_u = re.sub(r"[^A-Z0-9]", "", q.upper())
            len_q_eff = max(len(qcmp_u), 1)
            for j, c_idx in enumerate(cand_indices):
                ci = int(c_idx)
                cf = self.c_first_tok_upper[ci]
                cl = self.c_last_tok_upper[ci]
                cname = self.ix.names[ci]
                f_sim = _sim(q_first_u, cf) if (q_first_u and cf) else 0.0
                if len(qcmp_u) >= 3 and cname.startswith(qcmp_u):
                    f_sim = 1.0
                l_sim = _sim(q_last_u, cl) if (q_last_u and cl) else 0.0
                if len(qcmp_u) >= 3 and cname.endswith(qcmp_u):
                    l_sim = 1.0
                elif len(q_first_u) >= 2 and cname.startswith(q_first_u) and len(qcmp_u) > len(q_first_u) and len(cname) > len(q_first_u):
                    l_sim = max(l_sim, _sim(qcmp_u[len(q_first_u):], cname[len(q_first_u):]))
                out[j, 10] = f_sim
                out[j, 11] = l_sim
                out[j, 12] = max(0.0, float(len_q_eff - len(cname))) / float(len_q_eff)

        bg = F_base[:, col["best_gap"]]
        q_miss = bool(F_base[0, col["q_gt_missing"]] > 0.5)
        if not q_miss:
            win_mask = bg <= 0.09
            if not np.any(win_mask):
                win_mask = bg <= 0.25
            if not np.any(win_mask):
                win_mask = np.ones(n, dtype=bool)
        else:
            win_mask = np.ones(n, dtype=bool)

        win_idx = np.where(win_mask)[0]
        if ko:
            prim = out[:, 6]
            broad = np.maximum(prim, np.maximum(out[:, 4], out[:, 11]))
            tail_s = out[:, 3]
        else:
            rat01 = F_base[:, col["ratio"]] / 100.0
            prat01 = (F_base[:, col["pratio"]] / 100.0) * np.minimum(1.0, F_base[:, col["len_c"]] / np.maximum(F_base[:, col["len_q"]], 1.0))
            tset01 = F_base[:, col["tset"]] / 100.0
            prim = rat01
            broad = np.maximum(rat01, np.maximum(0.92 * prat01, 0.95 * tset01))
            tail_s = out[:, 11]

        w_prim = prim[win_idx]
        p_max = float(w_prim.max())
        p_mean = float(w_prim.mean())
        p_std = float(w_prim.std()) + 1e-3
        if len(w_prim) >= 2:
            top2 = np.partition(w_prim, -2)[-2:]
            p_1, p_2 = float(top2[1]), float(top2[0])
        else:
            p_1 = p_2 = p_max
        out[:, 13] = prim - p_max
        is_p_top1 = win_mask & (prim >= p_1 - 1e-9)
        out[:, 14] = np.where(is_p_top1, prim - p_2, prim - p_1)
        out[:, 15] = (prim - p_mean) / p_std

        w_broad = broad[win_idx]
        b_max = float(w_broad.max())
        if len(w_broad) >= 2:
            btop2 = np.partition(w_broad, -2)[-2:]
            b_1, b_2 = float(btop2[1]), float(btop2[0])
        else:
            b_1 = b_2 = b_max
        out[:, 16] = broad - b_max
        is_b_top1 = win_mask & (broad >= b_1 - 1e-9)
        out[:, 17] = np.where(is_b_top1, broad - b_2, broad - b_1)

        out[:, 18] = tail_s - float(tail_s[win_idx].max())

        order_bg = np.argsort(bg, kind="mergesort")
        ranks_bg = np.empty(n, dtype=np.float32)
        ranks_bg[order_bg] = np.arange(n, dtype=np.float32)
        out[:, 19] = ranks_bg

        tp = np.maximum(F_base[:, col["type_p"]], 0.005)
        fp = np.maximum(F_base[:, col["flag_p"]], 0.005)
        meta_sc = np.log(tp) + 0.5 * np.log(fp) - 40.0 * np.minimum(bg, 0.30)
        out[:, 20] = meta_sc
        out[:, 21] = meta_sc - float(meta_sc.max())

        r_gt = row.get("gross_tonnage")
        r_igt = row.get("international_gross_tonnage")
        try:
            g_f = float(r_gt) if r_gt is not None else np.nan
        except (ValueError, TypeError):
            g_f = np.nan
        try:
            ig_f = float(r_igt) if r_igt is not None else np.nan
        except (ValueError, TypeError):
            ig_f = np.nan
        if np.isfinite(g_f) and g_f > 0 and np.isfinite(ig_f) and ig_f > 0 and abs(g_f - ig_f) > 1.0:
            out[:, 22] = F_base[:, col["igt_gap"]]
        else:
            out[:, 22] = bg

        vtype_str = str(row.get("vessel_type"))
        u_cnt = self.type_unit_counts.get(vtype_str)
        u_tot = (sum(u_cnt.values()) + 2.0) if u_cnt else 0.0
        for j, c_idx in enumerate(cand_indices):
            ci = int(c_idx)
            if u_cnt:
                u_p = (u_cnt.get(self.c_units[ci], 0) + 0.2) / u_tot
                u_log = float(np.log(max(u_p, 1e-3)))
            else:
                u_log = -0.7
            out[j, 23] = u_log + 0.5 * self.c_sector_log_prior[ci]

        return out
