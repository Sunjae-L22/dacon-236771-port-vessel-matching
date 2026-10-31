"""Learned Hangul -> English transliteration model (stochastic transducer).

Each Hangul jamo unit (onset / vowel / coda) emits an English letter chunk of
length 0..L.  P(chunk | unit) is learned with EM on (hangul name, catalog
english name) pairs from the official train.csv, smoothed toward a hand-made
phonetic prior.  Scoring P(english | hangul) is vectorised over many catalog
candidates at once.
"""
from __future__ import annotations

import math
import re
import unicodedata

import numpy as np

import vm

L = 3  # max chunk length

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ",
        "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

ONSET_CORE = {
    "ㄱ": "GKCQX", "ㄲ": "KGCQ", "ㄴ": "N", "ㄷ": "DT", "ㄸ": "DT", "ㄹ": "RL", "ㅁ": "M",
    "ㅂ": "BVP", "ㅃ": "BP", "ㅅ": "SCZTHXJ", "ㅆ": "SCZ", "ㅈ": "JZGDCS", "ㅉ": "JZTSC",
    "ㅊ": "CHTSZJ", "ㅋ": "KCQGXH", "ㅌ": "TDH", "ㅍ": "PFVH", "ㅎ": "H",
}
VOWEL_CORE = {
    "ㅏ": "AOUE", "ㅐ": "AEIY", "ㅑ": "YIAE", "ㅒ": "YAE", "ㅓ": "OUEAI", "ㅔ": "EAIY",
    "ㅕ": "YEOIU", "ㅖ": "YEI", "ㅗ": "OAUW", "ㅘ": "WUOA", "ㅙ": "WUOEA", "ㅚ": "OEUWI",
    "ㅛ": "YOIE", "ㅜ": "UOW", "ㅝ": "WOUAE", "ㅞ": "WEU", "ㅟ": "WUIE", "ㅠ": "YUIEOW",
    "ㅡ": "EUIO", "ㅢ": "UIEY", "ㅣ": "IEYA",
}
CODA_CORE = {
    "ㄱ": "KGCXQ", "ㄲ": "KGC", "ㄳ": "KGCSX", "ㄴ": "N", "ㄵ": "NJ", "ㄶ": "NH", "ㄷ": "DT",
    "ㄹ": "LR", "ㄺ": "LKG", "ㄻ": "LM", "ㄼ": "LBP", "ㄽ": "LS", "ㄾ": "LT", "ㄿ": "LP",
    "ㅀ": "LH", "ㅁ": "MN", "ㅂ": "PB", "ㅄ": "PBS", "ㅅ": "TSDCZ", "ㅆ": "TS", "ㅇ": "NG",
    "ㅈ": "TJ", "ㅊ": "TCH", "ㅋ": "KC", "ㅌ": "T", "ㅍ": "PF", "ㅎ": "H",
}
SILENT = set("HWYER")


def build_units():
    units = []
    for c in CHO:
        if c != "ㅇ":
            units.append("O" + c)
    for v in JUNG:
        units.append("V" + v)
    for j in JONG[1:]:
        units.append("C" + j)
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        units.append("L" + ch)
    units.append("Rㄹ")  # onset ㄹ right after coda ㄹ
    units.append("Eㅡ")  # word-final ㅡ (epenthetic)
    return units


UNITS = build_units()
UNIT_ID = {u: i for i, u in enumerate(UNITS)}


def core_letters(u: str) -> str:
    kind, j = u[0], u[1:]
    if u == "Rㄹ":
        return "LR"
    if u == "Eㅡ":
        return "EUIO"
    if kind == "O":
        return ONSET_CORE[j]
    if kind == "V":
        return VOWEL_CORE[j]
    if kind == "C":
        return CODA_CORE[j]
    return j


def empty_weight(u: str) -> float:
    kind, j = u[0], u[1:]
    if u in ("Vㅡ", "Eㅡ"):
        return 2.0
    if u == "Rㄹ":
        return 1.0
    if u == "Oㅎ":
        return 0.5
    if kind == "C" and j in ("ㅅ", "ㅆ", "ㅎ"):
        return 0.3
    if kind == "V":
        return 0.05
    if kind == "L":
        return 0.01
    return 0.03


LEN_W = [None, 1.0, 0.25, 0.04]


# ----------------------------------------------------------------------------
# text -> units / letters
# ----------------------------------------------------------------------------
def hangul_text_variants(name: str, max_alts: int = 6) -> list[str]:
    """Hangul name -> list of cleaned text alternatives (no digits)."""
    toks, _nums = vm.ko_split(name)
    per = []
    for k, t in enumerate(toks):
        opts = [t]
        if k == len(toks) - 1 and t.endswith("호") and len(t) >= 2 and len(toks) + len(t) > 3:
            opts.append(t[:-1])
        more = []
        for o in opts:
            if vm.HANGUL_RE.search(o):
                letters = vm.ko_as_letters(o)
                if letters:
                    more.append(letters)
        per.append(opts + more)
    alts = [""]
    for opts in per:
        alts = [a + " " + o if a else o for a in alts for o in opts][:max_alts]
    return alts or [""]


def text_units(text: str) -> list[int]:
    out = []
    prev_coda = ""
    for k, ch in enumerate(text):
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:
            c, v, j = CHO[code // 588], JUNG[(code % 588) // 28], JONG[code % 28]
            nxt = text[k + 1] if k + 1 < len(text) else " "
            word_end = not ("가" <= nxt <= "힣")
            if c != "ㅇ":
                out.append(UNIT_ID["Rㄹ"] if (c == "ㄹ" and prev_coda == "ㄹ") else UNIT_ID["O" + c])
            if v == "ㅡ" and not j and word_end:
                out.append(UNIT_ID["Eㅡ"])
            else:
                out.append(UNIT_ID["V" + v])
            if j:
                out.append(UNIT_ID["C" + j])
            prev_coda = j
        elif "A" <= ch <= "Z":
            out.append(UNIT_ID["L" + ch]); prev_coda = ""
        elif "a" <= ch <= "z":
            out.append(UNIT_ID["L" + ch.upper()]); prev_coda = ""
        else:
            prev_coda = ""
    return out


def en_letters(name) -> str:
    w, _ = vm.en_split(name)
    return re.sub(r"[^A-Z]", "", "".join(w))


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
class Transducer:
    def __init__(self, catalog_names: list[str], max_len: int = 40):
        self.max_len = max_len
        self.letters = [en_letters(n)[:max_len] for n in catalog_names]
        vocab = {"": 0}
        for s in self.letters:
            for l in range(1, L + 1):
                for j in range(l, len(s) + 1):
                    sub = s[j - l:j]
                    if sub not in vocab:
                        vocab[sub] = len(vocab)
        self.vocab = vocab
        self.inv_vocab = list(vocab)
        self.V = len(vocab) + 1  # last id = invalid (prob 0)
        self.INVALID = len(vocab)
        # chunk id table for every catalog name: sub[c, j, l] = id of s[j-l:j]
        N = len(self.letters)
        sub = np.full((N, max_len + 1, L + 1), self.INVALID, dtype=np.int32)
        self.lens = np.array([len(s) for s in self.letters], dtype=np.int32)
        for c, s in enumerate(self.letters):
            m = len(s)
            sub[c, : m + 1, 0] = 0
            for l in range(1, L + 1):
                for j in range(l, m + 1):
                    sub[c, j, l] = vocab[s[j - l:j]]
        self.sub = sub
        # word boundaries in the concatenated letter string (for local scoring)
        wstart = np.zeros((N, max_len + 1), dtype=bool)
        wend = np.zeros((N, max_len + 1), dtype=bool)
        for c, n in enumerate(catalog_names):
            words, _ = vm.en_split(n)
            pos = 0
            for w in words:
                w = re.sub(r"[^A-Z]", "", w)
                if not w:
                    continue
                if pos <= max_len:
                    wstart[c, pos] = True
                pos += len(w)
                if pos <= max_len:
                    wend[c, pos] = True
            wstart[c, 0] = True
            wend[c, min(self.lens[c], max_len)] = True
        self.wstart, self.wend = wstart, wend
        self.P0 = self._prior()
        self.P = self.P0.copy()

    def _prior(self) -> np.ndarray:
        P0 = np.zeros((len(UNITS), self.V), dtype=np.float64)
        chunks = self.inv_vocab
        for ui, u in enumerate(UNITS):
            core = set(core_letters(u))
            row = np.zeros(self.V)
            row[0] = empty_weight(u)
            for ci in range(1, len(chunks)):
                ch = chunks[ci]
                w = LEN_W[len(ch)]
                for a in ch:
                    if a in core:
                        continue
                    w *= 0.1 if a in SILENT else 0.003
                row[ci] = w
            P0[ui] = row / row.sum()
        return P0

    # -- vectorised forward over a set of catalog candidates -----------------
    def logprob(self, units: list[int], cand: np.ndarray) -> np.ndarray:
        """log P(english_c | units) for every candidate index c (float64)."""
        C = len(cand)
        if C == 0:
            return np.zeros(0)
        lens = self.lens[cand]
        if not units:
            return np.where(lens == 0, 0.0, -1e9)
        M = self.max_len + 1
        sub = self.sub[cand]  # (C, M, L+1)
        alpha = np.zeros((C, M))
        alpha[:, 0] = 1.0
        logscale = np.zeros(C)
        P = self.P
        for u in units:
            em = P[u][sub]  # (C, M, L+1)
            new = alpha * em[:, :, 0]
            for l in range(1, L + 1):
                new[:, l:] += alpha[:, :-l] * em[:, l:, l]
            mx = new.max(axis=1)
            mx = np.where(mx > 0, mx, 1.0)
            alpha = new / mx[:, None]
            logscale += np.log(mx)
        fin = alpha[np.arange(C), lens]
        with np.errstate(divide="ignore"):
            out = np.log(fin) + logscale
        return np.where(np.isfinite(out), out, -1e9)

    def logprob_local(self, units: list[int], cand: np.ndarray, word_pen: float = math.log(0.1),
                      letter_pen: float = math.log(0.1)) -> np.ndarray:
        """Like logprob, but whole english words may be skipped at the start/end
        (abbreviated / prefix-dropped names) at a cost per word and per letter."""
        C = len(cand)
        if C == 0:
            return np.zeros(0)
        lens = self.lens[cand]
        if not units:
            return np.full(C, -1e9)
        M = self.max_len + 1
        sub = self.sub[cand]
        js = np.arange(M)[None, :]
        ws = self.wstart[cand]
        we = self.wend[cand]
        alpha = np.where(ws, np.exp(np.where(js > 0, word_pen + letter_pen * js, 0.0)), 0.0)
        logscale = np.zeros(C)
        P = self.P
        for u in units:
            em = P[u][sub]
            new = alpha * em[:, :, 0]
            for l in range(1, L + 1):
                new[:, l:] += alpha[:, :-l] * em[:, l:, l]
            mx = new.max(axis=1)
            mx = np.where(mx > 0, mx, 1.0)
            alpha = new / mx[:, None]
            logscale += np.log(mx)
        rem = lens[:, None] - js
        endw = np.where(we & (rem >= 0), np.exp(np.where(rem > 0, word_pen + letter_pen * rem, 0.0)), 0.0)
        fin = (alpha * endw).sum(axis=1)
        with np.errstate(divide="ignore"):
            out = np.log(fin) + logscale
        return np.where(np.isfinite(out), out, -1e9)

    def logprob_both(self, units: list[int], cand: np.ndarray, word_pen: float = math.log(0.25),
                     letter_pen: float = math.log(0.6), mid_pen: float = math.log(0.3), split: int = 16):
        """(global, local) log-probs in one pass; candidates bucketed by length.

        local: whole english words may be skipped at the start/end (word_pen +
        letter_pen per letter); the match may also end mid-word (extra mid_pen),
        which covers truncated hangul spellings."""
        C = len(cand)
        g_out = np.full(C, -1e9)
        l_out = np.full(C, -1e9)
        if C == 0 or not units:
            return g_out, l_out
        lens_all = self.lens[cand]
        for mask in (lens_all <= split, lens_all > split):
            idx = np.flatnonzero(mask)
            if len(idx) == 0:
                continue
            cc = cand[idx]
            lens = lens_all[idx]
            M = int(lens.max()) + 1
            sub = self.sub[cc, :M]
            js = np.arange(M)[None, :]
            ws = self.wstart[cc, :M]
            we = self.wend[cc, :M]
            n = len(cc)
            ag = np.zeros((n, M)); ag[:, 0] = 1.0
            al = np.where(ws, np.exp(np.where(js > 0, word_pen + letter_pen * js, 0.0)), 0.0)
            sg = np.zeros(n); sl = np.zeros(n)
            P = self.P
            for u in units:
                em = P[u][sub]
                e0 = em[:, :, 0]
                ng = ag * e0
                nl = al * e0
                for l in range(1, L + 1):
                    el = em[:, l:, l]
                    ng[:, l:] += ag[:, :-l] * el
                    nl[:, l:] += al[:, :-l] * el
                mg = ng.max(axis=1); mg = np.where(mg > 0, mg, 1.0)
                ml = nl.max(axis=1); ml = np.where(ml > 0, ml, 1.0)
                ag = ng / mg[:, None]; al = nl / ml[:, None]
                sg += np.log(mg); sl += np.log(ml)
            fin = ag[np.arange(n), lens]
            rem = lens[:, None] - js
            pen = np.where(rem > 0, word_pen + letter_pen * rem, 0.0) + np.where(we, 0.0, mid_pen)
            endw = np.where((rem >= 0) & (js > 0), np.exp(pen), 0.0)
            finl = (al * endw).sum(axis=1)
            with np.errstate(divide="ignore"):
                og = np.log(fin) + sg
                ol = np.log(finl) + sl
            g_out[idx] = np.where(np.isfinite(og), og, -1e9)
            l_out[idx] = np.where(np.isfinite(ol), ol, -1e9)
        return g_out, l_out

    # -- EM training on (units, catalog index) pairs --------------------------
    def _fb(self, units, c):
        """Unscaled forward-backward for one pair (names are short; float64 is enough)."""
        m = int(self.lens[c])
        sub = self.sub[c, : m + 1]  # (m+1, L+1)
        n = len(units)
        P = self.P
        em = [P[u][sub] for u in units]  # each (m+1, L+1): chunk ending at j with length l
        alpha = np.zeros((n + 1, m + 1))
        alpha[0, 0] = 1.0
        for i in range(1, n + 1):
            e = em[i - 1]
            a = alpha[i - 1] * e[:, 0]
            for l in range(1, L + 1):
                a[l:] += alpha[i - 1, :-l] * e[l:, l]
            alpha[i] = a
        Z = alpha[n, m]
        if not Z > 1e-300:
            return None, -1e9
        beta = np.zeros((n + 1, m + 1))
        beta[n, m] = 1.0
        for i in range(n, 0, -1):
            e = em[i - 1]
            b = beta[i] * e[:, 0]
            for l in range(1, L + 1):
                b[:-l] += beta[i, l:] * e[l:, l]
            beta[i - 1] = b
        counts = []
        for i in range(1, n + 1):
            e = em[i - 1]
            for l in range(0, L + 1):
                if l == 0:
                    w = alpha[i - 1] * e[:, 0] * beta[i]
                    ids = sub[:, 0]
                else:
                    w = alpha[i - 1, :-l] * e[l:, l] * beta[i, l:]
                    ids = sub[l:, l]
                w = w / Z
                nz = w > 1e-9
                if nz.any():
                    counts.append((units[i - 1], ids[nz], w[nz]))
        return counts, math.log(Z)

    def fit(self, pairs, iters: int = 8, tau: float = 3.0, verbose: bool = False):
        """pairs: list of (list_of_unit_alternatives, catalog_index)."""
        for it in range(iters):
            C = np.zeros_like(self.P)
            tot = 0.0
            for alts, c in pairs:
                best = None
                for units in alts:
                    if not units:
                        continue
                    lp = self.logprob(units, np.array([c]))[0]
                    if best is None or lp > best[0]:
                        best = (lp, units)
                if best is None or best[0] < -1e8:
                    continue
                counts, logZ = self._fb(best[1], c)
                if counts is None:
                    continue
                tot += logZ
                for u, ids, w in counts:
                    np.add.at(C[u], ids, w)
            C[:, self.INVALID] = 0
            self.P = (C + tau * self.P0) / (C.sum(axis=1, keepdims=True) + tau)
            if verbose:
                print(f"EM iter {it} loglik {tot:.1f}", flush=True)
        return self

    def top_chunks(self, unit: str, k: int = 8):
        row = self.P[UNIT_ID[unit]]
        idx = np.argsort(-row)[:k]
        return [(self.inv_vocab[i] if i < len(self.inv_vocab) else "<inv>", round(float(row[i]), 3)) for i in idx]


class LetterLM:
    """Interpolated character trigram LM over catalog english letter strings."""

    def __init__(self, strings: list[str], order: int = 3):
        from collections import Counter
        self.order = order
        self.counts = [Counter() for _ in range(order + 1)]
        self.ctx = [Counter() for _ in range(order + 1)]
        for s in strings:
            t = "^" * (order - 1) + s + "$"
            for i in range(order - 1, len(t)):
                for n in range(1, order + 1):
                    g = t[i - n + 1:i + 1]
                    self.counts[n][g] += 1
                    self.ctx[n][g[:-1]] += 1
        self.V = 28
        self.lam = [0.0, 0.1, 0.3, 0.6][: order + 1]
        self.cache = {}

    def logprob(self, s: str) -> float:
        if s in self.cache:
            return self.cache[s]
        order = self.order
        t = "^" * (order - 1) + s + "$"
        lp = 0.0
        for i in range(order - 1, len(t)):
            p = 1.0 / self.V * 0.01
            for n in range(1, order + 1):
                g = t[i - n + 1:i + 1]
                c = self.ctx[n].get(g[:-1], 0)
                if c:
                    p += self.lam[n] * self.counts[n].get(g, 0) / c
            lp += math.log(p)
        self.cache[s] = lp
        return lp
