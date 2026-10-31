"""Vessel matching core: normalization, Korean->Latin phonetic keys, retrieval.

Every query row is processed on its own. Only the official catalog (and,
for training-time statistics, the official train.csv) is ever indexed.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import numpy as np

# ----------------------------------------------------------------------------
# generic helpers
# ----------------------------------------------------------------------------
HANGUL_RE = re.compile("[가-힣]")


def is_hangul(s) -> bool:
    return isinstance(s, str) and bool(HANGUL_RE.search(s))


def pos_float(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) and x > 0 else None


ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
         "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12}
PREFIX_DROP = {"MV", "M/V", "MT", "M/T", "SS"}


def en_tokens(name) -> list[str]:
    """Upper-case alnum tokens of an English-ish name."""
    if not isinstance(name, str):
        return []
    s = unicodedata.normalize("NFKC", name).upper()
    s = s.replace("M/V ", "MV ").replace("M/T ", "MT ").replace("M.V.", "MV ")
    s = re.sub(r"[^0-9A-Z가-힣]+", " ", s)
    s = re.sub(r"(?<=[A-Z])(?=[0-9])|(?<=[0-9])(?=[A-Z])", " ", s)
    return s.split()


def en_split(name) -> tuple[list[str], list[str]]:
    """Return (word tokens, number tokens). Roman numerals (not first) -> numbers."""
    toks = en_tokens(name)
    if toks and toks[0] in PREFIX_DROP and len(toks) > 1:
        toks = toks[1:]
    words, nums = [], []
    for i, t in enumerate(toks):
        if t.isdigit():
            nums.append(str(int(t)))
        elif i > 0 and t in ROMAN:
            nums.append(str(ROMAN[t]))
        elif t in ("NO", "NR") and i + 1 < len(toks) and toks[i + 1].isdigit():
            continue
        else:
            words.append(t)
    return words, nums


def en_compact(name) -> str:
    w, n = en_split(name)
    return "".join(w) + "".join(n)


def raw_compact(name) -> str:
    return "".join(en_tokens(name))


# ----------------------------------------------------------------------------
# English -> consonant skeleton (Korean-loanword oriented classes)
#   K: g k c(hard) q   T: d t th   P: b p f v ph   S: s z j c(soft) ch sh x(s)
#   L: l r(prevocalic) M: m  N: n  G: ng / n before k,g   H: h
# ----------------------------------------------------------------------------
VOW = set("AEIOUY")


def en_skeleton_word(w: str, keep_vowels: bool = False) -> str:
    w = re.sub(r"[^A-Z]", "", w)
    # collapse doubled letters first (GG, LL, PP ...)
    w = re.sub(r"(.)\1+", r"\1", w)
    out = []
    i, n = 0, len(w)
    while i < n:
        c = w[i]
        nx = w[i + 1] if i + 1 < n else ""
        nx2 = w[i + 2] if i + 2 < n else ""
        if c in "AEIOU":
            if keep_vowels:
                out.append(c)
            i += 1
            continue
        if c == "Y":
            if keep_vowels and (i == n - 1 or nx not in VOW):
                out.append("I")
            i += 1
            continue
        if c == "W":
            i += 1
            continue
        two = w[i:i + 2]
        if two == "PH":
            out.append("P"); i += 2; continue
        if two == "CK":
            out.append("K"); i += 2; continue
        if two == "CH":
            # chem-, chris-, chr- are read with K in Korean loanwords
            if w[i + 2:i + 4] in ("EM", "RI", "RO", "RY", "RA") or nx2 in ("R", "L"):
                out.append("K")
            else:
                out.append("S")
            i += 2; continue
        if two in ("SH", "ZH"):
            out.append("S"); i += 2; continue
        if two == "TH":
            out.append("T"); i += 2; continue
        if two == "GH":
            i += 2; continue
        if two == "DG":
            out.append("S"); i += 2; continue
        if two == "QU":
            out.append("K"); i += 2; continue
        if w[i:i + 4] == "TION":
            out.append("S"); out.append("N"); i += 4; continue
        if two == "NG":
            out.append("N")
            if nx2 in ("E", "I"):
                out.append("S")
            elif nx2 and nx2 in "AOU":
                out.append("K")
            i += 2; continue
        if c == "C":
            out.append("S" if nx in ("E", "I", "Y") else "K"); i += 1; continue
        if c == "G":
            out.append("S" if nx in ("E", "I", "Y") and i > 0 or (i == 0 and nx in ("E", "I") and nx2 not in ("T", "R")) else "K")
            i += 1; continue
        if c == "X":
            if i == 0:
                out.append("S")
            else:
                out.append("K"); out.append("S")
            i += 1; continue
        if c == "H":
            if nx and nx in VOW and (i == 0 or w[i - 1] not in "BDKPRT"):
                out.append("H")
            i += 1
            continue
        if c == "R":
            if nx and nx in VOW:
                out.append("L")
            i += 1
            continue
        m = {"B": "P", "P": "P", "F": "P", "V": "P", "D": "T", "T": "T",
             "K": "K", "Q": "K", "S": "S", "Z": "S", "J": "S", "L": "L",
             "M": "M", "N": "N", "H": "H"}.get(c)
        if m:
            out.append(m)
        i += 1
    # collapse runs of identical consonants
    res = []
    for ch in out:
        if not res or res[-1] != ch or ch in "AEIOU":
            res.append(ch)
    return "".join(res)


# Letter-name reading of an acronym token (KMTC -> 케이엠티씨)
LETTER_SKEL = {"A": "", "B": "P", "C": "S", "D": "T", "E": "", "F": "P", "G": "S",
               "H": "S", "I": "", "J": "S", "K": "K", "L": "L", "M": "M", "N": "N",
               "O": "", "P": "P", "Q": "K", "R": "L", "S": "S", "T": "T", "U": "",
               "V": "P", "W": "TPL", "X": "KS", "Y": "", "Z": "S"}


def acronym_skeleton(w: str) -> str:
    s = "".join(LETTER_SKEL.get(c, "") for c in w)
    res = []
    for ch in s:
        if not res or res[-1] != ch:
            res.append(ch)
    return "".join(res)


def looks_acronym(w: str) -> bool:
    if len(w) > 5 or len(w) < 2:
        return False
    vowels = sum(c in "AEIOU" for c in w)
    return vowels == 0 or (len(w) <= 4 and vowels <= 1) or len(w) == 2


def en_skeletons(name, max_alts: int = 4) -> list[str]:
    """Skeleton(s) of an English name; acronym-like tokens get both readings."""
    words, _ = en_split(name)
    alts = [""]
    for w in words:
        base = en_skeleton_word(w)
        opts = [base]
        if looks_acronym(w):
            a = acronym_skeleton(w)
            if a != base:
                opts.append(a)
        alts = [x + o for x in alts for o in opts][:max_alts]
    return [collapse(a) for a in alts]


def collapse(s: str) -> str:
    res = []
    for ch in s:
        if not res or res[-1] != ch:
            res.append(ch)
    return "".join(res)


# ----------------------------------------------------------------------------
# Korean -> consonant skeleton
# ----------------------------------------------------------------------------
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ",
        "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
CHO_SK = {"ㄱ": "K", "ㄲ": "K", "ㄴ": "N", "ㄷ": "T", "ㄸ": "T", "ㄹ": "L", "ㅁ": "M",
          "ㅂ": "P", "ㅃ": "P", "ㅅ": "S", "ㅆ": "S", "ㅇ": "", "ㅈ": "S", "ㅉ": "S",
          "ㅊ": "S", "ㅋ": "K", "ㅌ": "T", "ㅍ": "P", "ㅎ": "H"}
JONG_SK = {"": "", "ㄱ": "K", "ㄲ": "K", "ㄳ": "K", "ㄴ": "N", "ㄵ": "N", "ㄶ": "N",
           "ㄷ": "T", "ㄹ": "L", "ㄺ": "L", "ㄻ": "M", "ㄼ": "L", "ㄽ": "L", "ㄾ": "L",
           "ㄿ": "P", "ㅀ": "L", "ㅁ": "M", "ㅂ": "P", "ㅄ": "P", "ㅅ": "T", "ㅆ": "T",
           "ㅇ": "N", "ㅈ": "T", "ㅊ": "T", "ㅋ": "K", "ㅌ": "T", "ㅍ": "P", "ㅎ": ""}
JUNG_V = {"ㅏ": "A", "ㅐ": "E", "ㅑ": "A", "ㅒ": "E", "ㅓ": "O", "ㅔ": "E", "ㅕ": "O",
          "ㅖ": "E", "ㅗ": "O", "ㅘ": "A", "ㅙ": "E", "ㅚ": "E", "ㅛ": "O", "ㅜ": "U",
          "ㅝ": "O", "ㅞ": "E", "ㅟ": "I", "ㅠ": "U", "ㅡ": "", "ㅢ": "I", "ㅣ": "I"}


def decompose(ch: str):
    code = ord(ch) - 0xAC00
    if not (0 <= code < 11172):
        return None
    return CHO[code // 588], JUNG[(code % 588) // 28], JONG[code % 28]


def ko_skeleton(word: str, keep_vowels: bool = False) -> str:
    out = []
    syl = [decompose(ch) for ch in word]
    for k, ch in enumerate(word):
        d = syl[k]
        if d is None:
            if ch.isalpha():
                out.append(en_skeleton_word(ch.upper(), keep_vowels))
            continue
        c, v, j = d
        nxt = syl[k + 1] if k + 1 < len(word) else None
        # 르 / 를 without coda: epenthetic for a non-prevocalic R -> silent
        if c == "ㄹ" and v == "ㅡ" and j == "" and k > 0:
            continue
        # ㅎ between vowels is often silent; keep only word-initially
        if c == "ㅎ" and k > 0 and syl[k - 1] is not None and syl[k - 1][2] == "":
            pass
        out.append(CHO_SK[c])
        if keep_vowels:
            out.append(JUNG_V[v])
        out.append(JONG_SK[j])
    s = "".join(out)
    if keep_vowels:
        return s
    return collapse(s)


# Korean spellings of Latin letter names
LETTER_KO = {
    "A": ["에이"], "B": ["비"], "C": ["씨", "시"], "D": ["디"], "E": ["이"],
    "F": ["에프"], "G": ["지", "쥐"], "H": ["에이치", "에치"], "I": ["아이"],
    "J": ["제이"], "K": ["케이"], "L": ["엘"], "M": ["엠"], "N": ["엔"], "O": ["오"],
    "P": ["피"], "Q": ["큐"], "R": ["알"], "S": ["에스"], "T": ["티"], "U": ["유"],
    "V": ["브이", "뷔"], "W": ["더블유", "더블류"], "X": ["엑스"], "Y": ["와이"],
    "Z": ["제트", "지"]}
KO_LETTER = sorted(((ko, l) for l, kos in LETTER_KO.items() for ko in kos),
                   key=lambda x: -len(x[0]))


@lru_cache(maxsize=100000)
def ko_as_letters(tok: str):
    """Segment a Hangul token into Latin letter names; return 'KMTC' or None."""
    n = len(tok)
    best = [None] * (n + 1)
    best[0] = ""
    for i in range(n):
        if best[i] is None:
            continue
        for ko, l in KO_LETTER:
            if tok.startswith(ko, i):
                j = i + len(ko)
                cand = best[i] + l
                if best[j] is None or len(cand) < len(best[j]):
                    best[j] = cand
    r = best[n]
    if r is None or len(r) < 2:
        return None
    return r


KO_NUM_RE = re.compile(r"제?\s*(\d+)\s*호?")


def ko_split(name: str):
    """Hangul name -> (list of word tokens, numbers). Strips 제N / N호 / 호 suffix."""
    s = unicodedata.normalize("NFKC", name).upper()
    nums = re.findall(r"\d+", s)
    nums = [str(int(x)) for x in nums]
    s = re.sub(r"제\s*\d+\s*호?", " ", s)
    s = re.sub(r"\d+\s*호", " ", s)
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"[^0-9A-Z가-힣]+", " ", s)
    toks = s.split()
    for i, t in enumerate(toks):
        if t.startswith("제") and len(t) > 1 and False:
            toks[i] = t[1:]
    return toks, nums


def ko_variants(name: str, dictionary: dict | None = None, max_alts: int = 8):
    """Generate candidate consonant skeletons for a Hangul name.

    Returns list of (skeleton, vowel_key) alternatives.
    """
    toks, nums = ko_split(name)
    # whole-name trailing 호 strip as an option
    per_tok = []
    for k, t in enumerate(toks):
        opts = []
        base = t
        cands = [base]
        if k == len(toks) - 1 and base.endswith("호") and len(base) >= 2 and len(toks) + len(base) > 3:
            cands.append(base[:-1])
        for c in cands:
            opts.append((ko_skeleton(c), ko_skeleton(c, True)))
            letters = ko_as_letters(c) if HANGUL_RE.search(c) else None
            if letters:
                opts.append((acronym_skeleton(letters), letters))
            if dictionary and c in dictionary:
                for en in dictionary[c][:2]:
                    opts.append((en_skeleton_word(en), en_skeleton_word(en, True)))
        # dedupe
        seen, uniq = set(), []
        for o in opts:
            if o[0] not in seen:
                seen.add(o[0]); uniq.append(o)
        per_tok.append(uniq)
    alts = [("", "")]
    for opts in per_tok:
        alts = [(a[0] + o[0], a[1] + o[1]) for a in alts for o in opts][:max_alts]
    return [(collapse(a[0]), a[1]) for a in alts], nums


def ko_romanize_simple(name: str) -> str:
    """Rough vowel-preserving romanisation (for char-ngram features)."""
    return ko_skeleton(name, keep_vowels=True)
