"""Vessel matching: candidate retrieval + pairwise ranking.

Every query row is processed on its own; lookups only use the public catalog
and the official train data.  Hangul names are compared with English catalog
names through a phonetic consonant skeleton and a romanization.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import bisect
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

try:  # fast path when available; identical results to the fallback
    from rapidfuzz.distance import Levenshtein as _RFLev

    def lev(a: str, b: str) -> int:
        return _RFLev.distance(a, b)

except Exception:  # pragma: no cover - bit-parallel (Myers/Hyyro) fallback

    def lev(a, b):
        if a == b:
            return 0
        if len(a) < len(b):
            a, b = b, a
        m = len(b)
        if m == 0:
            return len(a)
        peq = {}
        for i, ch in enumerate(b):
            peq[ch] = peq.get(ch, 0) | (1 << i)
        mask = (1 << m) - 1
        last = 1 << (m - 1)
        pv, mv, score = mask, 0, m
        for ch in a:
            eq = peq.get(ch, 0)
            xv = eq | mv
            xh = (((eq & pv) + pv) ^ pv) | eq
            ph = mv | (~(xh | pv) & mask)
            mh = pv & xh
            if ph & last:
                score += 1
            elif mh & last:
                score -= 1
            ph = ((ph << 1) | 1) & mask
            mh = (mh << 1) & mask
            pv = mh | (~(xv | ph) & mask)
            mv = ph & xv
        return score


def lev_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return 1.0 - lev(a, b) / max(len(a), len(b))


# ------------------------------------------------- typo-aware edit distance
# The Latin port-record names were corrupted by keyboard-style noise: almost every
# corrupted name differs from the catalog name by one or two of {dropped letter,
# doubled letter, adjacent transposition}.  Plain Levenshtein charges a transposition
# twice; OSA (optimal string alignment, restricted Damerau-Levenshtein) charges it once,
# so it ranks the true name higher when the candidate list is cut by edit distance.
def _osa_py(a: str, b: str) -> int:
    """Bit-parallel OSA distance (Hyyro 2003); identical to rapidfuzz.distance.OSA."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    m = len(b)
    if m == 0:
        return len(a)
    peq = {}
    for i, ch in enumerate(b):
        peq[ch] = peq.get(ch, 0) | (1 << i)
    full = (1 << m) - 1
    last = 1 << (m - 1)
    vp, vn, d0, pm_old, score = full, 0, 0, 0, m
    for ch in a:
        pm = peq.get(ch, 0)
        tr = (((~d0 & full) & pm) << 1) & pm_old
        d0 = ((((pm & vp) + vp) & full) ^ vp) | pm | vn | tr
        hp = vn | (~(d0 | vp) & full)
        hn = d0 & vp
        if hp & last:
            score += 1
        elif hn & last:
            score -= 1
        hp = ((hp << 1) | 1) & full
        hn = (hn << 1) & full
        vp = hn | (~(d0 | hp) & full)
        vn = hp & d0
        pm_old = pm
    return score


try:  # fast path when available; identical results to the fallback
    from rapidfuzz.distance import OSA as _RFOSA

    def osa(a: str, b: str) -> int:
        return _RFOSA.distance(a, b)

except Exception:  # pragma: no cover
    osa = _osa_py


def osa_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return 1.0 - osa(a, b) / max(len(a), len(b))


# ---------------------------------------------------------------- text keys
ROMAN = {"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "Ⅳ": "IV", "Ⅴ": "V", "Ⅵ": "VI", "Ⅶ": "VII", "Ⅷ": "VIII", "Ⅸ": "IX", "Ⅹ": "X"}
PREFIX_RE = re.compile(r"^(M/V|M/T|MV|MT|M\.V\.|M\.T\.)\s+")
NO_RE = re.compile(r"\bNO\.?\s*(?=\d)")
JE_RE = re.compile(r"제\s*(?=\d)")


HANGUL_JAMO_FIX = True   # repair standalone jamo (Hangul keyboard slips) instead of turning them into spaces
_FX_L = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_FX_V = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_FX_T = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
         "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
_FX_CONJ = {}
for _i, _c in enumerate(_FX_L):
    _FX_CONJ[chr(0x1100 + _i)] = _c
for _i, _c in enumerate(_FX_V):
    _FX_CONJ[chr(0x1161 + _i)] = _c
for _i, _c in enumerate(_FX_T[1:]):
    _FX_CONJ[chr(0x11A8 + _i)] = _c


def _fx_parts(ch):
    c = ord(ch) - 0xAC00
    return c // 588, (c % 588) // 28, c % 28


_FX_KL = dict(zip(_FX_L, "rRseEfaqQtTdwWczxvg"))
_FX_KV = {"ㅏ": "k", "ㅐ": "o", "ㅑ": "i", "ㅒ": "O", "ㅓ": "j", "ㅔ": "p", "ㅕ": "u", "ㅖ": "P", "ㅗ": "h", "ㅘ": "hk",
          "ㅙ": "ho", "ㅚ": "hl", "ㅛ": "y", "ㅜ": "n", "ㅝ": "nj", "ㅞ": "np", "ㅟ": "nl", "ㅠ": "b", "ㅡ": "m",
          "ㅢ": "ml", "ㅣ": "l"}
_FX_TC = {"ㄳ": "ㄱㅅ", "ㄵ": "ㄴㅈ", "ㄶ": "ㄴㅎ", "ㄺ": "ㄹㄱ", "ㄻ": "ㄹㅁ", "ㄼ": "ㄹㅂ", "ㄽ": "ㄹㅅ", "ㄾ": "ㄹㅌ",
          "ㄿ": "ㄹㅍ", "ㅀ": "ㄹㅎ", "ㅄ": "ㅂㅅ"}
_FX_K2J = {v: k for k, v in _FX_KL.items()}
_FX_K2J.update({v: k for k, v in _FX_KV.items() if len(v) == 1})
_FX_VC = {("ㅗ", "ㅏ"): "ㅘ", ("ㅗ", "ㅐ"): "ㅙ", ("ㅗ", "ㅣ"): "ㅚ", ("ㅜ", "ㅓ"): "ㅝ", ("ㅜ", "ㅔ"): "ㅞ",
          ("ㅜ", "ㅣ"): "ㅟ", ("ㅡ", "ㅣ"): "ㅢ"}
_FX_TCK = {tuple(v): k for k, v in _FX_TC.items()}


def _retype(text: str) -> str:
    """Hangul (syllables and standalone jamo) re-typed through a dubeolsik input automaton, so that jamo that can
    form syllables are composed (a fully decomposed name becomes syllables again; well-formed syllables are kept)."""
    keys = []
    for ch in text:
        if "가" <= ch <= "힣":
            l, v, t = _fx_parts(ch)
            keys.append(_FX_KL[_FX_L[l]] + _FX_KV[_FX_V[v]])
            if t:
                keys.append("".join(_FX_KL[x] for x in _FX_TC.get(_FX_T[t], _FX_T[t])))
        elif ch in _FX_KL:
            keys.append(_FX_KL[ch])
        elif ch in _FX_KV:
            keys.append(_FX_KV[ch])
        elif ch in _FX_TC:
            keys.append("".join(_FX_KL[x] for x in _FX_TC[ch]))
        else:
            keys.append("\0" + ch)
    seq = []
    for k in keys:
        if k.startswith("\0"):
            seq.append(k[1:])
        else:
            seq.extend(_FX_K2J[c] for c in k)
    out, i, n = [], 0, len(seq)
    while i < n:
        j = seq[i]
        if j in _FX_L and i + 1 < n and seq[i + 1] in _FX_V:
            L, V = j, seq[i + 1]
            i += 2
            if i < n and (V, seq[i]) in _FX_VC:
                V = _FX_VC[(V, seq[i])]
                i += 1
            T = ""
            if i < n and seq[i] in _FX_L and seq[i] in _FX_T and not (i + 1 < n and seq[i + 1] in _FX_V):
                T = seq[i]
                i += 1
                if i < n and seq[i] in _FX_L and (T, seq[i]) in _FX_TCK and not (i + 1 < n and seq[i + 1] in _FX_V):
                    T = _FX_TCK[(T, seq[i])]
                    i += 1
            out.append(chr(0xAC00 + _FX_L.index(L) * 588 + _FX_V.index(V) * 28 + _FX_T.index(T)))
        else:
            out.append(j)
            i += 1
    return "".join(out)


def repair_jamo(text: str) -> str:
    """Standalone jamo left by a keyboard slip in a Hangul name are folded back into syllables so that they do
    not split the word: the name is re-typed (jamo that can compose are composed), then a repeated jamo is
    dropped, a consonant after an open syllable becomes its final, any other consonant gets the neutral vowel (the
    loanword epenthetic vowel) and a vowel gets the silent initial."""
    text = unicodedata.normalize("NFC", text)
    chars = [_FX_CONJ.get(ch, ch) for ch in text]
    if not any(ch in _FX_L or ch in _FX_V or ch in _FX_TC for ch in chars):
        return text
    chars = list(_retype("".join(chars)))
    out = []
    for i, ch in enumerate(chars):
        is_c, is_v = ch in _FX_L, ch in _FX_V
        if not (is_c or is_v):
            out.append(ch)
            continue
        prev = out[-1] if out and "가" <= out[-1] <= "힣" else None
        nxt = chars[i + 1] if i + 1 < len(chars) else ""
        if is_c:
            if prev is not None and _FX_T[_fx_parts(prev)[2]] == ch:
                continue
            if "가" <= nxt <= "힣" and _FX_L[_fx_parts(nxt)[0]] == ch:
                continue
            if prev is not None and _fx_parts(prev)[2] == 0 and ch in _FX_T:
                out[-1] = chr(ord(prev) + _FX_T.index(ch))
                continue
            out.append(chr(0xAC00 + _FX_L.index(ch) * 588 + _FX_V.index("ㅡ") * 28))
        else:
            if prev is not None and _FX_V[_fx_parts(prev)[1]] == ch:
                continue
            out.append(chr(0xAC00 + _FX_L.index("ㅇ") * 588 + _FX_V.index(ch) * 28))
    return "".join(out)


def clean(value: object) -> str:
    if not isinstance(value, str):
        return ""
    if HANGUL_JAMO_FIX and any("\u3131" <= ch <= "\u318e" or "\u1100" <= ch <= "\u11ff" for ch in value):
        value = repair_jamo(value)
    for k, v in ROMAN.items():
        value = value.replace(k, v)
    value = unicodedata.normalize("NFKC", value).upper().strip()
    value = PREFIX_RE.sub("", value)
    value = NO_RE.sub("", value)
    value = JE_RE.sub("", value)
    value = re.sub(r"[^0-9A-Z가-힣 ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    if is_hangul(value) and value.endswith("호") and len(value.replace(" ", "")) > 2:
        value = value[:-1].strip()
    value = re.sub(r"(?<=[A-Z가-힣])(?=\d)|(?<=\d)(?=[A-Z가-힣])", " ", value)
    toks = value.split()
    return " ".join([t for t in toks if not t.isdigit()] + [t for t in toks if t.isdigit()])


def is_hangul(value: object) -> bool:
    return isinstance(value, str) and any("가" <= ch <= "힣" for ch in value)


def key(value: str) -> str:
    return value.replace(" ", "")


L_ROM = ["G", "KK", "N", "D", "TT", "R", "M", "B", "PP", "S", "SS", "", "J", "JJ", "CH", "K", "T", "P", "H"]
V_ROM = ["A", "AE", "YA", "YAE", "EO", "E", "YEO", "YE", "O", "WA", "WAE", "OE", "YO", "U", "WO", "WE", "WI", "YU", "EU", "UI", "I"]
T_ROM = ["", "K", "K", "KS", "N", "NJ", "NH", "T", "L", "LK", "LM", "LB", "LS", "LT", "LP", "LH", "M", "P", "PS", "T", "T", "NG", "T", "T", "K", "T", "P", "T"]
# consonant classes for the phonetic skeleton
L_SK = ["K", "K", "N", "T", "T", "L", "M", "P", "P", "S", "S", "", "S", "S", "S", "K", "T", "P", ""]
T_SK = ["", "K", "K", "K", "N", "N", "N", "T", "L", "L", "L", "L", "L", "L", "L", "L", "M", "P", "P", "T", "T", "N", "T", "T", "K", "T", "P", "T"]
# vowel classes (coarse) so that skeleton+vowel strings can also be compared
V_CL = ["A", "A", "A", "A", "O", "E", "O", "E", "O", "A", "A", "E", "O", "U", "O", "E", "I", "U", "", "I", "I"]


def romanize(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:
            out.append(L_ROM[code // 588] + V_ROM[(code % 588) // 28] + T_ROM[code % 28])
        else:
            out.append(ch)
    return "".join(out)


def _hangul_skeleton(text: str, vowels: bool) -> str:
    out = []
    for ch in text:
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:
            out.append(L_SK[code // 588])
            if vowels:
                out.append(V_CL[(code % 588) // 28].lower())
            out.append(T_SK[code % 28])
        else:
            out.append(ch)
    return "".join(out)


EN_SUBS = [
    ("SCH", "S"), ("PH", "F"), ("GH", ""), ("CK", "K"), ("SH", "S"), ("CH", "S"),
    ("TH", "S"), ("TS", "S"), ("TZ", "S"), ("X", "KS"), ("Q", "K"), ("NG", "N"),
]
EN_CLASS = {"B": "P", "P": "P", "F": "P", "V": "P", "G": "K", "K": "K", "C": "K", "D": "T", "T": "T",
            "L": "L", "R": "L", "S": "S", "Z": "S", "J": "S", "M": "M", "N": "N"}
EN_VOWEL = {"A": "a", "E": "e", "I": "i", "O": "o", "U": "u"}


# Alternate English codes (in the spirit of Double Metaphone's primary/alternate codes),
# tuned to how Korean transcribes foreign names.  Each rule is right for one family of
# words and wrong for another, so they never replace the primary code; they only offer a
# second reading that a Hangul query may match.  Applied in order, before EN_SUBS.
ALT_SUBS = [
    (re.compile(r"^X(?=[IU])"), "S"),             # pinyin x- (Xiu 슈, Xin 신)
    (re.compile(r"CHT"), "S"),                    # Dutch -gracht (그라츠)
    (re.compile(r"TION"), "SION"),                # -tion = 션 (Resolution 리솔루션)
    (re.compile(r"DG(?=[EI])"), "J"),             # Bridge 브리지
    (re.compile(r"CH(?=[RL]|EM|OR|AO)"), "K"),    # Greek ch (Chorus 코러스, Chemist 케미스트)
    (re.compile(r"TH(?=[AEIOUY])"), "T"),         # th as t (Thailand 타일랜드, Themis 테미스)
    (re.compile(r"(?<=[A-Z])TU(?=RE|NE)"), "CHU"),  # -ture/-tune (Fortune 포춘, Venture 벤처)
    (re.compile(r"G+(?=[EI])"), "J"),             # soft g (Angela 안젤라, Courage 커리지)
    (re.compile(r"NG(?=[AOULR])"), "NK"),         # g sounded after n (Tonga 통가, Sunglory 썬글로리)
    (re.compile(r"(?<=[AEIOUY])R(?=E$)"), ""),    # silent r in final vowel+re (Sapphire 사파이어)
]
# skeleton modes: 0 primary, 1 alternate, 2 alternate with every R kept as L
# (Korean 르 in Romance/Greek names: Nestor 네스토르, Roberto 로베르토, Jakarta 자카르타)
SKEL_MODES = (0, 1, 2)


def _english_skeleton(word: str, vowels: bool, mode: int = 0) -> str:
    w = word
    if mode:
        for pat, rep in (ALT_SUBS[:-1] if mode == 2 else ALT_SUBS):
            w = pat.sub(rep, w)
    w = re.sub(r"C(?=[EIY])", "S", w)
    for a, b in EN_SUBS:
        w = w.replace(a, b)
    if mode != 2:
        w = re.sub(r"R(?![AEIOUY])", "", w)  # non-rhotic R (Pioneer -> 파이오니어)
    out = []
    for ch in w:
        if ch.isdigit():
            out.append(ch)
        elif ch in EN_CLASS:
            out.append(EN_CLASS[ch])
        elif vowels and ch in EN_VOWEL:
            out.append(EN_VOWEL[ch])
    return "".join(out)


def _finish(s: str, vowels: bool) -> str:
    s = re.sub(r"(.)\1+", r"\1", s)
    if vowels:
        s = re.sub(r"([aeiou])[aeiou]+", r"\1", s)
    return s


def _run_codes(run: str, vowels: bool) -> tuple:
    """Codes of one Hangul / English run, one per skeleton mode (duplicates kept)."""
    if is_hangul(run):
        return (_hangul_skeleton(run, vowels),) * len(SKEL_MODES)
    return tuple(_english_skeleton(run, vowels, m) for m in SKEL_MODES)


def skeleton(text: str, vowels: bool = False, mode: int = 0) -> str:
    parts = []
    for run in re.findall(r"[가-힣]+|[0-9A-Z]+", text):
        if is_hangul(run):
            parts.append(_hangul_skeleton(run, vowels))
        else:
            parts.append(_english_skeleton(run, vowels, mode))
    return _finish("".join(parts), vowels)


def skeleton_options(text: str, vowels: bool = False, cache: dict | None = None) -> tuple:
    """Per run, its codes under every skeleton mode (for per-word best readings)."""
    out = []
    for run in re.findall(r"[가-힣]+|[0-9A-Z]+", text):
        if cache is None:
            out.append(_run_codes(run, vowels))
            continue
        codes = cache.get(run)
        if codes is None:
            codes = cache[run] = _run_codes(run, vowels)
        out.append(codes)
    return tuple(out)


def mode_code(options: tuple, mode: int, vowels: bool) -> str:
    """Whole-name code under one skeleton mode (equals skeleton(text, vowels, mode))."""
    return _finish("".join(o[mode] for o in options), vowels)


def mixed_codes(options: tuple, vowels: bool, limit: int = 32) -> list:
    """Whole-name codes choosing a reading per word (Hangul runs have one reading).

    Enumerates every combination when there are at most `limit`; otherwise only the
    whole-name modes."""
    distinct = [tuple(dict.fromkeys(o)) for o in options]
    n = 1
    for o in distinct:
        n *= len(o)
    if n == 1:
        return [_finish("".join(o[0] for o in distinct), vowels)]
    if n > limit:
        return list(dict.fromkeys(mode_code(options, m, vowels) for m in range(len(SKEL_MODES))))
    combos = [""]
    for o in distinct:
        combos = [c + x for c in combos for x in o]
    return list(dict.fromkeys(_finish(c, vowels) for c in combos))


# ------------------------------------------------- Hangul letter-name acronyms
# Korean writes Latin acronyms with the names of the letters (KMTC -> 케이엠티씨,
# SITC -> 에스아이티씨, J&N -> 제이앤).  A Hangul token that is entirely a sequence of
# letter names is read back as the acronym, so it compares with the English name
# letter for letter instead of through a romanization of the letter names.
LETTER_NAMES = {
    "에이": "A", "비": "B", "씨": "C", "시": "C", "디": "D", "이": "E", "에프": "F", "지": "G",
    "에이치": "H", "에치": "H", "에이취": "H", "아이": "I", "제이": "J", "케이": "K", "엘": "L",
    "엠": "M", "엔": "N", "오": "O", "피": "P", "큐": "Q", "알": "R", "아르": "R", "에스": "S",
    "에쓰": "S", "티": "T", "유": "U", "브이": "V", "더블유": "W", "더블류": "W", "엑스": "X",
    "와이": "Y", "제트": "Z",
}
AND_NAME = "앤"  # '&' between letters (C&C -> 시앤시, written CNC in the catalog)
# letter names that may stand alone as a one-letter token (multi-syllable and not a common word)
SINGLE_LETTER_OK = frozenset(["에이", "에스", "에쓰", "에이치", "케이", "제이", "와이", "브이", "엑스", "에프", "더블유"])
_LN_MAX = max(len(k) for k in LETTER_NAMES)
# Also split 'acronym+word' tokens (엠에스살비아 -> MS 살비아) and learn dictionary entries
# from such splits.  Off: CV showed no gain (base already matches these through the
# skeleton) while the extra training-side changes perturbed the ranker.
ACRONYM_GLUED = False


def letter_parse(tok: str, min_letters: int = 2) -> str | None:
    """Read a Hangul token made only of letter names as its letters (fewest letters wins).

    Returns None unless the whole token parses into at least `min_letters` letters.
    '앤' (and) is accepted between letters and read as N.
    """
    n = len(tok)
    best: list = [None] * (n + 1)
    best[0] = ""
    for i in range(n):
        if best[i] is None:
            continue
        for w in range(1, min(_LN_MAX, n - i) + 1):
            s = tok[i: i + w]
            if s in LETTER_NAMES:
                letter = LETTER_NAMES[s]
            elif s == AND_NAME and i > 0:
                letter = "N"
            else:
                continue
            cand = best[i] + letter
            if best[i + w] is None or len(cand) < len(best[i + w]):
                best[i + w] = cand
    out = best[n]
    if out is None or len(out) < min_letters:
        return None
    return out


def acronym_readings(tok: str) -> list:
    """Readings of a Hangul token as (ACRONYM, rest); rest is '' when the whole token is letters.

    * whole token = letter names (>= 2 letters, or one safe multi-syllable letter name);
    * token = letter-name prefix (>= 2 letters) glued to a word of >= 2 syllables
      (엠에스살비아 -> MS + 살비아), longest prefix first.
    """
    if not tok or not all("가" <= ch <= "힣" for ch in tok):
        return []
    full = letter_parse(tok)
    if full:
        return [(full, "")]
    if tok in SINGLE_LETTER_OK:
        return [(LETTER_NAMES[tok], "")]
    out = []
    if not ACRONYM_GLUED:
        return out
    for p in range(len(tok) - 2, 1, -1):
        a = letter_parse(tok[:p])
        if a:
            out.append((a, tok[p:]))
    return out


def grams(value: str, width: int) -> frozenset:
    if len(value) < width:
        return frozenset((value,)) if value else frozenset()
    return frozenset(value[i: i + width] for i in range(len(value) - width + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


TON_UNIT_RE = re.compile(r"(G/?T|GRT|TONS?|톤|T)$")


def positive(value: object) -> float:
    """Tonnage as a positive float, else NaN.  Text such as '12,345', '12 345' or '12345 GT' is read
    as 12345 (thousands separators and a trailing unit are dropped)."""
    if isinstance(value, str):
        value = TON_UNIT_RE.sub("", re.sub(r"[,\s]", "", unicodedata.normalize("NFKC", value)).upper())
    try:
        x = float(value)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) and x > 0 else np.nan


def rel_gap(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return 2.0
    return min(abs(a - b) / max(a, b, 1.0), 2.0)


# ------------------------------------------------------------ catalog index
TOK_FUZZY_MIN = 4  # record words this long also match catalog words one edit away


class Catalog:
    def __init__(self, catalog: pd.DataFrame):
        self.df = catalog.reset_index(drop=True)
        self.ids = self.df["vessel_id"].astype(str).tolist()
        self.id_to_index = {v: i for i, v in enumerate(self.ids)}
        self.clean = [clean(v) for v in self.df["vessel_name"]]
        self.key = [key(v) for v in self.clean]
        cache_c, cache_v = {}, {}
        # per-run codes under every skeleton mode (primary / alternate / alternate keeping R)
        self.skel_opts = [skeleton_options(v, False, cache_c) for v in self.clean]
        self.skelv_opts = [skeleton_options(v, True, cache_v) for v in self.clean]
        self.skel = [mode_code(o, 0, False) for o in self.skel_opts]
        self.skelv = [mode_code(o, 0, True) for o in self.skelv_opts]
        self.skel_alt = {m: [mode_code(o, m, False) for o in self.skel_opts] for m in SKEL_MODES[1:]}
        self.skelv_alt = {m: [mode_code(o, m, True) for o in self.skelv_opts] for m in SKEL_MODES[1:]}
        self.tokens = [frozenset(v.split()) for v in self.clean]
        self.gt = pd.to_numeric(self.df["gross_tonnage"], errors="coerce").to_numpy(float)
        self.dwt = pd.to_numeric(self.df["deadweight_tonnage"], errors="coerce").to_numpy(float)
        self.log_gt = np.log(np.where(self.gt > 0, self.gt, np.nan))
        self.type = self.df["vessel_type"].fillna("").astype(str).tolist()
        self.flag = self.df["flag_name"].fillna("").astype(str).tolist()
        self.key_count = Counter(self.key)
        self.skel_count = Counter(self.skel)
        # keys in sorted order: the names that start with a given string are one contiguous run
        self.pref_idx = np.array(sorted(range(len(self.key)), key=lambda i: (self.key[i], i)), dtype=np.int64)
        self.pref_keys = [self.key[i] for i in self.pref_idx]

        self.vec_key = CharTfidf((2, 3))
        self.mat_key = self.vec_key.fit_transform([f" {k} " for k in self.key])
        self.vec_skel = CharTfidf((1, 3))
        self.mat_skel = self.vec_skel.fit_transform([f" {k} " for k in self.skel])
        self.vec_skelv = CharTfidf((2, 4))
        self.mat_skelv = self.vec_skelv.fit_transform([f" {k} " for k in self.skelv])
        # alternate codes live in the primary vector spaces (same n-gram vocabulary and idf)
        self.mat_skel_alt = {m: self.vec_skel.transform([f" {k} " for k in v]) for m, v in self.skel_alt.items()}
        self.mat_skelv_alt = {m: self.vec_skelv.transform([f" {k} " for k in v]) for m, v in self.skelv_alt.items()}
        # retrieve() multiplies by the transposed matrices; scipy converted each one to CSR on every
        # batch, so the same CSR matrices are made once here
        self.t_key, self.t_skel, self.t_skelv = (m.T.tocsr() for m in (self.mat_key, self.mat_skel, self.mat_skelv))
        self.t_skel_alt = {m: v.T.tocsr() for m, v in self.mat_skel_alt.items()}
        self.t_skelv_alt = {m: v.T.tocsr() for m, v in self.mat_skelv_alt.items()}
        # token blocking (Papadakis et al. 2013): word -> catalog rows, plus a one-deletion index of
        # the words (FastSS, Bocek et al. 2007) so a record word with one typo still finds its block
        post = defaultdict(list)
        for i, toks in enumerate(self.tokens):
            for w in toks:
                post[w].append(i)
        self.tok_post = {w: np.array(v, dtype=np.int64) for w, v in post.items()}
        self.tok_del = defaultdict(list)
        for w in sorted(self.tok_post):
            if len(w) >= TOK_FUZZY_MIN and not w.isdigit():
                for d in {w} | {w[:j] + w[j + 1:] for j in range(len(w))}:
                    self.tok_del[d].append(w)
        self.tok_cache = {}
        self.type_arr = np.array(self.type, dtype=object)
        self.flag_arr = np.array(self.flag, dtype=object)
        # gross tonnage -> catalog rows with exactly that tonnage (the EXACT_CHANNEL block)
        gt_rows = defaultdict(list)
        for i, v in enumerate(self.gt):
            if v > 0:
                gt_rows[float(v)].append(i)
        self.gt_index = {v: np.array(r, dtype=np.int64) for v, r in gt_rows.items()}
        self._skel_index = None

    def skel_index(self):
        """(10/31) Phonetic-skeleton indexes for the Hangul skeleton channels, built on first use: whole-name skeletons
        of every reading in sorted order (prefix lookup) and word skeleton -> catalog rows (word blocking)."""
        if getattr(self, "_skel_index", None) is None:
            names, rows = [], []
            for m in SKEL_MODES:
                codes = self.skel if m == 0 else self.skel_alt[m]
                for i, c in enumerate(codes):
                    if c:
                        names.append(c)
                        rows.append(i)
            order = sorted(range(len(names)), key=lambda k: (names[k], rows[k]))
            post = defaultdict(set)
            cache = {}
            for i, v in enumerate(self.clean):
                for w in v.split():
                    if w.isdigit():
                        continue
                    codes = cache.get(w)
                    if codes is None:
                        codes = cache[w] = {_finish(c, False) for c in _run_codes(w, False)} - {""}
                    for c in codes:
                        post[c].add(i)
            self._skel_index = ([names[k] for k in order], np.array([rows[k] for k in order], dtype=np.int64),
                                {c: np.array(sorted(r), dtype=np.int64) for c, r in post.items()})
        return self._skel_index

    def token_block(self, word: str) -> np.ndarray:
        """Sorted catalog rows having `word`, or (words of TOK_FUZZY_MIN+ letters) a word one
        insertion, deletion, substitution or swap away."""
        hit = self.tok_cache.get(word)
        if hit is not None:
            return hit
        words = {word} if word in self.tok_post else set()
        if len(word) >= TOK_FUZZY_MIN and not word.isdigit():
            near = set()
            for d in {word} | {word[:j] + word[j + 1:] for j in range(len(word))}:
                near.update(self.tok_del.get(d, ()))
            words |= {w for w in near if osa(word, w) <= 1}
        hit = (np.unique(np.concatenate([self.tok_post[w] for w in sorted(words)])) if words
               else np.empty(0, dtype=np.int64))
        self.tok_cache[word] = hit
        return hit



class CharTfidf:
    """Character n-gram TF-IDF fitted on catalog strings.

    Same output as TfidfVectorizer(analyzer="char", sublinear_tf=True, dtype=np.float32) under
    numpy 2, but the idf vector is always computed in float32. With 78,595 catalog names
    scikit-learn's `n_samples / df` becomes float64 under numpy 1.x (legacy value-based casting),
    which moved cosine scores by ~1e-7 between Colab (numpy 2) and the evaluation server (numpy 1).
    Needs scikit-learn >= 1.5 (idf_ is a plain float32 attribute there; 1.6+ computes the same).
    """

    def __init__(self, ngram_range: tuple):
        self.counter = CountVectorizer(analyzer="char", ngram_range=ngram_range, dtype=np.float32)
        self.tfidf = TfidfTransformer(sublinear_tf=True)

    def fit_transform(self, docs: list):
        counts = self.counter.fit_transform(docs)
        self.tfidf.fit(counts)
        df = np.bincount(counts.indices, minlength=counts.shape[1]).astype(np.float32)
        df += 1.0
        self.tfidf.idf_ = np.log(np.float32(counts.shape[0] + 1) / df) + np.float32(1.0)
        return self.tfidf.transform(counts, copy=False)

    def transform(self, docs: list):
        return self.tfidf.transform(self.counter.transform(docs), copy=False)


# ------------------------------------------- learned Hangul -> English transliteration
# Character-level joint "pair-unit" model in the spirit of stochastic edit distance
# (Ristad & Yianilos 1998) with many-to-many units (Jiampojamarn et al. 2007).  A Hangul
# name is written as conjoining jamo (NFD: initial / vowel / final are distinct symbols),
# the English name as letters.  A name pair is generated as a sequence of units
# (jamo chunk of 0-2, letter chunk of 0-3); unit probabilities are learned with EM
# (forward-backward over all monotone segmentations) from train pairs, then smoothed
# towards a back-off P(type) * prod q_x(jamo) * prod q_y(letter).  A (query, candidate)
# pair is scored by the Viterbi log-probability of the best segmentation.
XL_DXMAX = 1        # jamo per unit (jamo are already phoneme-sized; 2 overfits on ~500 pairs)
XL_TYPES = ((1, 0), (0, 1)) + tuple((dx, dy) for dx in range(1, XL_DXMAX + 1) for dy in (1, 2, 3))
XL_MAXLEN = 16       # longer training strings are skipped (10/27: 30 -> 16; the longest train pair is 16 jamo /
                     # 16 letters, so train-only tables are unchanged; it bounds the EM arrays when LLM spellings are added)
XL_ITERS = 8
XL_ALPHA = 5.0       # back-off pseudo-count mass
XL_NEG = -1e9


def _jamo(text: str) -> str:
    """Hangul syllables of `text` as conjoining jamo; everything else is dropped."""
    return "".join(unicodedata.normalize("NFD", ch) for ch in text if "가" <= ch <= "힣")


def _letters(text: str) -> str:
    return "".join(ch for ch in text if "A" <= ch <= "Z")


def xlit_strings(qclean: str, cclean: str):
    """(jamo of the Hangul part of the query, letters of the candidate) for scoring.

    Latin query tokens that also appear in the candidate are identity matches and are
    removed from both sides; digits are ignored.
    """
    qlat = {t for t in qclean.split() if t.isalpha() and not is_hangul(t)}
    y = "".join(_letters(t) for t in cclean.split() if t not in qlat)
    return _jamo(qclean), y


def _xlit_pairs(train: pd.DataFrame, cat: "Catalog") -> list:
    """Unique (jamo, letters) training pairs: word by word when the token counts agree."""
    pairs = set()
    idx = train["target_vessel_id"].map(cat.id_to_index)
    for q, i in zip(train["vessel_name"], idx):
        qc = clean(q)
        if not is_hangul(qc):
            continue
        qt = [t for t in qc.split() if not t.isdigit()]
        qlat = {t for t in qt if t.isalpha() and not is_hangul(t)}
        ct = [t for t in cat.clean[i].split() if not t.isdigit() and t not in qlat]
        ct = [_letters(t) for t in ct if _letters(t)]
        hq = [t for t in qt if t not in qlat]
        if len(hq) == len(ct) and all(all("가" <= ch <= "힣" for ch in t) for t in hq):
            cand = [(_jamo(a), b) for a, b in zip(hq, ct)]
        else:
            cand = [(_jamo("".join(hq)), "".join(ct))]
        for a, b in cand:
            if a and b and len(a) <= XL_MAXLEN and len(b) <= XL_MAXLEN and 0.25 <= len(b) / len(a) <= 3.0:
                pairs.add((a, b))
    return sorted(pairs)


def build_xlit(train: pd.DataFrame, cat: "Catalog") -> dict:
    pairs = _xlit_pairs(train, cat)
    if not pairs:
        return {}
    P = len(pairs)
    N = max(len(a) for a, _ in pairs)
    M = max(len(b) for _, b in pairs)
    T = len(XL_TYPES)
    ids: dict = {}
    keyid = np.full((P, N + 1, M + 1, T), -1, dtype=np.int64)
    for p, (x, y) in enumerate(pairs):
        n, m = len(x), len(y)
        for i in range(n + 1):
            for j in range(m + 1):
                for t, (dx, dy) in enumerate(XL_TYPES):
                    if i >= dx and j >= dy:
                        keyid[p, i, j, t] = ids.setdefault((x[i - dx: i], y[j - dy: j]), len(ids))
    K = len(ids)
    keyid[keyid < 0] = K                     # slot K has probability 0
    nlen = np.array([len(a) for a, _ in pairs])
    mlen = np.array([len(b) for _, b in pairs])
    ar = np.arange(P)
    t_ins = XL_TYPES.index((0, 1))
    pr = np.ones(K + 1)
    pr[K] = 0.0
    counts = np.zeros(K + 1)
    for _ in range(XL_ITERS):
        lp = pr[keyid]                       # (P, N+1, M+1, T)
        A = np.zeros((P, N + 1, M + 1))
        for i in range(N + 1):
            row = np.zeros((P, M + 1))
            if i == 0:
                row[:, 0] = 1.0
            for t, (dx, dy) in enumerate(XL_TYPES):
                if dx and i >= dx:
                    row[:, dy:] += A[:, i - dx, : M + 1 - dy] * lp[:, i, dy:, t]
            ins = lp[:, i, :, t_ins]
            for j in range(1, M + 1):
                row[:, j] += row[:, j - 1] * ins[:, j]
            A[:, i] = row
        Z = A[ar, nlen, mlen]
        B = np.zeros((P, N + 1, M + 1))
        B[ar, nlen, mlen] = 1.0
        for i in range(N, -1, -1):
            row = B[:, i].copy()
            for t, (dx, dy) in enumerate(XL_TYPES):
                if dx and i + dx <= N:
                    row[:, : M + 1 - dy] += lp[:, i + dx, dy:, t] * B[:, i + dx, dy:]
            ins = lp[:, i, :, t_ins]
            for j in range(M - 1, -1, -1):
                row[:, j] += ins[:, j + 1] * row[:, j + 1]
            B[:, i] = row
        ok = Z > 0
        w = np.where(ok, 1.0 / np.where(ok, Z, 1.0), 0.0)
        counts = np.zeros(K + 1)
        for t, (dx, dy) in enumerate(XL_TYPES):
            post = A[:, : N + 1 - dx, : M + 1 - dy] * lp[:, dx:, dy:, t] * B[:, dx:, dy:] * w[:, None, None]
            counts += np.bincount(keyid[:, dx:, dy:, t].ravel(), weights=post.ravel(), minlength=K + 1)
        counts[K] = 0.0
        pr = counts / counts.sum()
        pr[K] = 0.0
    # smoothing: p(u) = (c(u) + alpha * b(u)) / (C + alpha), b = P(type) * prod q_x * prod q_y
    keys = list(ids)
    C = float(counts[:K].sum())
    cx, cy = Counter(), Counter()
    for a, b in pairs:
        cx.update(a)
        cy.update(b)
    nx, ny = sum(cx.values()), sum(cy.values())
    lx = {s: float(np.log((c + 0.5) / (nx + 0.5 * (len(cx) + 1)))) for s, c in cx.items()}
    ly = {s: float(np.log((c + 0.5) / (ny + 0.5 * (len(cy) + 1)))) for s, c in cy.items()}
    lx_unk = float(np.log(0.5 / (nx + 0.5 * (len(cx) + 1))))
    ly_unk = float(np.log(0.5 / (ny + 0.5 * (len(cy) + 1))))
    tcount = Counter()
    for k, (a, b) in enumerate(keys):
        tcount[(len(a), len(b))] += counts[k]
    lt = {ty: float(np.log(XL_ALPHA * (tcount[ty] + 1.0) / (C + len(XL_TYPES))) - np.log(C + XL_ALPHA))
          for ty in XL_TYPES}
    tab: dict = {}
    for k, (a, b) in enumerate(keys):
        c = counts[k]
        if c < 0.01:
            continue
        back = np.exp(lt[(len(a), len(b))] + sum(lx.get(s, lx_unk) for s in a) + sum(ly.get(s, ly_unk) for s in b))
        tab.setdefault(a, {})[b] = float(np.log(c / (C + XL_ALPHA) + back))
    return {"tab": tab, "lx": lx, "ly": ly, "lx_unk": lx_unk, "ly_unk": ly_unk, "lt": lt}


def xlit_prepare(xl: dict, x: str):
    """Per-query structures: for each end position i, the x chunks ending there."""
    tab, lx, lx_unk = xl["tab"], xl["lx"], xl["lx_unk"]
    xs = [()]
    for i in range(1, len(x) + 1):
        lst = []
        for dx in range(1, XL_DXMAX + 1):
            if i >= dx:
                s = x[i - dx: i]
                lst.append((dx, tab.get(s, {}), sum(lx.get(ch, lx_unk) for ch in s)))
        xs.append(tuple(lst))
    return xs


def xlit_viterbi(xl: dict, xs: list, y: str) -> float:
    """Best-path log-probability of (x, y); xs = xlit_prepare(xl, x)."""
    ly, ly_unk, lt = xl["ly"], xl["ly_unk"], xl["lt"]
    ins_row = xl["tab"].get("", {})
    lt10 = lt[(1, 0)]
    lt01 = lt[(0, 1)]
    ltd = {dx: {dy: lt[(dx, dy)] for dy in (1, 2, 3)} for dx in range(1, XL_DXMAX + 1)}
    m = len(y)
    ys = [()]
    ins = [0.0]
    for j in range(1, m + 1):
        lst = []
        for dy in (1, 2, 3):
            if j >= dy:
                s = y[j - dy: j]
                lst.append((dy, s, sum(ly.get(ch, ly_unk) for ch in s)))
        ys.append(lst)
        v = ins_row.get(y[j - 1])
        ins.append(v if v is not None else lt01 + lst[0][2])
    n = len(xs) - 1
    D = [[XL_NEG] * (m + 1) for _ in range(n + 1)]
    D0 = D[0]
    D0[0] = 0.0
    for j in range(1, m + 1):
        D0[j] = D0[j - 1] + ins[j]
    for i in range(1, n + 1):
        Di = D[i]
        for j in range(m + 1):
            best = XL_NEG
            for dx, row, bx in xs[i]:
                Dp = D[i - dx]
                if dx == 1:
                    v = row.get("")
                    s = Dp[j] + (v if v is not None else lt10 + bx)
                    if s > best:
                        best = s
                lt_dx = ltd[dx]
                for dy, s_y, by in ys[j]:
                    v = row.get(s_y)
                    s = Dp[j - dy] + (v if v is not None else lt_dx[dy] + bx + by)
                    if s > best:
                        best = s
            if j:
                s = Di[j - 1] + ins[j]
                if s > best:
                    best = s
            Di[j] = best
    return D[n][m]


# ------------------------------------- context-dependent transliteration (joint-unit bigram)
# The xlit units are context free, so a choice that depends on the neighbouring sounds
# (c -> ㅋ or ㅅ, -s -> 스, or -> 오/어, ch -> ㅊ/ㅋ, a silent final e) is scored by its overall
# frequency.  Joint-sequence ("graphone") n-gram (Bisani & Ney 2008; joint source-channel model,
# Li, Zhang & Su 2004): every training pair is segmented into xlit units by Viterbi, and a bigram
# over the unit sequences (with <s> / </s>; interpolated absolute discounting whose unigram level
# is the xlit unit model itself) is estimated.  A (query, candidate) pair is scored by an exact
# Viterbi DP over (jamo position, letter position, type of the last unit): the last unit is fixed
# by its type and end cell, so 6 states per cell suffice (units hold at most one jamo).
# Unit id = jamo symbol (0 = none) * TNG_NCH + letter chunk code (base 27, 0 = empty chunk).
assert XL_TYPES == ((1, 0), (0, 1), (1, 1), (1, 2), (1, 3))   # the DP below assumes XL_DXMAX = 1
TNG_JAMO = ([chr(c) for c in range(0x1100, 0x1113)] + [chr(c) for c in range(0x1161, 0x1176)]
            + [chr(c) for c in range(0x11A8, 0x11C3)])        # initial / vowel / final jamo
TNG_XSYM = {ch: k + 1 for k, ch in enumerate(TNG_JAMO)}
TNG_NXS = len(TNG_JAMO) + 1
TNG_NCH = 27 ** 3
TNG_NU = TNG_NXS * TNG_NCH
TNG_BOS, TNG_EOS = TNG_NU, TNG_NU + 1
TNG_NH = TNG_NU + 2          # unit ids + <s> + </s>
TNG_TB = len(XL_TYPES)       # pseudo type of <s>
TNG_NEG = -1e30
TNG_TOL = 1e-9               # segmentation scores closer than this are ties
_TNG_DIG = np.stack([np.arange(TNG_NCH) % 27, np.arange(TNG_NCH) // 27 % 27, np.arange(TNG_NCH) // 729])
_TNG_CHLEN = np.full(TNG_NCH, -1)        # letters in a chunk code (-1 = not a chunk)
_TNG_CHLEN[0] = 0
_TNG_CHLEN[(_TNG_DIG[0] > 0) & (_TNG_DIG[1] == 0) & (_TNG_DIG[2] == 0)] = 1
_TNG_CHLEN[(_TNG_DIG[0] > 0) & (_TNG_DIG[1] > 0) & (_TNG_DIG[2] == 0)] = 2
_TNG_CHLEN[(_TNG_DIG[0] > 0) & (_TNG_DIG[1] > 0) & (_TNG_DIG[2] > 0)] = 3


def _tng_chunk(s: str) -> int:
    return sum((ord(ch) - 64) * 27 ** k for k, ch in enumerate(s))


def _xlit_dense(xl: dict) -> np.ndarray:
    """xlit unit log-probabilities as a dense table over unit ids: the scores xlit_viterbi uses
    (unseen units take the same back-off sums, added in the same order)."""
    lyv = np.array([0.0] + [xl["ly"].get(chr(64 + k), xl["ly_unk"]) for k in range(1, 27)])
    lxv = np.array([xl["lx"].get(ch, xl["lx_unk"]) for ch in TNG_JAMO])
    chly = lyv[_TNG_DIG[0]] + lyv[_TNG_DIG[1]] + lyv[_TNG_DIG[2]]
    L = np.full((TNG_NXS, TNG_NCH), TNG_NEG)
    for dy in range(4):
        mk = _TNG_CHLEN == dy
        L[1:, mk] = xl["lt"][(1, dy)] + lxv[:, None] + chly[mk][None, :]
    mk = _TNG_CHLEN == 1
    L[0, mk] = xl["lt"][(0, 1)] + chly[mk]
    for a, row in xl["tab"].items():
        xs = TNG_XSYM[a] if a else 0
        for b, v in row.items():
            L[xs, _tng_chunk(b)] = v
    return L.ravel()


def tng_level1(xl: dict, tng: dict) -> np.ndarray:
    """Level 1 of the bigram over unit ids, <s> and </s>: the xlit units scaled by 1 - P(</s>).

    Rebuilt from the frozen xlit table when features are computed instead of being stored in
    model.pkl (10.7 MB); additions only, so it has the same bits under numpy 1 and numpy 2."""
    l1 = np.full(TNG_NH, TNG_NEG)
    l1[:TNG_NU] = _xlit_dense(xl) + tng["shift"]
    l1[TNG_EOS] = tng["end"]
    return l1


def _tng_segment(L1: np.ndarray, x: str, y: str) -> list:
    """Unit ids of the best segmentation of (x, y) under the context-free unit table L1.

    A score must beat the best so far by more than TNG_TOL, so (near) ties keep the first unit in
    the fixed order deletion, 1 / 2 / 3 letters, insertion.  numpy 1 and 2 sum the xlit EM counts
    in a different order; with a strict '>' the last-bit differences split exact ties differently."""
    n, m = len(x), len(y)
    D = [[TNG_NEG] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    yc = {(j, dy): _tng_chunk(y[j - dy: j]) for j in range(m + 1) for dy in (1, 2, 3) if j >= dy}
    for i in range(n + 1):
        base = TNG_XSYM[x[i - 1]] * TNG_NCH if i else 0
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best, arg = TNG_NEG, None
            if i:
                for dy in (0, 1, 2, 3):
                    if j >= dy:
                        u = base + (yc[(j, dy)] if dy else 0)
                        s = D[i - 1][j - dy] + L1[u]
                        if s > best + TNG_TOL:
                            best, arg = s, (i - 1, j - dy, u)
            if j:
                u = yc[(j, 1)]
                s = D[i][j - 1] + L1[u]
                if s > best + TNG_TOL:
                    best, arg = s, (i, j - 1, u)
            D[i][j], back[i][j] = best, arg
    units = []
    i, j = n, m
    while i or j:
        i, j, u = back[i][j]
        units.append(int(u))
    return units[::-1]


def build_tng(train: pd.DataFrame, cat: "Catalog", xl: dict) -> dict:
    """Joint-unit bigram over the xlit training pairs; xl = build_xlit(train, cat) of the same rows.

    P(u | h) = max(c(h, u) - D, 0) / c(h) + D * n1(h) / c(h) * P1(u), D = n1 / (n1 + 2 n2).
    Stored: sorted bigram keys h * TNG_NH + u with log P(u | h), sorted histories h with the log
    back-off weight D * n1(h) / c(h), and the level-1 shift log(1 - P(</s>)) and log P(</s>)."""
    if not xl:
        return {}
    L1 = _xlit_dense(xl)
    seqs = [_tng_segment(L1, a, b) for a, b in _xlit_pairs(train, cat)]
    p_end = len(seqs) / (sum(len(s) for s in seqs) + len(seqs))
    tng = {"shift": math.log(1.0 - p_end), "end": math.log(p_end)}
    l1 = tng_level1(xl, tng)
    cnt = Counter()
    for s in seqs:
        cnt.update(zip([TNG_BOS] + s, s + [TNG_EOS]))
    c_h, n1 = Counter(), Counter()
    for (h, _), c in cnt.items():
        c_h[h] += c
        n1[h] += 1
    cc = Counter(cnt.values())
    disc = cc[1] / (cc[1] + 2 * cc[2]) if cc[1] else 0.5
    hists, pairs = sorted(c_h), sorted(cnt)
    tng["gk"] = np.array(hists, dtype=np.int64)
    tng["gv"] = np.array([math.log(disc * n1[h] / c_h[h]) for h in hists])
    tng["k"] = np.array([h * TNG_NH + u for h, u in pairs], dtype=np.int64)
    tng["v"] = np.array([math.log(max(cnt[h, u] - disc, 0.0) / c_h[h] + disc * n1[h] / c_h[h] * math.exp(l1[u]))
                         for h, u in pairs])
    return tng


def _sorted_get(keys: np.ndarray, vals: np.ndarray, q: np.ndarray):
    """(found, value) of every query id in a sorted key array."""
    if not len(keys):
        return np.zeros(q.shape, bool), np.zeros(q.shape)
    pos = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
    return keys[pos] == q, vals[pos]


_TNG_LOOKUP = []   # [(tng["k"], tables)]: the last few bigrams' dense lookup tables (not stored in model.pkl)


def _tng_tables(tng: dict):
    """Dense lookup tables of a bigram (built once per bigram, kept for the last 4 used): the back-off weight of
    every history and, for the histories / units that occur in a stored bigram, the index of that bigram's value.
    They give exactly the values of the sorted key arrays (10/27: about 3x faster than np.searchsorted)."""
    k = tng["k"]
    for kk, tab in _TNG_LOOKUP:
        if kk is k:
            return tab
    gw = np.zeros(TNG_NH)
    gw[tng["gk"]] = tng["gv"]
    hk, uk = k // TNG_NH, k % TNG_NH
    hs, us = np.unique(hk), np.unique(uk)
    hidx = np.full(TNG_NH, -1, dtype=np.int32)
    uidx = np.full(TNG_NH, -1, dtype=np.int32)
    hidx[hs] = np.arange(len(hs), dtype=np.int32)
    uidx[us] = np.arange(len(us), dtype=np.int32)
    pos = np.full((len(hs), len(us)), -1, dtype=np.int32)
    pos[hidx[hk], uidx[uk]] = np.arange(len(k), dtype=np.int32)
    tab = (gw, hidx, uidx, pos)
    _TNG_LOOKUP.insert(0, (k, tab))
    del _TNG_LOOKUP[4:]
    return tab


def _tng_lp(tng: dict, l1: np.ndarray, h: np.ndarray, u: np.ndarray) -> np.ndarray:
    """log P(u | h) for arrays of unit ids (-1 = no unit -> TNG_NEG)."""
    valid = (u >= 0) & (h >= 0)
    h = np.where(valid, h, 0).astype(np.int64)
    u = np.where(valid, u, 0).astype(np.int64)
    if not len(tng["k"]):
        gh, gv = _sorted_get(tng["gk"], tng["gv"], h)
        return np.where(valid, l1[u] + np.where(gh, gv, 0.0), TNG_NEG)
    gw, hidx, uidx, pos = _tng_tables(tng)
    out = l1[u] + gw[h]
    hi, ui = hidx[h], uidx[u]
    m = (hi >= 0) & (ui >= 0)
    p = np.full(h.shape, -1, dtype=np.int32)
    p[m] = pos[hi[m], ui[m]]
    hit = p >= 0
    out[hit] = tng["v"][p[hit]]
    return np.where(valid, out, TNG_NEG)


def _tng_trans(tng: dict, l1: np.ndarray, U: np.ndarray, skip: int) -> dict:
    """{(tp, t): _tng_lp(tng, l1, _tng_shift(U[tp], dx, dy + skip), U[t])} for every unit type t = (dx, dy) and
    previous type tp, with the same values; the per-unit and per-history lookups are made once per type
    instead of once per pair of types (10/27, speed only)."""
    nt = TNG_TB
    if not len(tng["k"]):
        return {(tp, t): _tng_lp(tng, l1, _tng_shift(U[tp], dx, dy + skip), U[t])
                for t, (dx, dy) in enumerate(XL_TYPES) for tp in range(nt + 1)}
    gw, hidx, uidx, pos = _tng_tables(tng)
    v = tng["v"]
    HW, HI = [], []
    for tp in range(nt + 1):
        h0 = np.where(U[tp] >= 0, U[tp], 0)
        HW.append(gw[h0])
        HI.append(np.where(U[tp] >= 0, hidx[h0], -1))
    out = {}
    for t, (dx, dy) in enumerate(XL_TYPES):
        u = U[t]
        uok = u >= 0
        u0 = np.where(uok, u, 0)
        l1u = l1[u0]
        ui = np.where(uok, uidx[u0], -1)
        for tp in range(nt + 1):
            hs = _tng_shift(U[tp], dx, dy + skip)
            valid = uok & (hs >= 0)
            r = l1u + _tng_shift_f(HW[tp], dx, dy + skip)
            hi = _tng_shift(HI[tp], dx, dy + skip)
            m = valid & (hi >= 0) & (ui >= 0)
            p = np.full(u.shape, -1, dtype=np.int32)
            p[m] = pos[hi[m], ui[m]]
            hit = p >= 0
            r[hit] = v[p[hit]]
            out[tp, t] = np.where(valid, r, TNG_NEG)
    return out


def _tng_shift_f(A: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """_tng_shift for a float array (0.0 outside; those cells are masked by the caller)."""
    B = np.zeros(A.shape, dtype=A.dtype)
    n1, m1 = A.shape[-2], A.shape[-1]
    if dx < n1 and dy < m1:
        B[..., dx:, dy:] = A[..., : n1 - dx, : m1 - dy]
    return B


def _tng_grids(x: str, ys: list):
    """U[t, c, i, j] = id of the type-t unit ending at cell (i, j) for candidate c (-1 = none);
    U[TNG_TB] holds <s> at (0, 0).  Also returns the letter counts and their maximum."""
    n, C = len(x), len(ys)
    mlen = np.array([len(y) for y in ys], dtype=np.int64)
    M = int(mlen.max())
    Y = np.zeros((C, M + 3), dtype=np.int64)
    for c, y in enumerate(ys):
        if y:
            Y[c, : len(y)] = np.frombuffer(y.encode("ascii"), dtype=np.uint8) - 64
    jj = np.arange(M + 1)
    chunk = {0: np.where(jj[None, :] <= mlen[:, None], 0, -1)}
    for dy in (1, 2, 3):
        arr = np.full((C, M + 1), -1, dtype=np.int64)
        for j in range(dy, M + 1):
            arr[:, j] = np.where(j <= mlen, sum(Y[:, j - dy + k] * 27 ** k for k in range(dy)), -1)
        chunk[dy] = arr
    xsym = np.array([-1] + [TNG_XSYM[ch] for ch in x], dtype=np.int64)[None, :, None]
    U = np.full((TNG_TB + 1, C, n + 1, M + 1), -1, dtype=np.int64)
    for t, (dx, dy) in enumerate(XL_TYPES):
        ch = chunk[dy][:, None, :]
        if dx:
            U[t] = np.where((xsym >= 0) & (ch >= 0), xsym * TNG_NCH + ch, -1)
        else:
            U[t] = np.where(ch >= 0, ch, -1)
    U[TNG_TB, :, 0, 0] = TNG_BOS
    return U, mlen, M


def _tng_shift(A: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """B[..., i, j] = A[..., i - dx, j - dy] (-1 outside)."""
    B = np.full(A.shape, -1, dtype=A.dtype)
    n1, m1 = A.shape[-2], A.shape[-1]
    if dx < n1 and dy < m1:
        B[..., dx:, dy:] = A[..., : n1 - dx, : m1 - dy]
    return B


def tng_viterbi(tng: dict, l1: np.ndarray, x: str, ys: list, c_del: float | None = None):
    """Best-path log P(x, y) under the joint-unit bigram for every y in ys; l1 = tng_level1(...).

    One DP for all candidates, but each value depends only on its own y (cells past a
    candidate's length are unreachable from its end cell).

    c_del (typo-tolerant score): a letter of y may also be skipped at cost c_del, i.e. the Latin
    name was misspelled (a letter lost) before it was written in Hangul.  The unit after the
    skipped letter is still conditioned on the unit before it; two skips in a row are not allowed.
    K[tp][c, i, j] = a path whose last unit (type tp) ends at (i, j - 1), with letter j skipped."""
    U, mlen, M = _tng_grids(x, ys)
    n, nt = len(x), TNG_TB
    typo = c_del is not None
    t_ins = XL_TYPES.index((0, 1))
    # TR[tp, t][c, i, j] = log P(type-t unit ending at (i, j) | type-tp unit right before it)
    TR = _tng_trans(tng, l1, U, 0)
    # TRk: the same unit after a skipped letter (the unit before it ends one letter earlier)
    TRk = _tng_trans(tng, l1, U, 1) if typo else {}
    V = np.full((nt + 1, len(ys), n + 1, M + 1), TNG_NEG)
    K = np.full((nt + 1, len(ys), n + 1, M + 1), TNG_NEG) if typo else None
    V[TNG_TB, :, 0, 0] = 0.0
    for i in range(n + 1):
        for t, (dx, dy) in enumerate(XL_TYPES):
            if dx and i and dy <= M:
                opts = [V[tp][:, i - 1, : M + 1 - dy] + TR[tp, t][:, i, dy:] for tp in range(nt + 1)]
                if typo:
                    opts += [K[tp][:, i - 1, : M + 1 - dy] + TRk[tp, t][:, i, dy:] for tp in range(nt + 1)]
                V[t][:, i, dy:] = np.max(opts, axis=0)
        for j in range(1, M + 1):   # insertions run along j within the row
            opts = [V[tp][:, i, j - 1] + TR[tp, t_ins][:, i, j] for tp in range(nt + 1)]
            if typo:
                prev = V[:, :, i, j - 1]
                K[:, :, i, j] = np.where(prev > TNG_NEG / 2, prev + c_del, TNG_NEG)
                if j >= 2:
                    opts += [K[tp][:, i, j - 1] + TRk[tp, t_ins][:, i, j] for tp in range(nt + 1)]
            V[t_ins][:, i, j] = np.max(opts, axis=0)
    ar = np.arange(len(ys))
    eos = np.full(len(ys), TNG_EOS, dtype=np.int64)
    ends = [V[t][ar, n, mlen] + _tng_lp(tng, l1, U[t][ar, n, mlen], eos) for t in range(nt)]
    if typo:   # the last letter skipped
        m1 = np.maximum(mlen - 1, 0)
        ends += [np.where(mlen >= 1, K[t][ar, n, mlen] + _tng_lp(tng, l1, U[t][ar, n, m1], eos), TNG_NEG)
                 for t in range(nt + 1)]
    return np.max(ends, axis=0)


# ------------------------------------------- n-best back-transliteration (candidate generation)
# A Hangul name of a vessel that is not in train is matched only through its sound, and the
# n-gram channels compare consonant skeletons, which many catalog names share (구비 -> KP), so the
# true name can fall out of the candidate list, above all when the record has no tonnage.
# Back-transliteration (Knight & Graehl 1998): the joint-unit bigram above generates the most
# likely English spellings of the Hangul name by beam search (n-best, as in Bisani & Ney 2008),
# only the catalog names close to one of them are looked at (letter n-grams), and those are
# re-scored by the exact transliteration score; the best K_GEN join the candidates.
GEN_BEAM = 48      # beam per Hangul token
GEN_OPTS = 10      # spellings tried per jamo (most likely units first)
GEN_INS = 6        # letters that may be inserted between two jamo
GEN_TOK = 12       # spellings kept per token
GEN_NBEST = 30     # spellings kept per name
GEN_POOL = 200     # catalog names nearest one of the spellings (letter n-grams)
K_GEN = 10         # of those, the best by transliteration score join the candidates
GEN_NBEST_SWAP = 10  # spellings kept per other word order
HAN_ORDERS = 4     # word orders scored per Hangul name (written + adjacent swaps)
# 10/24 (Public +4 Hangul rows as an inference-only change of submit_4): extra Hangul candidates
# from the same back-transliteration spellings.  With tonnage: the catalog names within V2_BAND of the
# record tonnage nearest a spelling.  Without tonnage: the same among vessels of a compatible type.
V2_CHANNEL = True
V2_BAND = 0.05     # relative tonnage gap of the V2 band
V2_K = 15          # candidates per V2 channel
# Records without tonnage: the tonnage band of retrieve() was empty (-1 everywhere) and its k_band slots
# went to catalog rows 0..24 (the largest tankers).  They now take the best name scores among vessels
# whose catalog type is compatible with the record type.  "hangul": Hangul records only (the same 25
# candidates submit_5_v2 added to them); True: every record without tonnage (fold 0 with the 10/23
# model: Latin records without tonnage -2 of 2,285); False: off.
NOTON_TYPE_BAND = "hangul"
TYPE_MIN_SHARE = 0.01  # catalog type compatible = >= 1% of train records of that record type had it


def _gen_token(tab: dict, tng: dict, l1: np.ndarray, tok: str) -> list:
    """n-best (log P, letters) of one Hangul token, best first; equal scores in letter order."""
    x = _jamo(tok)
    if not x:
        return []
    ins = sorted(tab.get("", {}).items(), key=lambda t: (-t[1], t[0]))[:GEN_INS]
    ins_u = np.array([_tng_chunk(b) for b, _ in ins], dtype=np.int64)
    sc, H, Y = np.zeros(1), np.array([TNG_BOS], dtype=np.int64), [""]
    for pos, ch in enumerate(x):
        row = sorted(tab.get(ch, {}).items(), key=lambda t: (-t[1], t[0]))[:GEN_OPTS] or [("", 0.0)]
        ju = np.array([TNG_XSYM[ch] * TNG_NCH + (_tng_chunk(b) if b else 0) for b, _ in row], dtype=np.int64)
        if pos and len(ins_u):  # optionally one inserted letter before this jamo
            lp = _tng_lp(tng, l1, np.repeat(H, len(ins_u)), np.tile(ins_u, len(H))).reshape(len(H), len(ins_u))
            s2 = (sc[:, None] + lp).ravel()
            keep = np.argsort(-s2, kind="stable")[:GEN_BEAM]
            a, b = keep // len(ins_u), keep % len(ins_u)
            sc = np.concatenate([sc, s2[keep]])
            H = np.concatenate([H, ins_u[b]])
            Y = Y + [Y[i] + ins[j][0] for i, j in zip(a, b)]
        lp = _tng_lp(tng, l1, np.repeat(H, len(ju)), np.tile(ju, len(H))).reshape(len(H), len(ju))
        s2 = (sc[:, None] + lp).ravel()
        nsc, nH, nY, seen = [], [], [], set()
        for k in np.argsort(-s2, kind="stable"):
            a, b = divmod(int(k), len(ju))
            y = Y[a] + row[b][0]
            if (y, int(ju[b])) in seen:
                continue
            seen.add((y, int(ju[b])))
            nsc.append(s2[k])
            nH.append(ju[b])
            nY.append(y)
            if len(nsc) >= GEN_BEAM:
                break
        sc, H, Y = np.array(nsc), np.array(nH, dtype=np.int64), nY
    sc = sc + _tng_lp(tng, l1, H, np.full(len(H), TNG_EOS, dtype=np.int64))
    best = {}
    for v, y in zip(sc.tolist(), Y):
        if y and (y not in best or v > best[y]):
            best[y] = v
    return sorted(((v, y) for y, v in best.items()), key=lambda t: (-t[0], t[1]))[:GEN_TOK]


def _word_orders(toks: list) -> list:
    """The written word order and, for names of 2+ words, the orders with one adjacent pair of
    words swapped (records can list the words in another order: 글로비스 코러스 / 코러스 글로비스);
    at most HAN_ORDERS orders, written order first."""
    out = [list(toks)]
    for i in range(len(toks) - 1):
        if len(out) >= HAN_ORDERS:
            break
        p = toks[:i] + [toks[i + 1], toks[i]] + toks[i + 2:]
        if p not in out:
            out.append(p)
    return out


def _han_orders(qc: str) -> list:
    """Jamo strings of the Hangul words of a cleaned name in the orders of _word_orders."""
    seen = []
    for p in _word_orders([t for t in qc.split() if is_hangul(t)]):
        x = "".join(_jamo(t) for t in p)
        if x and x not in seen:
            seen.append(x)
    return seen or [_jamo(qc)]


def gen_spellings(pri: "Priors", l1: np.ndarray, qclean: str) -> list:
    """n-best keys (no spaces) of a cleaned Hangul name; Latin / digit tokens are kept as written.
    The written word order gives GEN_NBEST keys, every other order of _word_orders GEN_NBEST_SWAP."""
    tab, tng = pri.xlit.get("tab", {}) if pri.xlit else {}, getattr(pri, "tng", None)
    if not tab or not tng:
        return []
    words = [t for t in qclean.split() if not t.isdigit()]
    digits = [t for t in qclean.split() if t.isdigit()]
    opts = {t: (_gen_token(tab, tng, l1, t) if is_hangul(t) else [(0.0, t)]) for t in words + digits}
    out, seen = [], set()
    for k, order in enumerate(_word_orders(words)):
        combos = [(0.0, "")]
        for t in order + digits:
            if opts[t]:
                combos = sorted(((v + v2, y + y2) for v, y in combos for v2, y2 in opts[t]),
                                key=lambda c: (-c[0], c[1]))[:GEN_NBEST]
        for _, y in combos[: GEN_NBEST if k == 0 else GEN_NBEST_SWAP]:
            if y and y not in seen:
                seen.add(y)
                out.append(y)
    return out


# --------------------------------------------------------- learned priors
class Priors:
    """Conditional tables learned from train: query type/flag -> catalog type/flag."""

    def __init__(self, train: pd.DataFrame, cat: Catalog):
        idx = train["target_vessel_id"].map(cat.id_to_index)
        self.type_tab = self._table(train["vessel_type"].fillna(""), [cat.type[i] for i in idx])
        self.flag_tab = self._table(train["flag_name"].fillna(""), [cat.flag[i] for i in idx])
        # token dictionary hangul token -> english token, from aligned names
        tok = defaultdict(Counter)
        for q, i in zip(train["vessel_name"], idx):
            qc = clean(q)
            if not is_hangul(qc):
                continue
            qt, ct = qc.split(), cat.clean[i].split()
            if ACRONYM_GLUED and len(qt) < len(ct):
                qt = self._unglue(qt, frozenset(ct))
            if len(qt) == len(ct):
                for a, b in zip(qt, ct):
                    if is_hangul(a):
                        tok[a][b] += 1
        self.tok = {a: c.most_common(1)[0][0] for a, c in tok.items()}
        # learned character-level Hangul -> English transliteration model
        self.xlit = build_xlit(train, cat)
        # joint-unit bigram over the xlit units (built on the xlit model just fitted)
        self.tng = build_tng(train, cat, self.xlit)

    @staticmethod
    def _unglue(qt, ctoks):
        """Split a glued 'acronym+word' Hangul token when its acronym is an English token."""
        out = []
        for t in qt:
            for a, rest in acronym_readings(t):
                if rest and a in ctoks:
                    out.extend([t[: len(t) - len(rest)], rest])
                    break
            else:
                out.append(t)
        return out

    @staticmethod
    def _table(src, dst):
        tab = defaultdict(Counter)
        for a, b in zip(src, dst):
            tab[a][b] += 1
        return {a: (c, sum(c.values())) for a, c in tab.items()}

    @staticmethod
    def top(tab, a):
        if a not in tab:
            return ""
        c = tab[a][0]
        return min(c, key=lambda b: (-c[b], b))

    @staticmethod
    def prob(tab, a, b):
        if a not in tab:
            return -1.0
        c, n = tab[a]
        return (c.get(b, 0) + 0.1) / (n + 1.0)

    def translate(self, qclean: str) -> str:
        out = []
        for t in qclean.split():
            if t in self.tok:
                out.append(self.tok[t])
                continue
            reads = acronym_readings(t)
            if not reads:
                out.append(t)
                continue
            # prefer the split whose remaining word the dictionary knows, else the longest prefix
            a, rest = next(((a, r) for a, r in reads if r in self.tok), reads[0])
            out.append(a)
            if rest:
                out.append(self.tok.get(rest, rest))
        return " ".join(out)


# --------------------------------------------------------------- queries
class QueryBatch:
    def __init__(self, df: pd.DataFrame, cat: Catalog, pri: Priors):
        self.df = df.reset_index(drop=True)
        self.clean = [clean(v) for v in self.df["vessel_name"]]
        self.hangul = np.array([is_hangul(v) for v in self.clean])
        self.trans = [pri.translate(c) if h else c for c, h in zip(self.clean, self.hangul)]
        self.key = [key(v) for v in self.trans]
        self.roman = [key(romanize(v)) for v in self.trans]
        self.skel = [skeleton(v) for v in self.trans]
        self.skelv = [skeleton(v, True) for v in self.trans]
        self.gt = np.array([positive(v) for v in self.df["gross_tonnage"]])
        self.igt = np.array([positive(v) for v in self.df["international_gross_tonnage"]])
        self.type = self.df["vessel_type"].fillna("").astype(str).tolist()
        self.flag = self.df["flag_name"].fillna("").astype(str).tolist()
        # most frequent catalog type / flag of train records with this type / flag ('' if unseen)
        self.type_c = [pri.top(pri.type_tab, v) for v in self.type]
        self.flag_c = [pri.top(pri.flag_tab, v) for v in self.flag]
        self.year = pd.to_numeric(self.df.get("arrival_year"), errors="coerce").to_numpy(float) if "arrival_year" in self.df else np.full(len(self.df), np.nan)
        self.m_key = cat.vec_key.transform([f" {k} " for k in self.key])
        self.m_roman = cat.vec_key.transform([f" {k} " for k in self.roman])
        self.m_skel = cat.vec_skel.transform([f" {k} " for k in self.skel])
        self.m_skelv = cat.vec_skelv.transform([f" {k} " for k in self.skelv])
        # alternate codes of the (translated) query; only Hangul rows use them
        self.m_skel_alt = {m: cat.vec_skel.transform([f" {skeleton(v, False, m)} " for v in self.trans])
                           for m in SKEL_MODES[1:]}
        self.m_skelv_alt = {m: cat.vec_skelv.transform([f" {skeleton(v, True, m)} " for v in self.trans])
                            for m in SKEL_MODES[1:]}
        # back-transliteration channel (Hangul rows): spellings, their n-gram vectors, scorer tables
        self.gen_tng = getattr(pri, "tng", None) if pri.xlit else None
        self.gen_l1 = tng_level1(pri.xlit, self.gen_tng) if self.gen_tng and self.hangul.any() else None
        self.gen_ly = (pri.xlit["ly"], pri.xlit["ly_unk"]) if pri.xlit else None
        memo, spell = {}, []
        for c, h, k in zip(self.clean, self.hangul, self.key):
            if self.gen_l1 is None or not h or not k:
                spell.append([])
                continue
            if c not in memo:
                memo[c] = gen_spellings(pri, self.gen_l1, c)
            spell.append(memo[c])
        self.gen_off = np.concatenate([[0], np.cumsum([len(v) for v in spell])]).astype(np.int64)
        flat = [y for v in spell for y in v]
        self.m_gen = cat.vec_key.transform([f" {y} " for y in flat]) if flat else None
        # compatible catalog types per record type (train table); None = unknown type, no restriction.
        # Each mask depends only on the record's own type and the train table (row independent).
        self.type_ok = {}
        for t in sorted(set(self.type)):
            if t in pri.type_tab:
                c, n = pri.type_tab[t]
                self.type_ok[t] = np.isin(cat.type_arr, [k for k, v in c.items() if v / n >= TYPE_MIN_SHARE])
            else:
                self.type_ok[t] = None


_TIE = {}


def _tie(n: int) -> np.ndarray:
    if n not in _TIE:
        _TIE[n] = np.arange(n) * 1e-12
    return _TIE[n]


def _row_top(v: np.ndarray, cols: np.ndarray, k: int) -> np.ndarray:
    """Columns of the k largest v, largest first; equal values in column order (cols ascending)."""
    top = np.argpartition(-v, k - 1)[:k]
    kth = v[top].min()
    if kth != kth:  # NaN among the top (scores are finite in retrieve(); kept for safety)
        return cols[top[np.argsort(-v[top], kind="stable")]]
    sel = np.flatnonzero(v >= kth)  # every value equal to the k-th one, too
    return cols[sel[np.lexsort((sel, -v[sel]))[:k]]]


def _topk(scores: np.ndarray, k: int, step: int = 16, cap: int = 4096) -> np.ndarray:
    """Top-k columns per row of s = score - column * 1e-12, largest first.

    The column term breaks ties by catalog order; values that are still equal (possible only in
    the float64 tonnage matrices, for a non-integer tonnage) are put in column order, so the result
    does not depend on the CPU or the numpy version.
    Partitioning every whole row was half of the inference time, so the k-th largest s of every
    step-th column is taken first: it is a lower bound of the row's k-th largest s, and only the
    columns at or above it are ranked.  score >= s, so `score >= bound` keeps every such column
    (also when numpy 1 compares in float32, since rounding keeps the order)."""
    n = scores.shape[1]
    k = min(k, n)
    tie = _tie(n)
    out = np.empty((scores.shape[0], k), dtype=np.intp)
    if n < 4 * step * k:
        s = scores.astype(np.float64) - tie
        for r in range(scores.shape[0]):
            out[r] = _row_top(s[r], np.arange(n), k)
        return out
    sub = scores[:, ::step].astype(np.float64) - tie[::step]
    bound = -np.partition(-sub, k - 1, axis=1)[:, k - 1]
    for r in range(scores.shape[0]):
        idx = np.flatnonzero(scores[r] >= bound[r])
        if len(idx) < k or len(idx) > cap:  # many equal scores (e.g. no tonnage), or a NaN bound
            s = scores[r].astype(np.float64) - tie
            idx = np.flatnonzero(s >= bound[r]) if len(idx) >= k else np.arange(n)
            v = s[idx]
        else:
            v = scores[r, idx].astype(np.float64) - tie[idx]
        out[r] = _row_top(v, idx, k)
    return out


K_DWT = 5      # candidates whose deadweight is nearest the record tonnage (GT written as DWT)
K_EMPTY = 100  # candidates of a record without a name (nearest tonnage)


def _gen_channel(q: QueryBatch, cat: Catalog, i: int, gap_row: np.ndarray | None = None) -> np.ndarray:
    """Hangul records: the catalog names nearest an n-best back-transliteration of the name
    (GEN_POOL by letter n-grams), re-scored by the transliteration log-odds; the best K_GEN.
    V2_CHANNEL adds the V2_K names nearest a spelling within the tonnage band (record with tonnage)
    or among vessels of a compatible type (record without tonnage)."""
    none = np.empty(0, dtype=np.int64)
    a, b = q.gen_off[i], q.gen_off[i + 1]
    if b <= a or q.m_gen is None:
        return none
    sim = (q.m_gen[a:b] @ cat.t_key).toarray().max(axis=0)
    pool = _topk(sim[None], GEN_POOL)[0]
    qc = q.clean[i]
    qlat = {t for t in qc.split() if t.isalpha() and not is_hangul(t)}
    ys = ["".join(_letters(t) for t in cat.clean[c].split() if t not in qlat) for c in pool]
    uniq = sorted(set(y for y in ys if y))
    base = none
    if uniq:
        pos = {y: k for k, y in enumerate(uniq)}
        ly, ly_unk = q.gen_ly
        J = np.max([tng_viterbi(q.gen_tng, q.gen_l1, x, uniq) for x in _han_orders(qc)], axis=0)
        lo_u = J - np.array([sum(ly.get(ch, ly_unk) for ch in y) for y in uniq])
        lo = np.array([lo_u[pos[y]] if y else -np.inf for y in ys])
        base = pool[np.lexsort((pool, -lo))[:K_GEN]]
    if not V2_CHANNEL:
        return base
    add = []
    if np.isfinite(q.gt[i]) or np.isfinite(q.igt[i]):
        if gap_row is not None:
            v = np.where(gap_row <= V2_BAND, sim, -1.0)
            if (v > -1).any():
                add.append(np.argsort(-v, kind="stable")[:V2_K])
    else:
        m = q.type_ok.get(q.type[i])
        v = np.where(m, sim, -1.0) if m is not None else sim
        add.append(np.argsort(-v, kind="stable")[:V2_K])
    if not add:
        return base
    return np.fromiter(dict.fromkeys(np.concatenate([base] + add).tolist()), dtype=np.int64)


K_TOK = 10     # token-block candidates: catalog names holding every record word, nearest tonnage first
# 10/31: the token block also for Hangul records, on the words already read as Latin (dictionary words such as
# 모닝 -> MORNING, letter names such as 에스티아이 -> STI).  A Hangul name cut to its fleet word (모닝, 에스티아이) found its
# vessel only if the n-gram channels ranked it among the fleet; the block ranks the fleet by tonnage instead
# (fold 0: STI Lombard missing from the candidates of '에스티아이').  False = 10/29.
HANGUL_TOKEN_CHANNEL = False


def _token_channel(q: QueryBatch, cat: Catalog, i: int, gap: np.ndarray, name: np.ndarray) -> np.ndarray:
    """Token blocking for Latin records that lost or garbled a word: the catalog names holding every
    word of the record, ranked by tonnage gap, then type / flag agreement, then name score.
    A common surviving word (STI, SEA, HAFNIA) matches thousands of names, and the n-gram channels
    rank the true one too low; within the block the tonnage usually finds it."""
    none = np.empty(0, dtype=np.int64)
    if q.hangul[i] and not HANGUL_TOKEN_CHANNEL:  # Hangul records keep the n-gram channels (10/29)
        return none
    near = np.flatnonzero(gap <= 0.12) if (q.gt[i] > 0 or q.igt[i] > 0) else np.arange(len(gap))
    if any(cat.key[c] == q.key[i] for c in near[name[near] >= 0.999]):
        return none  # the record name is a catalog name of near tonnage: nothing is lost
    toks = q.trans[i].split()
    if q.hangul[i]:  # 10/31: only the words the dictionary / letter-name reading turned into Latin (MORNING, STI)
        toks = [t for t in toks if not is_hangul(t) and not t.isdigit()]
        if not toks:
            return none
    # a one-letter word is read as an initial (G. SPIRIT for GOLDEN SPIRIT): the name must also have
    # another word starting with that letter
    inits = sorted({w for w in toks if len(w) == 1 and w.isalpha()})
    words = sorted({w for w in toks if w not in inits})
    if not words:
        return none
    blk = None
    for w in sorted(words, key=lambda w: (len(cat.token_block(w)), w)):
        b = cat.token_block(w)
        blk = b if blk is None else np.intersect1d(blk, b, assume_unique=True)
        if not blk.size:
            return blk
    if inits:
        wset = set(words)
        blk = np.array([c for c in blk if all(any(t[0] == a and t not in wset for t in cat.tokens[c]) or a in cat.tokens[c]
                                              for a in inits)], dtype=np.int64)
        if not blk.size:
            return blk
    sc = (np.minimum(gap[blk], 1.0) + 0.1 * (cat.type_arr[blk] != q.type_c[i]) + 0.05 * (cat.flag_arr[blk] != q.flag_c[i])
          - 0.01 * name[blk])
    return blk[np.lexsort((blk, sc))[:K_TOK]]


# 10/28: a second blocking key that does not use the name (multi-pass blocking, Hernandez & Stolfo 1995; Christen
# 2012: several blocking keys on different fields).  Every channel above starts from the record name, so a renamed
# vessel (record name = an old name or another vessel's) is found only if it is among the K_TON nearest tonnages of
# the whole catalog, ties broken by name similarity, which is noise for such a record: 15 catalog car carriers share
# 57,692 GT.  In train 54% of the records with a tonnage (Hangul 72%) carry the answer's catalog gross tonnage exactly
# as their GT or international GT, and a median of 5 catalog vessels share it (4 of a compatible type).  The block:
# the catalog vessels whose gross tonnage equals the record's GT or international GT, compatible type first, then flag
# agreement, then name similarity.
# Off by default (the candidates of submit_14); pipeline.score_candidates() and fit() turn it on for the second pass
# (pipeline.RENAME_PASS) only.
EXACT_CHANNEL = False
K_EXACT = 15


def _exact_channel(q: QueryBatch, cat: Catalog, i: int, name_row: np.ndarray) -> np.ndarray:
    none = np.empty(0, dtype=np.int64)
    keys = sorted({float(v) for v in (q.gt[i], q.igt[i]) if np.isfinite(v) and v > 0})
    if not keys:
        return none
    idx = np.unique(np.concatenate([cat.gt_index.get(v, none) for v in keys]))
    if not len(idx):
        return idx
    m = q.type_ok.get(q.type[i])
    t_bad = (~m[idx]).astype(float) if m is not None else np.zeros(len(idx))
    f_bad = (cat.flag_arr[idx] != q.flag_c[i]).astype(float)
    sc = t_bad + 0.5 * f_bad - 0.01 * name_row[idx]
    return idx[np.lexsort((idx, sc))[:K_EXACT]]


K_PREF = 10    # catalog names that start with the record name (a cut-off name), nearest tonnage first
PREF_MIN = 4   # shortest record key looked up as a prefix


def _prefix_channel(qk: str, cat: Catalog, gap_row: np.ndarray) -> np.ndarray:
    """Records can carry a name cut at the end (GOLDEN SPI): the catalog names that start with the
    record key, the K_PREF nearest in tonnage (ties: shorter name, then catalog order)."""
    if len(qk) < PREF_MIN:
        return np.empty(0, dtype=np.int64)
    lo = bisect.bisect_left(cat.pref_keys, qk)
    hi = bisect.bisect_left(cat.pref_keys, qk + "\uffff")
    idx = cat.pref_idx[lo:hi]
    if len(idx) > K_PREF:
        ln = np.array([len(cat.key[c]) for c in idx])
        idx = idx[np.lexsort((idx, ln, gap_row[idx]))[:K_PREF]]
    return idx


# 10/31: Hangul candidate channels on the phonetic skeleton.  The n-gram channels score the whole name, so a Hangul name
# that lost its last syllables or one of its words ranks its vessel below the many catalog names sharing a few
# consonants.  (a) prefix: catalog names whose skeleton (any reading) starts with the record's skeleton (a cut name),
# (b) word block: catalog names holding a word whose skeleton equals the skeleton of each record word but at most one
# (a dropped / added / garbled word; token blocking, Papadakis et al. 2013, on a phonetic key, Christen 2012 ch. 4).
# Both rank their block by tonnage gap, then type / flag agreement, then name score, like _token_channel.  Only
# Hangul records.  False = the candidates of 10/29.
HANGUL_SKEL_CHANNELS = False
K_SKPREF = 10
K_SKTOK = 10
SKPREF_MIN = 3     # shortest record skeleton looked up as a prefix
SKTOK_MAX = 4000   # skip a word whose block is larger than this (a common skeleton such as 'S')


def _rank_block(q: QueryBatch, cat: Catalog, i: int, blk: np.ndarray, gap: np.ndarray, name: np.ndarray, k: int):
    sc = (np.minimum(gap[blk], 1.0) + 0.1 * (cat.type_arr[blk] != q.type_c[i]) + 0.05 * (cat.flag_arr[blk] != q.flag_c[i])
          - 0.01 * name[blk])
    return blk[np.lexsort((blk, sc))[:k]]


def _skel_channels(q: QueryBatch, cat: Catalog, i: int, gap: np.ndarray, name: np.ndarray) -> np.ndarray:
    none = np.empty(0, dtype=np.int64)
    if not q.hangul[i]:
        return none
    names, rows, post = cat.skel_index()
    out = []
    qs = q.skel[i]
    if len(qs) >= SKPREF_MIN:
        lo = bisect.bisect_left(names, qs)
        hi = bisect.bisect_left(names, qs + "￿")
        if hi > lo:
            out.append(_rank_block(q, cat, i, np.unique(rows[lo:hi]), gap, name, K_SKPREF))
    words = [t for t in q.trans[i].split() if not t.isdigit()]
    codes = []
    for w in words:
        c = {_finish(x, False) for x in _run_codes(w, False)} - {""}
        blk = [post[x] for x in c if x in post]
        b = np.unique(np.concatenate(blk)) if blk else none
        codes.append(b if len(b) <= SKTOK_MAX else None)
    usable = [b for b in codes if b is not None]
    if usable and len(words) >= 1:
        need = max(1, len(usable) - 1) if len(words) >= 2 else 1
        allr = np.concatenate(usable)
        u, cnt = np.unique(allr, return_counts=True)
        blk = u[cnt >= need]
        if len(blk):
            out.append(_rank_block(q, cat, i, blk, gap, name, K_SKTOK))
    if not out:
        return none
    return np.fromiter(dict.fromkeys(np.concatenate(out).tolist()), dtype=np.int64)


def retrieve(q: QueryBatch, cat: Catalog, k_name=30, k_band=25, k_ton=10, batch=50):
    """Return per query: candidate indices and a dict of matrix similarity values."""
    out = []
    n = len(q.df)
    for s in range(0, n, batch):
        e = min(n, s + batch)
        sk = (q.m_key[s:e] @ cat.t_key).toarray()
        sr = (q.m_roman[s:e] @ cat.t_key).toarray()
        ss = (q.m_skel[s:e] @ cat.t_skel).toarray()
        sv = (q.m_skelv[s:e] @ cat.t_skelv).toarray()
        hr = np.flatnonzero(q.hangul[s:e])
        if len(hr):  # Hangul rows: best of primary and alternate readings
            for m in SKEL_MODES[1:]:
                ss[hr] = np.maximum(ss[hr], (q.m_skel_alt[m][s:e][hr] @ cat.t_skel_alt[m]).toarray())
                sv[hr] = np.maximum(sv[hr], (q.m_skelv_alt[m][s:e][hr] @ cat.t_skelv_alt[m]).toarray())
        name = np.maximum(np.maximum(sk, sr), np.maximum(ss, sv))
        top_sk, top_sr = _topk(sk, 300), _topk(sr, 300)
        pool = np.concatenate([top_sk, top_sr], axis=1)
        gt = q.gt[s:e, None]
        igt = q.igt[s:e, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            g1 = np.abs(gt - cat.gt[None]) / np.maximum(gt, cat.gt[None])
            g2 = np.abs(igt - cat.gt[None]) / np.maximum(igt, cat.gt[None])
            gap = np.fmin(g1, g2)
        gap = np.where(np.isfinite(gap), gap, 2.0)
        with np.errstate(invalid="ignore", divide="ignore"):  # record tonnage vs catalog deadweight
            gapd = np.fmin(np.abs(gt - cat.dwt[None]) / np.maximum(gt, cat.dwt[None]),
                           np.abs(igt - cat.dwt[None]) / np.maximum(igt, cat.dwt[None]))
        gapd = np.where(np.isfinite(gapd), gapd, 2.0)
        band = np.where(gap <= 0.12, name, -1.0)
        if NOTON_TYPE_BAND:
            noton = ~(np.isfinite(q.gt[s:e]) | np.isfinite(q.igt[s:e]))
            if NOTON_TYPE_BAND == "hangul":
                noton &= np.asarray(q.hangul[s:e], dtype=bool)
            for r in np.flatnonzero(noton):
                m = q.type_ok.get(q.type[s + r])
                band[r] = np.where(m, name[r], -1.0) if m is not None else name[r]
        combo = name - 0.5 * np.minimum(gap, 1.0)
        # the top of a longer sorted list is the shorter list
        t_name = np.concatenate([top_sk[:, :k_name], top_sr[:, : k_name // 2], _topk(ss, k_name // 2), _topk(sv, k_name // 2)], axis=1)
        t_skel = _topk(name, k_name // 2)
        t_band = _topk(band, k_band)
        t_combo = _topk(combo, k_band)
        t_ton = _topk(-gap - 0.01 * (1 - name), k_ton)
        t_dwt = _topk(-gapd - 0.01 * (1 - name), K_DWT)
        for r in range(e - s):
            qk, qr = q.key[s + r], q.roman[s + r]
            if not qk:  # no name: the nearest tonnages (as GT or as deadweight) are all there is
                seen = dict.fromkeys(np.concatenate([_topk(-gap[r: r + 1], K_EMPTY)[0], t_dwt[r]]).tolist())
                cand = np.fromiter(seen, int)
                out.append((cand, sk[r, cand], sr[r, cand], ss[r, cand], sv[r, cand], gap[r, cand]))
                continue
            pl = np.unique(pool[r])
            # Latin rows: OSA instead of Levenshtein (a swapped pair of letters costs one edit, as in the noise)
            if q.hangul[s + r]:
                lv = np.array([max(lev_ratio(qk, cat.key[c]), lev_ratio(qr, cat.key[c])) for c in pl])
            elif qr == qk:
                lv = np.array([osa_ratio(qk, cat.key[c]) for c in pl])
            else:
                lv = np.array([max(osa_ratio(qk, cat.key[c]), osa_ratio(qr, cat.key[c])) for c in pl])
            t_lev = pl[np.argsort(-lv, kind="stable")[:15]]
            t_gen = _gen_channel(q, cat, s + r, gap[r]) if q.hangul[s + r] else np.empty(0, dtype=np.int64)
            t_tok = _token_channel(q, cat, s + r, gap[r], name[r])
            # (skipped when the record name is a catalog name of near tonnage: nothing was cut)
            t_pref = (np.empty(0, dtype=np.int64) if any(cat.key[c] == qk for c in t_band[r][:5])
                      else _prefix_channel(qk, cat, gap[r]))
            t_ex = _exact_channel(q, cat, s + r, name[r]) if EXACT_CHANNEL else np.empty(0, dtype=np.int64)
            t_sk = (_skel_channels(q, cat, s + r, gap[r], name[r]) if HANGUL_SKEL_CHANNELS
                    else np.empty(0, dtype=np.int64))
            seen = dict.fromkeys(np.concatenate([t_lev, t_name[r], t_skel[r], t_band[r], t_combo[r], t_ton[r], t_dwt[r],
                                                 t_gen, t_tok, t_pref, t_ex, t_sk]).tolist())
            cand = np.fromiter(seen, int)
            out.append((cand, sk[r, cand], sr[r, cand], ss[r, cand], sv[r, cand], gap[r, cand]))
    return out



# ------------------------------------------------- word-level features
# The record name can lose a word, gain one, abbreviate one (HYUNDAI -> HD / HYUN / HYND) or
# join / split words.  Whole-string similarities drop for the true vessel as much as for its
# namesakes, so these features compare the names word by word: each word of one name is matched
# to its best contiguous run of up to 3 words of the other (a run is joined, so spacing changes
# do not matter), in the spirit of Monge & Elkan's (1996) recursive field matching.
W_SPAN = 3
W_HIT = 0.75
# 10/31: a Hangul word is compared with every skeleton reading of the catalog word (primary / alternate / R kept), as
# the whole-name features already do (mixed_codes); 10/29 compared it with the primary reading only.  False = 10/29.
WORD_SIM_ALT = False


def _abbr_of(a: str, b: str) -> bool:
    """a is a short form of the English word b: same first letter and the rest of a in order in b
    (prefix HYUN, initial H, consonants HYND / HD of HYUNDAI, SHPG of SHIPPING)."""
    if not (len(a) < len(b) and a.isalpha() and b.isalpha() and a[0] == b[0]):
        return False
    it = iter(b[1:])
    return all(ch in it for ch in a[1:])


def _word_sim(a: str, b: str, cache: dict) -> tuple:
    """(similarity, matched-as-abbreviation) of a query word a and a catalog run b (joined)."""
    k = (a, b)
    if k in cache:
        return cache[k]
    if a == b:
        out = (1.0, False)
    elif a.isdigit() or b.isdigit():
        out = (0.0, False)
    elif is_hangul(a):  # Hangul word the dictionary did not translate: phonetic skeletons
        modes = SKEL_MODES if WORD_SIM_ALT else (0,)
        out = (max(max(lev_ratio(skeleton(a), skeleton(b, False, m)), lev_ratio(skeleton(a, True), skeleton(b, True, m)))
                   for m in modes), False)
    else:
        s = lev_ratio(a, b)
        ab = _abbr_of(a, b)
        out = (max(s, 0.9 if len(a) >= 3 else 0.8 if len(a) == 2 else 0.7), True) if ab and s < 0.9 else (s, False)
    cache[k] = out
    return out


def _runs(toks: list) -> list:
    return [(i, j, "".join(toks[i:j])) for i in range(len(toks)) for j in range(i + 1, min(len(toks), i + W_SPAN) + 1)]


def _align(qw: list, cw: list, cache: dict) -> tuple:
    """Best run of cw for every word of qw: (mean best sim, share of qw words >= W_HIT,
    share of cw words covered by a hit, abbreviation hits, qw words with best < 0.5)."""
    runs = _runs(cw)
    best, cov, nab, miss = [], set(), 0, 0
    for a in qw:
        sb, ab, span = 0.0, False, None
        for i, j, b in runs:
            sim, isab = _word_sim(a, b, cache)
            if sim > sb:
                sb, ab, span = sim, isab, (i, j)
        best.append(sb)
        if sb >= W_HIT:
            cov.update(range(*span))
            nab += ab
        elif sb < 0.5:
            miss += 1
    n = max(len(qw), 1)
    return sum(best) / n, sum(b >= W_HIT for b in best) / n, len(cov) / max(len(cw), 1), nab, miss


def _words(text: str) -> tuple:
    toks = text.split()
    return [t for t in toks if not t.isdigit()], frozenset(t for t in toks if t.isdigit())


# 10/31: hull numbers written as digits, roman numerals or Hangul number words ('2', 'II', '투') are compared as
# integers in the w_num feature (non-standard-word normalisation, Sproat et al. 2001; vessel registers convert roman
# numerals to arabic before matching).  Only unambiguous positions are read: a roman token of 2+ letters anywhere, a
# one-letter one (I, V, X) only as the last word of a name of 2+ words, a Hangul number word only as its own word.
# The name strings themselves are unchanged (an exact 'AMAL II' still prefers 'Amal II' over 'Amal 2').  False = 10/29.
NUM_NORM = True
ROMAN_VAL = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11,
             "XII": 12, "XIII": 13, "XIV": 14, "XV": 15, "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20}
HANGUL_NUM = {"원": 1, "투": 2, "쓰리": 3, "스리": 3, "포": 4, "파이브": 5, "화이브": 5, "식스": 6, "세븐": 7, "에잇": 8,
              "에이트": 8, "나인": 9, "텐": 10}


def name_numbers(text: str) -> frozenset:
    """Integers written in a cleaned (or translated) name: digits, roman numerals, Hangul number words."""
    toks = text.split()
    out = set()
    for i, t in enumerate(toks):
        if t.isdigit():
            out.add(int(t))
        elif t in ROMAN_VAL and (len(t) >= 2 or (i == len(toks) - 1 and len(toks) >= 2)):
            out.add(ROMAN_VAL[t])
        elif t in HANGUL_NUM and len(toks) >= 2:
            out.add(HANGUL_NUM[t])
    return frozenset(out)


def _word_block(q: "QueryBatch", cat: "Catalog", r: int, cand) -> np.ndarray:
    """Columns: me_q, hit_q, cov_c, abbr_q, miss_q (query words vs catalog runs), me_c, hit_c,
    cov_q (catalog words vs query runs), number state, first-word sim, combined, best-in-record
    combined minus combined."""
    qw, qd = _words(q.trans[r])
    if NUM_NORM:
        qd = name_numbers(q.trans[r]) | name_numbers(q.clean[r])
    cache: dict = {}
    out = np.zeros((len(cand), 12), dtype=np.float32)
    for j, c in enumerate(cand):
        cw, cd = _words(cat.clean[c])
        if NUM_NORM:
            cd = name_numbers(cat.clean[c])
        if qw and cw:
            me_q, hit_q, cov_c, nab, miss = _align(qw, cw, cache)
            me_c, hit_c, cov_q, _, _ = _align(cw, qw, cache) if not q.hangul[r] else _align_rev(qw, cw, cache)
            first = _word_sim(qw[0], cw[0], cache)[0]
        else:
            me_q = hit_q = cov_c = me_c = hit_c = cov_q = first = 0.0
            nab = miss = 0
        num = 0 if not qd and not cd else 1 if qd == cd else 2 if not cd else 3 if not qd else 4
        out[j, :10] = (me_q, hit_q, cov_c, nab, miss, me_c, hit_c, cov_q, num, first)
        out[j, 10] = (me_q + me_c) / 2 + 0.25 * (cov_c + cov_q)
    out[:, 11] = out[:, 10].max() - out[:, 10]
    return out


def _align_rev(qw: list, cw: list, cache: dict) -> tuple:
    """Catalog words vs query runs for a Hangul query: similarities stay oriented query -> catalog
    (a Hangul word is only compared through its skeleton), so each catalog word takes its best
    query run as a."""
    runs = _runs(qw)
    best, cov = [], set()
    for b in cw:
        sb, span = 0.0, None
        for i, j, a in runs:
            sim = _word_sim(a, b, cache)[0]
            if sim > sb:
                sb, span = sim, (i, j)
        best.append(sb)
        if sb >= W_HIT:
            cov.update(range(*span))
    n = max(len(cw), 1)
    return sum(best) / n, sum(b >= W_HIT for b in best) / n, len(cov) / max(len(qw), 1), 0, 0

FEATURES = [
    "hangul", "key_exact", "key_lev", "key_cos", "roman_lev", "roman_cos", "skel_exact", "skel_lev",
    "skel_cos", "skelv_lev", "skelv_cos", "tok_jacc", "bigram_jacc", "prefix", "len_ratio",
    "gap_gt", "gap_igt", "gap_min", "exact_gt", "exact_igt", "logratio", "q_ton_missing", "c_gt_missing",
    "type_p", "flag_p", "flag_missing", "key_dups", "skel_dups", "c_dwt_ratio",
    "r_name", "d_name", "r_gap", "d_gap", "n_cand", "r_lev", "d_lev",
    "xl_pl", "xl_d", "xl_dpl", "xl_lx",
    "ng2_pl", "ng2_lo", "ng2_dlo", "ng2_dpl",
    "w_me_q", "w_hit_q", "w_cov_c", "w_abbr_q", "w_miss_q", "w_me_c", "w_hit_c", "w_cov_q", "w_num", "w_first",
    "w_comb", "w_dcomb", "gap_dwt",
    "ngt_lo", "ngt_dlo",
]


def _rank(values: np.ndarray) -> np.ndarray:
    """Rank 0 = smallest value; equal values share the smallest rank.  (Breaking ties by candidate
    order gave one of two same-name catalog vessels an arbitrary edge.)"""
    order = np.argsort(values, kind="stable")
    v = values[order]
    start = np.r_[True, v[1:] != v[:-1]]
    rk = np.empty(len(values), dtype=np.int64)
    rk[order] = np.maximum.accumulate(np.where(start, np.arange(len(values)), 0))
    return rk


def features(q: QueryBatch, cat: Catalog, pri: Priors, retrieved) -> tuple[np.ndarray, list[np.ndarray]]:
    rows, cands = [], []
    # level-1 table of the transliteration bigram (from the frozen tables only; Hangul rows need it)
    tng = getattr(pri, "tng", None)
    l1 = tng_level1(pri.xlit, tng) if tng and q.hangul.any() else None
    for r, (cand, sk, sr, ss, sv, gap) in enumerate(retrieved):
        qk, qr, qs, qv = q.key[r], q.roman[r], q.skel[r], q.skelv[r]
        qtok = frozenset(q.trans[r].split())
        qbi = grams(qk, 2)
        gt, igt = q.gt[r], q.igt[r]
        f, gd = [], []
        for j, c in enumerate(cand):
            ck = cat.key[c]
            cgt = cat.gt[c]
            pre = 0
            for a, b in zip(qk, ck):
                if a != b:
                    break
                pre += 1
            g1, g2 = rel_gap(gt, cgt), rel_gap(igt, cgt)
            qref = gt if np.isfinite(gt) else igt
            if q.hangul[r]:  # best per-word reading (primary / alternate) of the catalog name
                cs = mixed_codes(cat.skel_opts[c], False)
                cv = mixed_codes(cat.skelv_opts[c], True)
                s_exact = float(qs in cs)
                s_lev = max(lev_ratio(qs, x) for x in cs)
                v_lev = max(lev_ratio(qv, x) for x in cv)
            else:
                s_exact = float(qs == cat.skel[c])
                s_lev = lev_ratio(qs, cat.skel[c])
                v_lev = lev_ratio(qv, cat.skelv[c])
            f.append([
                float(q.hangul[r]), float(qk == ck), lev_ratio(qk, ck), sk[j], lev_ratio(qr, ck), sr[j],
                s_exact, s_lev, ss[j], v_lev, sv[j],
                jaccard(qtok, cat.tokens[c]), jaccard(qbi, grams(ck, 2)), pre / max(len(qk), len(ck), 1),
                min(len(qk), len(ck)) / max(len(qk), len(ck), 1),
                g1, g2, min(g1, g2), float(gt == cgt), float(igt == cgt),
                np.log(qref / cgt) if np.isfinite(qref) and cgt > 0 else 0.0,
                float(not np.isfinite(gt) and not np.isfinite(igt)), float(not cgt > 0),
                pri.prob(pri.type_tab, q.type[r], cat.type[c]), pri.prob(pri.flag_tab, q.flag[r], cat.flag[c]),
                float(q.flag[r] == ""), np.log1p(cat.key_count[ck]), np.log1p(cat.skel_count[cat.skel[c]]),
                cat.dwt[c] / cgt if cgt > 0 and np.isfinite(cat.dwt[c]) else -1.0,
            ])
            gd.append(min(rel_gap(gt, cat.dwt[c]), rel_gap(igt, cat.dwt[c])))
        f = np.asarray(f, dtype=np.float32)
        name = np.max(f[:, [3, 5, 8, 10]], axis=1)
        levs = np.max(f[:, [2, 4, 7, 9]], axis=1)
        gm = f[:, 17]
        rel = np.column_stack([
            _rank(-name), name.max() - name, _rank(gm), gm - gm.min(),
            np.full(len(f), len(f)), _rank(-levs), levs.max() - levs,
        ]).astype(np.float32)
        tb = _tng_block(q, cat, pri, r, cand, l1)
        rows.append(np.hstack([f, rel, _xlit_block(q, cat, pri, r, cand), tb[:, :4],
                               _word_block(q, cat, r, cand), np.asarray(gd, dtype=np.float32)[:, None],
                               _tngt_block(q.clean[r] if q.hangul[r] else "", cat, pri, cand, l1)]))
        cands.append(cand)
    return rows, cands


def _xlit_block(q: QueryBatch, cat: Catalog, pri: Priors, r: int, cand) -> np.ndarray:
    """Transliteration features (NaN for Latin rows).

    ll  = Viterbi log P(x, y) of query jamo x and candidate letters y under the joint model;
    llr = ll - log q(y), q = letter unigram background (~ log P(x | y) up to a constant).
    Columns: ll per unit, best-in-record llr minus llr, best-in-record ll/unit minus ll/unit,
    llr per jamo.
    """
    out = np.full((len(cand), 4), np.nan, dtype=np.float32)
    xl = getattr(pri, "xlit", None)
    if not q.hangul[r] or not xl:
        return out
    qc = q.clean[r]
    x = _jamo(qc)
    if not x:
        return out
    xs = xlit_prepare(xl, x)
    ly, ly_unk = xl["ly"], xl["ly_unk"]
    qlat = {t for t in qc.split() if t.isalpha() and not is_hangul(t)}
    ll = np.empty(len(cand))
    pl = np.empty(len(cand))
    llr = np.empty(len(cand))
    for j, c in enumerate(cand):
        y = "".join(_letters(t) for t in cat.clean[c].split() if t not in qlat)
        ll[j] = xlit_viterbi(xl, xs, y)
        pl[j] = ll[j] / (len(x) + len(y))
        llr[j] = ll[j] - sum(ly.get(ch, ly_unk) for ch in y)
    out[:, 0] = pl
    out[:, 1] = llr.max() - llr
    out[:, 2] = pl.max() - pl
    out[:, 3] = llr / len(x)
    return out


def _tng_block(q: QueryBatch, cat: Catalog, pri: Priors, r: int, cand, l1) -> np.ndarray:
    """Joint-bigram transliteration features (NaN for Latin rows).

    J  = Viterbi log P(x, y) of query jamo x and candidate letters y under the bigram;
    lo = J - log q(x) - log q(y), q = jamo / letter unigram backgrounds (log-odds vs unrelated names).
    Columns: J per unit, lo per jamo, best-in-record lo minus lo, best-in-record J/unit minus J/unit.
    """
    out = np.full((len(cand), 4), np.nan, dtype=np.float32)
    xl, tng = getattr(pri, "xlit", None), getattr(pri, "tng", None)
    if not q.hangul[r] or not xl or not tng:
        return out
    qc = q.clean[r]
    x = _jamo(qc)
    if not x:
        return out
    lx, lx_unk, ly, ly_unk = xl["lx"], xl["lx_unk"], xl["ly"], xl["ly_unk"]
    qlat = {t for t in qc.split() if t.isalpha() and not is_hangul(t)}
    ys = ["".join(_letters(t) for t in cat.clean[c].split() if t not in qlat) for c in cand]
    uniq = sorted(set(ys))                     # each distinct letter string is scored once
    pos = {y: k for k, y in enumerate(uniq)}
    ix = np.array([pos[y] for y in ys])
    J = tng_viterbi(tng, l1, x, uniq)[ix]
    bx = sum(lx.get(ch, lx_unk) for ch in x)
    lo = J - np.array([sum(ly.get(ch, ly_unk) for ch in y) for y in uniq])[ix] - bx
    pl = J / (len(x) + np.array([len(y) for y in ys], float))
    out[:, 0] = pl
    out[:, 1] = lo / len(x)
    out[:, 2] = lo.max() - lo
    out[:, 3] = pl.max() - pl
    return out


TYPO_DEL = -8.0     # log cost of one letter lost from the Latin name before it was written in Hangul


def _tngt_block(qc: str, cat: Catalog, pri: Priors, cand, l1) -> np.ndarray:
    """Typo-tolerant joint-bigram features (NaN for Latin rows; qc = cleaned Hangul name or "").

    Same log-odds as _tng_block, but the Viterbi may skip letters of the candidate name
    (tng_viterbi c_del): a Hangul name written from a misspelled Latin name (SOUHTPORT, PINEER)
    still lines up with the catalog spelling.  Columns: lo per jamo, best-in-record lo minus lo."""
    out = np.full((len(cand), 2), np.nan, dtype=np.float32)
    xl, tng = getattr(pri, "xlit", None), getattr(pri, "tng", None)
    if not qc or not xl or not tng:
        return out
    x = _jamo(qc)
    if not x:
        return out
    lx, lx_unk, ly, ly_unk = xl["lx"], xl["lx_unk"], xl["ly"], xl["ly_unk"]
    qlat = {t for t in qc.split() if t.isalpha() and not is_hangul(t)}
    ys = ["".join(_letters(t) for t in cat.clean[c].split() if t not in qlat) for c in cand]
    uniq = sorted(set(ys))
    pos = {y: k for k, y in enumerate(uniq)}
    ix = np.array([pos[y] for y in ys])
    J = tng_viterbi(tng, l1, x, uniq, c_del=TYPO_DEL)[ix]
    lo = J - np.array([sum(ly.get(ch, ly_unk) for ch in y) for y in uniq])[ix] - sum(lx.get(ch, lx_unk) for ch in x)
    out[:, 0] = lo / len(x)
    out[:, 1] = lo.max() - lo
    return out


EXTRA_NCOLS = 2          # columns appended after gap_dwt (exp3 patch_feat.py adds them to cached features)
extra_cols = _tngt_block
