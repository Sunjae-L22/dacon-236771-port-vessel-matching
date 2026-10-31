# author: argon
"""Korean <-> English vessel name transliteration and phonetic matcher (KoMatcher v2).

Designed for the DACON Port Vessel Matching Competition (236771).
Key upgrades in v2:
1. Zero-leakage universal dictionary: only general English dictionary loanwords &
   26-letter alphabet readings (all train proper nouns & ship-specific regexes removed).
2. Algorithmic Korean alphabet-initial parser (`케이엠티씨` -> `kmtc`, `에스티아이` -> `sti`,
   `지하이` -> `zyhi`) + DP syllable-to-token subword alignment in `_fit_train_pairs`
   so compound Korean names without spaces (`모닝쿠리어` <-> `Morning Courier`,
   `동방자이언트제2호` <-> `Dongbang Giant No. 2`) are segmented and learned per fold.
3. Universal Korean loanword phonology (`v->b`, `f->p`, `z->j`, Pinyin `zh->j`,
   `x(i/u)->sy`, `shou->seou`, Korean final `ㅅ/ㅆ->t`, geminate vs vowel-separated
   consonant skeleton, dual rhotic/non-rhotic projection `cat_en2ko` & `cat_en2ko_r`).
4. Corruption-resilient retrieval & 5 new features (`ko_tail_sim`, `ko_partial_sim`,
   `ko_prefix_exact`, `ko_token_cov`, `ko_jamo_sim`) for truncated, dropped-word,
   and typo-corrupted queries, plus class-level catalog index caching & precomputed
   token phonetic projections for zero regex calls inside the candidate scoring loop.
"""
from __future__ import annotations

import bisect
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from functools import lru_cache
import re
import unicodedata

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from rapidfuzz import fuzz, process
    from rapidfuzz.distance import JaroWinkler, Levenshtein
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


# ---------------------------------------------------------------------------
# 1. Korean / English structural & number normalization
# ---------------------------------------------------------------------------

_ROMAN_MAP = {
    "VIII": "8", "VII": "7", "III": "3", "XII": "12", "XI": "11",
    "IX": "9", "IV": "4", "VI": "6", "II": "2", "X": "10", "V": "5", "I": "1",
}
_KO_NUM_WORDS = [
    ("일레븐", "11"), ("트웰브", "12"), ("세븐", "7"), ("에이트", "8"),
    ("파이브", "5"), ("식스", "6"), ("나인", "9"), ("쓰리", "3"),
    ("스리", "3"), ("원", "1"), ("투", "2"), ("포", "4"), ("텐", "10"),
]

_EN_LETTER_TO_KO_LAT = {
    "A": "ei", "B": "bi", "C": "si", "D": "di", "E": "i", "F": "ef",
    "G": "ji", "H": "eichi", "I": "ai", "J": "jei", "K": "kei", "L": "el",
    "M": "em", "N": "en", "O": "o", "P": "pi", "Q": "kyu", "R": "al",
    "S": "es", "T": "ti", "U": "yu", "V": "vi", "W": "deoblyu", "X": "eks",
    "Y": "wai", "Z": "ji",
}

# 26 English alphabet letter names in Korean (universal, zero train proper nouns)
_KO_ALPHABET_SYLLABLES = [
    ("더블유", "w"), ("에이치", "h"), ("제트", "z"), ("에프", "f"),
    ("에스", "s"), ("에쓰", "s"), ("엑스", "x"), ("에이", "a"),
    ("아이", "i"), ("제이", "j"), ("케이", "k"), ("브이", "v"),
    ("와이", "y"), ("앤드", "n"), ("비", "b"), ("씨", "c"),
    ("시", "c"), ("디", "d"), ("이", "e"), ("지", "g"),
    ("엘", "l"), ("엠", "m"), ("엔", "n"), ("오", "o"),
    ("피", "p"), ("큐", "q"), ("알", "r"), ("티", "t"),
    ("유", "u"), ("앤", "n"),
]

# Universal English dictionary loanwords ONLY (no shipping line or vessel proper nouns from train)
_UNIVERSAL_KO_SUBWORDS = [
    ("파이오니어", "pioneer"), ("파이오니아", "pioneer"),
    ("익스프레스", "express"), ("익스플로러", "explorer"),
    ("엔터프라이즈", "enterprise"), ("프로스페리티", "prosperity"), ("프라스페리티", "prosperity"),
    ("디스커버리", "discovery"), ("인피니티", "infinity"), ("크리스탈", "crystal"),
    ("다이아몬드", "diamond"), ("트레이더", "trader"), ("트랜스포머", "transformer"),
    ("리플렉션", "reflection"), ("챔피언", "champion"), ("글로리아", "gloria"),
    ("글로벌", "global"), ("글로리", "glory"), ("보이저", "voyager"), ("보이져", "voyager"),
    ("갤럭시", "galaxy"), ("프린세스", "princess"),
    ("스피리트", "spirit"), ("스피릿", "spirit"), ("스프릿", "spirit"),
    ("선샤인", "sunshine"), ("선라이즈", "sunrise"), ("하모니", "harmony"),
    ("아틀란틱", "atlantic"), ("퍼시픽", "pacific"), ("오리엔탈", "oriental"),
    ("유니버설", "universal"), ("얼라이언스", "alliance"), ("릴라이언스", "reliance"),
    ("아일랜드", "island"), ("하이웨이", "highway"), ("스피드웨이", "speedway"),
    ("프리웨이", "freeway"), ("에메랄드", "emerald"), ("사파이어", "sapphire"),
    ("카멜리아", "camellia"), ("아젤리아", "azalea"), ("프리지아", "freesia"),
    ("아카시아", "acacia"), ("에델바이스", "edelweiss"),
    ("크라운", "crown"), ("브리지", "bridge"), ("브릿지", "bridge"),
    ("레인보우", "rainbow"), ("호라이즌", "horizon"), ("마리너", "mariner"),
    ("마리나", "marina"), ("마린", "marine"), ("자이언트", "giant"),
    ("드래곤", "dragon"), ("드라곤", "dragon"), ("유니크", "unique"),
    ("슈피리어", "superior"), ("슈프림", "supreme"), ("커리지", "courage"),
    ("트러스트", "trust"), ("빅토리", "victory"), ("석세스", "success"),
    ("포춘", "fortune"), ("포천", "fortune"), ("포르테", "forte"),
    ("브릴리언트", "brilliant"), ("브라이트", "bright"), ("샤이닝", "shining"),
    ("스카이", "sky"), ("스타즈", "stars"), ("스타", "star"),
    ("스마일", "smile"), ("스키퍼", "skipper"), ("스텔라", "stella"),
    ("스콜피오", "scorpio"), ("스프링", "spring"), ("서머", "summer"), ("윈터", "winter"),
    ("골든", "golden"), ("골드", "gold"), ("실버", "silver"), ("코퍼", "copper"),
    ("블루", "blue"), ("그린", "green"), ("레드", "red"), ("화이트", "white"), ("블랙", "black"),
    ("오션", "ocean"), ("모닝", "morning"), ("써니", "sunny"), ("서니", "sunny"),
    ("해피", "happy"), ("럭키", "lucky"), ("피스", "peace"), ("피이스", "peace"),
    ("드림", "dream"), ("호프", "hope"), ("조이", "joy"), ("비전", "vision"),
    ("그레이스", "grace"), ("블레스", "bless"), ("페이버", "favor"), ("아너", "honour"), ("오너", "honour"),
    ("리더", "leader"), ("캡틴", "captain"), ("마스터", "master"), ("마스타", "master"),
    ("치프", "chief"), ("퀸", "queen"), ("킹", "king"), ("레이디", "lady"),
    ("노블", "noble"), ("로얄", "royal"), ("프리티", "pretty"),
    ("그레이트", "great"), ("그랜드", "grand"), ("슈퍼", "super"), ("메가", "mega"),
    ("퍼스트", "first"), ("유나이티드", "united"), ("에이스", "ace"),
    ("케이프", "cape"), ("베이", "bay"), ("하버", "harbour"), ("레이크", "lake"),
    ("포트", "port"), ("로드", "road"), ("웨이브", "wave"), ("윈드", "wind"),
    ("노스", "north"), ("노드", "nord"), ("사우스", "south"), ("서던", "southern"),
    ("웨스트", "west"), ("이스트", "east"), ("이스턴", "eastern"), ("웨스턴", "western"),
    ("아시안", "asian"), ("코리아", "korea"), ("차이나", "china"),
    ("비너스", "venus"), ("머큐리", "mercury"), ("주피터", "jupiter"), ("넵튠", "neptune"),
    ("오로라", "aurora"), ("코스모스", "cosmos"), ("노바", "nova"), ("솔라", "solar"), ("루나", "luna"), ("폴라", "polar"),
    ("이글", "eagle"), ("팔콘", "falcon"), ("타이거", "tiger"), ("라이온", "lion"),
    ("울프", "wolf"), ("퓨마", "puma"), ("콘도르", "condor"), ("도브", "dove"),
    ("돌핀", "dolphin"), ("벨루가", "beluga"), ("오팔", "opal"), ("제이드", "jade"),
    ("루비", "ruby"), ("코랄", "coral"), ("토파즈", "topaz"),
    ("가스", "gas"), ("에너지", "energy"), ("에코", "eco"), ("바이오", "bio"),
    ("바우", "bow"), ("프런트", "front"), ("애로우", "arrow"), ("쿠리어", "courier"),
]

_CHO = ["g", "kk", "n", "d", "tt", "l", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"]
_JUNG = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i"]
# Korean final ㅅ/ㅆ (indices 19, 20) are pronounced [t] (e.g. 스피릿 -> spilit = Spirit)
_JONG = ["", "k", "k", "k", "n", "n", "n", "t", "l", "l", "l", "l", "l", "l", "l", "l", "m", "p", "p", "t", "t", "ng", "t", "t", "k", "t", "p", "t"]

_CHO_JAMO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG_JAMO = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG_JAMO = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]


@lru_cache(maxsize=65536)
def decompose_jamo(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3:
            o -= 0xAC00
            out.append(_CHO_JAMO[o // 588])
            out.append(_JUNG_JAMO[(o % 588) // 28])
            jong = _JONG_JAMO[o % 28]
            if jong:
                out.append(jong)
        elif ch.isalnum():
            out.append(ch.lower())
    return "".join(out)


@lru_cache(maxsize=65536)
def romanize_hangul(s: str, drop_eu: bool = False) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3:
            o -= 0xAC00
            cho = _CHO[o // 588]
            jung = _JUNG[(o % 588) // 28]
            jong = _JONG[o % 28]
            if drop_eu and jung == "eu" and cho in ("s", "ss", "k", "t", "p", "b", "g", "d", "j", "ch", "l") and not jong:
                out.append(cho)
            else:
                out.append(cho + jung + jong)
        else:
            out.append(ch.lower())
    return "".join(out)


def parse_ko_letter_acronym(tok: str) -> tuple[str, str] | None:
    """If `tok` starts with 2+ Korean alphabet letter names (e.g. 케이엠티씨 -> kmtc),
    return (latin_letters, remaining_hangul_suffix)."""
    i = 0
    n = len(tok)
    letters = []
    syllables_used = 0
    while i < n:
        matched = False
        for k_let, e_let in _KO_ALPHABET_SYLLABLES:
            if tok.startswith(k_let, i):
                letters.append(e_let)
                i += len(k_let)
                syllables_used += len(k_let)
                matched = True
                break
        if not matched:
            break
    if len(letters) >= 2 and (i == n or syllables_used >= 3):
        return "".join(letters), tok[i:]
    return None


def extract_digits_and_clean_ko(ko_name: str) -> tuple[str, str]:
    """Return (cleaned_ko_body, canonical_digits_str)."""
    if not isinstance(ko_name, str):
        return "", ""
    s = unicodedata.normalize("NFKC", ko_name).strip()
    s = re.sub(r"^\s*(?:M/V|MV|M\.V\.|MT|M/T|S/S)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(VIII|VII|III|XII|XI|IX|IV|VI|II|X|V|I)\b", lambda m: _ROMAN_MAP[m.group(1)], s)
    found_digits = []

    def _SaveJe(m):
        found_digits.append(m.group(1))
        return " "
    s = re.sub(r"제\s*(\d+)\s*호?", _SaveJe, s)

    s = re.sub(r"(\d+)\s*호(?=\s|$)", r"\1 ", s)
    tokens = s.split()
    cleaned_tokens = []
    for idx, tok in enumerate(tokens):
        if tok.endswith("호") and len(tok) >= 3:
            tok = tok[:-1]
        elif tok.endswith("호") and idx == len(tokens) - 1 and len(tokens) >= 2:
            tok = tok[:-1]
        cleaned_tokens.append(tok)
    s = " ".join(cleaned_tokens)

    for kw, num in _KO_NUM_WORDS:
        if s.endswith(" " + kw):
            found_digits.append(num)
            s = s[: -len(kw)].strip()

    for d in re.findall(r"\d+", s):
        if d not in found_digits:
            found_digits.append(d)
    s_no_digits = re.sub(r"\d+", " ", s)
    s_no_digits = " ".join(s_no_digits.split())
    return s_no_digits, "".join(found_digits)


def extract_digits_and_clean_en(en_name: str, join_hyphen: bool = False) -> tuple[str, str]:
    """Return (cleaned_en_body, canonical_digits_str)."""
    if not isinstance(en_name, str):
        return "", ""
    s = unicodedata.normalize("NFKC", en_name).strip()
    s = re.sub(r"^\s*(?:M/V|MV|M\.V\.|MT|M/T|S/S)\s+", "", s, flags=re.IGNORECASE)
    found_digits = []

    def _SaveNo(m):
        found_digits.append(m.group(1))
        return " "
    s = re.sub(r"\b[Nn][Oo]\.?\s*(\d+)\b", _SaveNo, s)

    m_rom = re.search(r"\b(VIII|VII|III|XII|XI|IX|IV|VI|II|X|V)\s*$", s)
    if m_rom:
        found_digits.append(_ROMAN_MAP[m_rom.group(1)])
        s = s[: m_rom.start()].strip()

    for d in re.findall(r"\d+", s):
        if d not in found_digits:
            found_digits.append(d)
    s_no_digits = re.sub(r"\d+", " ", s)
    if join_hyphen:
        s_no_digits = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", "", s_no_digits)
    s_no_digits = re.sub(r"[^A-Za-z]+", " ", s_no_digits).strip()
    return s_no_digits, "".join(found_digits)


def expand_en_acronyms(en_body: str) -> str:
    """Expand uppercase acronym tokens into Korean-style phonetic Latin."""
    toks = []
    for t in en_body.split():
        if t.isupper() and t.isalpha() and 2 <= len(t) <= 4 and not re.search(r"[AEIOU][^AEIOU]+[AEIOU]", t):
            toks.append("".join(_EN_LETTER_TO_KO_LAT.get(c, c.lower()) for c in t))
        else:
            toks.append(t.lower())
    return " ".join(toks)


@lru_cache(maxsize=131072)
def en_to_ko_phonetic_lat(en_body: str, keep_precons_r: bool = False) -> str:
    """Project an English ship name into Korean loanword phonetic Latin using pure phonological rules."""
    s = en_body.lower()
    s = re.sub(r"tion\b", "syeon", s)
    s = re.sub(r"sion\b", "jeon", s)
    s = re.sub(r"liance\b", "laieonseu", s)
    s = re.sub(r"ture\b", "cheo", s)
    s = re.sub(r"shou", "seou", s)
    s = re.sub(r"zh", "j", s)
    s = re.sub(r"ph", "p", s)
    s = re.sub(r"tch|ch", "ch", s)
    s = re.sub(r"sh(?=[iu])", "sy", s)
    s = re.sub(r"sh", "s", s)
    s = re.sub(r"th", "s", s)
    s = re.sub(r"ck|qu", "k", s)
    s = re.sub(r"x(?=[iu])", "sy", s)
    s = re.sub(r"q(?=[iu])", "ch", s)
    s = re.sub(r"c(?=[eiy])", "s", s)
    s = re.sub(r"c", "k", s)
    s = re.sub(r"g(?=[eiy])", "j", s)
    s = re.sub(r"v", "b", s)
    s = re.sub(r"f", "p", s)
    s = re.sub(r"z", "j", s)
    s = re.sub(r"eer\b|ier\b", "ieo", s)
    s = re.sub(r"(?:er|or|ar|ur|our)\b", "eo", s)
    if keep_precons_r:
        s = re.sub(r"r(?![aeiouy])", "l", s)
    else:
        s = re.sub(r"ar(?![aeiouy])", "a", s)
        s = re.sub(r"or(?![aeiouy])", "o", s)
        s = re.sub(r"(?:ir|ur|er)(?![aeiouy])", "eo", s)
    s = re.sub(r"ee|ea", "i", s)
    s = re.sub(r"oo|ou", "u", s)
    s = re.sub(r"ai|ay", "ei", s)
    s = re.sub(r"oa|ow", "o", s)
    s = re.sub(r"au|aw", "o", s)
    s = re.sub(r"r", "l", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


@lru_cache(maxsize=131072)
def canonical_cons_key(s: str, keep_h: bool = True, soft_g: bool = False) -> str:
    """Consonant skeleton preserving initial/medial H and geminate vs syllable-separated consonants."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9]", "", s)
    s = re.sub(r"tion|sion", "JN", s)
    s = re.sub(r"zh", "J", s)
    s = re.sub(r"tch|ch", "J", s)
    s = re.sub(r"sh", "S", s)
    s = re.sub(r"ph", "P", s)
    s = re.sub(r"th", "T", s)
    s = re.sub(r"ck|qu|q", "K", s)
    s = re.sub(r"ng", "N", s)
    s = re.sub(r"x(?=[iu])", "S", s)
    s = re.sub(r"x", "KS", s)
    s = re.sub(r"c(?=[eiy])", "S", s)
    s = re.sub(r"c", "K", s)
    if soft_g:
        s = re.sub(r"g(?=[eiy])", "J", s)
    s = re.sub(r"r(?![aeiouy])", "", s)
    cons_map = {
        "b": "P", "p": "P", "f": "P", "v": "P",
        "d": "T", "t": "T",
        "g": "K", "k": "K",
        "s": "S", "z": "J",
        "j": "J",
        "l": "L", "r": "L",
        "m": "M", "n": "N",
        "h": "H" if keep_h else "",
        "w": "", "y": "",
    }
    s = s.translate(str.maketrans(cons_map))
    # Collapse only truly adjacent identical consonants (geminates like 'll' in Alliance / 얼라이언스)
    # BEFORE stripping vowels, so 'Reliance' (L-e-L) keeps two L's while 'Alliance' (a-LL) has one L!
    s = re.sub(r"([PTKSJLMNH])\1+", r"\1", s)
    s = re.sub(r"[aeiou]", "", s)
    return s


@lru_cache(maxsize=131072)
def canonical_phon_key(s: str, soft_g: bool = False) -> str:
    """Coarse vowel + consonant phonetic key for cross-script matching."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9]", "", s)
    s = re.sub(r"tion|sion", "jon", s)
    s = re.sub(r"shou", "so", s)
    s = re.sub(r"xiu", "su", s)
    s = re.sub(r"zh", "J", s)
    s = re.sub(r"tch|ch", "J", s)
    s = re.sub(r"sh", "S", s)
    s = re.sub(r"ph", "P", s)
    s = re.sub(r"ck|qu|q", "K", s)
    s = re.sub(r"ng", "N", s)
    s = re.sub(r"x(?=[iu])", "S", s)
    s = re.sub(r"c(?=[eiy])", "S", s)
    s = re.sub(r"c", "K", s)
    if soft_g:
        s = re.sub(r"g(?=[eiy])", "J", s)
    s = re.sub(r"r(?![aeiouy])", "", s)
    s = re.sub(r"eo|eu|ou|oo|wo|u", "O", s)
    s = re.sub(r"ae|ai|ay|ei|ey|ye|we|e", "E", s)
    s = re.sub(r"ya|wa|a", "A", s)
    s = re.sub(r"wi|ui|yi|i", "I", s)
    s = re.sub(r"yo|oe|o", "O", s)
    mapping = {
        "b": "P", "p": "P", "f": "P", "v": "P",
        "d": "T", "t": "T",
        "g": "K", "k": "K",
        "s": "S", "z": "J", "x": "S",
        "j": "J",
        "l": "L", "r": "L",
        "m": "M", "n": "N",
        "h": "H", "w": "", "y": "",
    }
    s = s.translate(str.maketrans(mapping))
    return re.sub(r"(.)\1+", r"\1", s)


@lru_cache(maxsize=262144)
def _sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if HAS_RAPIDFUZZ:
        return fuzz.ratio(a, b) / 100.0
    return SequenceMatcher(None, a, b).ratio()


def _psim(a: str, b: str) -> float:
    """Partial / prefix similarity for truncated or single-word queries."""
    if not a or not b:
        return 0.0
    if a == b or b.startswith(a) or b.endswith(a):
        return 1.0
    la, lb = len(a), len(b)
    if la <= lb:
        pref = _sim(a, b[:la])
        suff = _sim(a, b[-la:])
        if HAS_RAPIDFUZZ:
            return max(pref, suff, fuzz.partial_ratio(a, b) / 100.0)
        return max(pref, suff, _sim(a, b))
    else:
        if HAS_RAPIDFUZZ:
            return fuzz.partial_ratio(a, b) / 100.0
        return _sim(a[:lb], b)


def _lcp_ratio(q: str, c: str) -> float:
    """Fraction of query `q` matched as an exact prefix of `c`."""
    if not q or not c:
        return 0.0
    limit = min(len(q), len(c))
    k = 0
    while k < limit and q[k] == c[k]:
        k += 1
    return k / len(q)


@lru_cache(maxsize=262144)
def _jw(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if HAS_RAPIDFUZZ:
        return float(JaroWinkler.normalized_similarity(a, b))
    return SequenceMatcher(None, a, b).ratio()


def _ngrams(s: str, n: int = 2) -> set[str]:
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _pair_phon_sim(kt: str, et: str) -> float:
    """Phonetic compatibility score between a Hangul slice `kt` and English token `et`."""
    if not kt or not et:
        return 0.0
    kr = romanize_hangul(kt, drop_eu=True)
    kr_full = romanize_hangul(kt, drop_eu=False)
    et_acr = expand_en_acronyms(et.upper()) if len(et) <= 4 else et.lower()
    e2k = en_to_ko_phonetic_lat(et_acr, keep_precons_r=False)
    e2k_r = en_to_ko_phonetic_lat(et_acr, keep_precons_r=True)
    s1 = max(_sim(kr, e2k), _sim(kr_full, e2k), _sim(kr, e2k_r), _sim(kr, et.lower()))
    s2 = _sim(canonical_cons_key(kr, keep_h=True), canonical_cons_key(et_acr, keep_h=True, soft_g=True))
    s3 = _sim(canonical_phon_key(kr_full), canonical_phon_key(et_acr, soft_g=True))
    return max(s1, 0.55 * s2 + 0.45 * s3)


# ---------------------------------------------------------------------------
# 2. KoMatcher Main Class
# ---------------------------------------------------------------------------

class KoMatcher:
    """Fold-safe Korean-to-English vessel name matcher and feature generator."""

    # Keep ko_trans_sim at [0] and ko_combined_score at [-1] so opus_lib.py works seamlessly
    FEATURE_NAMES = (
        "ko_trans_sim",
        "ko_trans_jw",
        "ko_en2ko_sim",
        "ko_cons_sim",
        "ko_cons_acr_sim",
        "ko_phon_sim",
        "ko_no_eu_sim",
        "ko_prefix_sim",
        "ko_digit_match",
        "ko_tail_sim",
        "ko_partial_sim",
        "ko_prefix_exact",
        "ko_token_cov",
        "ko_jamo_sim",
        "ko_combined_score",
    )

    _CATALOG_CACHE: dict[tuple, dict] = {}

    def __init__(self, catalog, train_pairs=None):
        if pd is not None and isinstance(catalog, pd.DataFrame):
            cat_names = catalog["vessel_name"].fillna("").astype(str).tolist()
            cat_ids = catalog["vessel_id"].astype(str).tolist() if "vessel_id" in catalog.columns else [str(i) for i in range(len(cat_names))]
        else:
            cat_names = [str(r.get("vessel_name") or "") for r in catalog]
            cat_ids = [str(r.get("vessel_id", i)) for i, r in enumerate(catalog)]

        self.n_cat = len(cat_names)
        self.cat_names_raw = cat_names
        self.id_to_idx = {vid: i for i, vid in enumerate(cat_ids)}

        cache_key = (self.n_cat, cat_names[0] if cat_names else "", cat_names[-1] if cat_names else "")
        cached = self._CATALOG_CACHE.get(cache_key)
        if cached is None:
            cached = self._build_catalog_index(cat_names)
            self._CATALOG_CACHE[cache_key] = cached

        self.cat_en_clean = cached["cat_en_clean"]
        self.cat_en_tokens = cached["cat_en_tokens"]
        self.cat_tok_e2k = cached["cat_tok_e2k"]
        self.cat_digits = cached["cat_digits"]
        self.cat_en_compact = cached["cat_en_compact"]
        self.cat_en2ko = cached["cat_en2ko"]
        self.cat_en2ko_r = cached["cat_en2ko_r"]
        self.cat_cons = cached["cat_cons"]
        self.cat_cons_acr = cached["cat_cons_acr"]
        self.cat_phon = cached["cat_phon"]
        self.exact_compact = cached["exact_compact"]
        self.exact_cons = cached["exact_cons"]
        self.bigram_postings = cached["bigram_postings"]
        self.cons_bigram_postings = cached["cons_bigram_postings"]
        self.sorted_compact = cached["sorted_compact"]
        self.sorted_compact_keys = cached["sorted_compact_keys"]
        self.sorted_en2ko = cached["sorted_en2ko"]
        self.sorted_en2ko_keys = cached["sorted_en2ko_keys"]
        self.sorted_cons = cached["sorted_cons"]
        self.sorted_cons_keys = cached["sorted_cons_keys"]
        self.first_tok_postings = cached["first_tok_postings"]

        self.exact_word_dict: dict[str, str] = {}
        self.learned_subwords: list[tuple[str, str]] = []
        self._fit_train_pairs(train_pairs)

    @staticmethod
    def _build_catalog_index(cat_names: list[str]) -> dict:
        cat_en_clean = []
        cat_en_tokens = []
        cat_tok_e2k = []
        cat_digits = []
        cat_en_compact = []
        cat_en2ko = []
        cat_en2ko_r = []
        cat_cons = []
        cat_cons_acr = []
        cat_phon = []

        exact_compact = defaultdict(list)
        exact_cons = defaultdict(list)
        bigram_postings = defaultdict(list)
        cons_bigram_postings = defaultdict(list)
        first_tok_postings = defaultdict(list)

        for i, raw in enumerate(cat_names):
            body, d = extract_digits_and_clean_en(raw, join_hyphen=True)
            compact = re.sub(r"[^a-z0-9]", "", body.lower())
            toks = body.lower().split()
            tok_proj = [
                (
                    ct,
                    en_to_ko_phonetic_lat(ct, False),
                    en_to_ko_phonetic_lat(ct, True),
                    canonical_cons_key(ct, soft_g=True),
                )
                for ct in toks
            ]
            acr_body = expand_en_acronyms(body)
            en2ko = en_to_ko_phonetic_lat(acr_body, keep_precons_r=False)
            en2ko_r = en_to_ko_phonetic_lat(acr_body, keep_precons_r=True)
            cons = canonical_cons_key(body, keep_h=True, soft_g=True)
            cons_acr = canonical_cons_key(acr_body, keep_h=True, soft_g=True)
            phon = canonical_phon_key(acr_body, soft_g=True)

            cat_en_clean.append(body.lower())
            cat_en_tokens.append(toks)
            cat_tok_e2k.append(tok_proj)
            cat_digits.append(d)
            cat_en_compact.append(compact)
            cat_en2ko.append(en2ko)
            cat_en2ko_r.append(en2ko_r)
            cat_cons.append(cons)
            cat_cons_acr.append(cons_acr)
            cat_phon.append(phon)

            if toks:
                first_tok_postings[toks[0]].append(i)
            if compact:
                exact_compact[compact].append(i)
                for bg in _ngrams(compact, 3 if len(compact) >= 4 else 2):
                    bigram_postings[bg].append(i)
            if cons:
                exact_cons[cons].append(i)
                for bg in _ngrams(cons, 2):
                    cons_bigram_postings[bg].append(i)
            if cons_acr and cons_acr != cons:
                exact_cons[cons_acr].append(i)
                for bg in _ngrams(cons_acr, 2):
                    cons_bigram_postings[bg].append(i)

        sorted_compact = sorted((c, i) for i, c in enumerate(cat_en_compact) if c)
        sorted_en2ko = sorted((c, i) for i, c in enumerate(cat_en2ko) if c)
        sorted_cons = sorted((c, i) for i, c in enumerate(cat_cons_acr) if c)

        return {
            "cat_en_clean": cat_en_clean,
            "cat_en_tokens": cat_en_tokens,
            "cat_tok_e2k": cat_tok_e2k,
            "cat_digits": cat_digits,
            "cat_en_compact": cat_en_compact,
            "cat_en2ko": cat_en2ko,
            "cat_en2ko_r": cat_en2ko_r,
            "cat_cons": cat_cons,
            "cat_cons_acr": cat_cons_acr,
            "cat_phon": cat_phon,
            "exact_compact": exact_compact,
            "exact_cons": exact_cons,
            "bigram_postings": bigram_postings,
            "cons_bigram_postings": cons_bigram_postings,
            "first_tok_postings": first_tok_postings,
            "sorted_compact": sorted_compact,
            "sorted_compact_keys": [c for c, _ in sorted_compact],
            "sorted_en2ko": sorted_en2ko,
            "sorted_en2ko_keys": [c for c, _ in sorted_en2ko],
            "sorted_cons": sorted_cons,
            "sorted_cons_keys": [c for c, _ in sorted_cons],
        }

    def _fit_train_pairs(self, train_pairs) -> None:
        """Learn token-level and DP-segmented subword Korean->English translations strictly from fold train data."""
        base_dict = dict(_UNIVERSAL_KO_SUBWORDS)
        word_dict = dict(_UNIVERSAL_KO_SUBWORDS)
        if train_pairs is not None:
            if pd is not None and isinstance(train_pairs, pd.DataFrame):
                records = train_pairs.to_dict("records")
            else:
                records = list(train_pairs)

            pair_counts = defaultdict(Counter)
            full_stem_counts = defaultdict(Counter)

            for r in records:
                ko_raw = str(r.get("vessel_name") or "")
                if not any("가" <= c <= "힣" for c in ko_raw):
                    continue
                en_raw = r.get("catalog_name")
                if not en_raw and "target_vessel_id" in r:
                    idx = self.id_to_idx.get(str(r["target_vessel_id"]))
                    if idx is not None:
                        en_raw = self.cat_names_raw[idx]
                if not en_raw:
                    continue

                ko_body, _ = extract_digits_and_clean_ko(ko_raw)
                en_body, _ = extract_digits_and_clean_en(str(en_raw), join_hyphen=True)
                ko_toks = ko_body.split()
                en_toks = en_body.lower().split()
                if not ko_toks or not en_toks:
                    continue

                ko_nospace = "".join(ko_toks)
                en_nospace = "".join(en_toks)
                if len(ko_nospace) >= 2 and len(en_nospace) >= 2:
                    full_stem_counts[ko_nospace][en_nospace] += 1

                if len(ko_toks) == len(en_toks):
                    for kt, et in zip(ko_toks, en_toks):
                        if len(kt) >= 2 and len(et) >= 1:
                            pair_counts[kt][et] += 2
                else:
                    if len(ko_toks) >= 2 and len(en_toks) >= 2:
                        if len(ko_toks[0]) >= 2 and _pair_phon_sim(ko_toks[0], en_toks[0]) >= 0.50:
                            pair_counts[ko_toks[0]][en_toks[0]] += 2
                        if len(ko_toks[-1]) >= 2 and _pair_phon_sim(ko_toks[-1], en_toks[-1]) >= 0.50:
                            pair_counts[ko_toks[-1]][en_toks[-1]] += 2
                        if len(ko_toks) == 2 and len(en_toks) == 3:
                            e01 = en_toks[0] + en_toks[1]
                            if _pair_phon_sim(ko_toks[0], e01) >= 0.50:
                                pair_counts[ko_toks[0]][e01] += 2
                            if len(ko_toks[1]) >= 1 and _pair_phon_sim(ko_toks[1], en_toks[2]) >= 0.50:
                                pair_counts[ko_toks[1]][en_toks[2]] += 2

                    L_ko = len(ko_nospace)
                    if 2 <= len(en_toks) <= 3 and len(en_toks) <= L_ko <= 12:
                        if len(en_toks) == 2:
                            best_sc = -1.0
                            best_sp = None
                            for sp in range(1, L_ko):
                                k1, k2 = ko_nospace[:sp], ko_nospace[sp:]
                                s1 = _pair_phon_sim(k1, en_toks[0])
                                s2 = _pair_phon_sim(k2, en_toks[1])
                                if s1 >= 0.52 and s2 >= 0.52 and (s1 + s2) > best_sc:
                                    best_sc = s1 + s2
                                    best_sp = (k1, k2)
                            if best_sp is not None:
                                k1, k2 = best_sp
                                if len(k1) >= 2:
                                    pair_counts[k1][en_toks[0]] += 1
                                if len(k2) >= 2:
                                    pair_counts[k2][en_toks[1]] += 1
                        elif len(en_toks) == 3:
                            best_sc = -1.0
                            best_sp = None
                            for sp1 in range(1, L_ko - 1):
                                for sp2 in range(sp1 + 1, L_ko):
                                    k1, k2, k3 = ko_nospace[:sp1], ko_nospace[sp1:sp2], ko_nospace[sp2:]
                                    s1 = _pair_phon_sim(k1, en_toks[0])
                                    s2 = _pair_phon_sim(k2, en_toks[1])
                                    s3 = _pair_phon_sim(k3, en_toks[2])
                                    if s1 >= 0.52 and s2 >= 0.52 and s3 >= 0.52 and (s1 + s2 + s3) > best_sc:
                                        best_sc = s1 + s2 + s3
                                        best_sp = (k1, k2, k3)
                            if best_sp is not None:
                                for ki, ei in zip(best_sp, en_toks):
                                    if len(ki) >= 2:
                                        pair_counts[ki][ei] += 1

            for kt, cnt in full_stem_counts.items():
                best_et, _ = cnt.most_common(1)[0]
                word_dict[kt] = best_et

            for kt, cnt in pair_counts.items():
                best_et, best_c = cnt.most_common(1)[0]
                if best_c >= 1:
                    if _pair_phon_sim(kt, best_et) >= 0.45 or best_c >= 2:
                        word_dict[kt] = best_et
                        if len(kt) >= 2:
                            base_dict[kt] = best_et

        self.exact_word_dict = word_dict
        self.learned_subwords = sorted(base_dict.items(), key=lambda x: (-len(x[0]), x[0]))

    def _translate_single_token(self, tok: str) -> list[str]:
        if tok in self.exact_word_dict:
            return [self.exact_word_dict[tok]]
        acr = parse_ko_letter_acronym(tok)
        if acr is not None:
            lat_init, rem_ko = acr
            if not rem_ko:
                return [lat_init]
            return [lat_init] + self._translate_single_token(rem_ko)

        out = []
        i = 0
        n = len(tok)
        while i < n:
            matched = False
            for k_sub, e_sub in self.learned_subwords:
                if len(k_sub) == 2 and not (i == 0 or i + 2 == n):
                    continue
                if tok.startswith(k_sub, i):
                    out.append(e_sub)
                    i += len(k_sub)
                    matched = True
                    break
            if not matched:
                out.append(romanize_hangul(tok[i], drop_eu=True))
                i += 1
        return out

    def translate_ko(self, ko_name: str) -> dict:
        """Produce multiple complementary Latin/phonetic projections of a Korean vessel name."""
        ko_body, q_digits = extract_digits_and_clean_ko(ko_name)
        ko_nospace = re.sub(r"\s+", "", ko_body)

        trans_tokens = []
        if ko_nospace in self.exact_word_dict and len(ko_body.split()) == 1:
            trans_tokens = [self.exact_word_dict[ko_nospace]]
        else:
            for tok in ko_body.split():
                trans_tokens.extend(self._translate_single_token(tok))

        trans_compact = re.sub(r"[^a-z0-9]", "", "".join(trans_tokens).lower())
        rom_full = re.sub(r"[^a-z0-9]", "", romanize_hangul(ko_nospace, drop_eu=False))
        rom_no_eu = re.sub(r"[^a-z0-9]", "", romanize_hangul(ko_nospace, drop_eu=True))

        ko_tok_list = ko_body.split()
        tok_projections = []
        for kt in ko_tok_list:
            t_tr = re.sub(r"[^a-z0-9]", "", "".join(self._translate_single_token(kt)).lower())
            t_rom = re.sub(r"[^a-z0-9]", "", romanize_hangul(kt, drop_eu=True))
            t_cons = canonical_cons_key(t_tr or t_rom, keep_h=True, soft_g=True)
            tok_projections.append((t_tr, t_rom, t_cons))

        cons_trans = canonical_cons_key(trans_compact, keep_h=True, soft_g=True)
        cons_rom = canonical_cons_key(rom_no_eu, keep_h=True, soft_g=False)
        phon_trans = canonical_phon_key(trans_compact, soft_g=True)
        phon_rom = canonical_phon_key(rom_full, soft_g=False)

        return {
            "ko_body": ko_body,
            "ko_nospace": ko_nospace,
            "digits": q_digits,
            "trans": trans_compact,
            "trans_tokens": [re.sub(r"[^a-z0-9]", "", t.lower()) for t in trans_tokens if t],
            "tok_projections": tok_projections,
            "rom_full": rom_full,
            "rom_no_eu": rom_no_eu,
            "cons_trans": cons_trans,
            "cons_rom": cons_rom,
            "phon_trans": phon_trans,
            "phon_rom": phon_rom,
        }

    def _score_from_qinfo(self, qinfo: dict, idx: int) -> dict[str, float]:
        c_compact = self.cat_en_compact[idx]
        c_tok_e2k = self.cat_tok_e2k[idx]
        c_en2ko = self.cat_en2ko[idx]
        c_en2ko_r = self.cat_en2ko_r[idx]
        c_cons = self.cat_cons[idx]
        c_cons_acr = self.cat_cons_acr[idx]
        c_phon = self.cat_phon[idx]
        c_digits = self.cat_digits[idx]

        q_trans = qinfo["trans"]
        q_rom_full = qinfo["rom_full"]
        q_rom_no_eu = qinfo["rom_no_eu"]
        q_cons_t = qinfo["cons_trans"]
        q_cons_r = qinfo["cons_rom"]
        q_phon_t = qinfo["phon_trans"]
        q_phon_r = qinfo["phon_rom"]
        q_digits = qinfo["digits"]
        tok_proj = qinfo["tok_projections"]

        trans_sim = max(_sim(q_trans, c_compact), _sim(q_trans, c_en2ko), _sim(q_trans, c_en2ko_r))
        trans_jw = max(_jw(q_trans, c_compact), _jw(q_trans, c_en2ko), _jw(q_trans, c_en2ko_r))
        en2ko_sim = max(
            _sim(q_rom_full, c_en2ko),
            _sim(q_rom_no_eu, c_en2ko),
            _sim(q_rom_full, c_en2ko_r),
            _sim(q_rom_no_eu, c_en2ko_r),
            _sim(q_rom_no_eu, c_compact),
        )
        cons_sim = max(_sim(q_cons_t, c_cons), _sim(q_cons_r, c_cons))
        cons_acr_sim = max(_sim(q_cons_t, c_cons_acr), _sim(q_cons_r, c_cons_acr))
        phon_sim = max(_sim(q_phon_t, c_phon), _sim(q_phon_r, c_phon))
        no_eu_sim = max(_sim(q_rom_no_eu, c_compact), _sim(q_rom_no_eu, c_en2ko_r))

        pref_len = min(5, len(q_trans), len(c_compact))
        prefix_sim = max(
            _sim(q_trans[:pref_len], c_compact[:pref_len]),
            _sim(q_rom_no_eu[:pref_len], c_en2ko[:pref_len]),
        ) if pref_len > 0 else 0.0

        if q_digits and c_digits:
            digit_match = 1.0 if q_digits == c_digits else -1.0
        elif q_digits or c_digits:
            digit_match = -0.85
        else:
            digit_match = 0.0

        # 10. ko_tail_sim: suffix similarity after stripping shared prefix or first token
        lcp_tc = 0
        lim_tc = min(len(q_trans), len(c_compact))
        while lcp_tc < lim_tc and q_trans[lcp_tc] == c_compact[lcp_tc]:
            lcp_tc += 1
        if len(tok_proj) >= 2 and len(c_tok_e2k) >= 2:
            qt_last, qr_last, qc_last = tok_proj[-1]
            ct_last, ce_last, ce_last_r, cc_last = c_tok_e2k[-1]
            tail_sim = max(
                _sim(qt_last, ct_last),
                _sim(qt_last, ce_last),
                _sim(qr_last, ce_last),
                _sim(qr_last, ce_last_r),
                _sim(qc_last, cc_last),
            )
            if qr_last and ct_last and (qr_last[0] in "aeiou") != (ct_last[0] in "aeiou"):
                tail_sim *= 0.78
        elif lcp_tc >= 3 and (len(q_trans) > lcp_tc or len(c_compact) > lcp_tc):
            q_tail_t = q_trans[lcp_tc:]
            c_tail_t = c_compact[lcp_tc:]
            c_tail_e2k = en_to_ko_phonetic_lat(c_tail_t, False)
            c_tail_e2k_r = en_to_ko_phonetic_lat(c_tail_t, True)
            tail_sim = max(
                _sim(q_tail_t, c_tail_t),
                _sim(q_tail_t, c_tail_e2k),
                _sim(q_tail_t, c_tail_e2k_r),
            )
            if q_tail_t and c_tail_t:
                if (q_tail_t[0] in "aeiou") != (c_tail_t[0] in "aeiou"):
                    tail_sim *= 0.78
                elif q_tail_t[0] != c_tail_t[0] and (not c_tail_e2k or q_tail_t[0] != c_tail_e2k[0]):
                    tail_sim *= 0.85
        else:
            tail_sim = max(trans_sim, en2ko_sim)

        # 11. ko_partial_sim & 12. ko_prefix_exact: handles truncated / dropped-word queries
        partial_sim = max(
            _psim(q_trans, c_compact),
            _psim(q_trans, c_en2ko),
            _psim(q_rom_no_eu, c_en2ko),
            _psim(q_rom_no_eu, c_en2ko_r),
            _psim(q_cons_t, c_cons_acr),
        )
        prefix_exact = max(
            _lcp_ratio(q_trans, c_compact),
            _lcp_ratio(q_trans, c_en2ko),
            _lcp_ratio(q_rom_no_eu, c_en2ko),
            _lcp_ratio(q_rom_no_eu, c_en2ko_r),
            _lcp_ratio(q_cons_t, c_cons_acr),
        )

        # 13. ko_token_cov: average best match of each query token against precomputed candidate tokens
        if tok_proj and c_tok_e2k:
            tok_scores = []
            for qt_tr, qt_rom, qt_cons in tok_proj:
                best_t = 0.0
                q_vow = bool(qt_rom and qt_rom[0] in "aeiou")
                for ct, ce1, ce2, cc in c_tok_e2k:
                    c_vow = bool(ct and ct[0] in "aeiou")
                    sc = max(_sim(qt_tr, ct), _sim(qt_tr, ce1), _sim(qt_rom, ce1), _sim(qt_rom, ce2), 0.9 * _sim(qt_cons, cc))
                    if len(qt_tr) >= 3 and (ct.startswith(qt_tr) or ce1.startswith(qt_rom)):
                        sc = max(sc, 0.92)
                    if q_vow != c_vow:
                        sc *= 0.85
                    if sc > best_t:
                        best_t = sc
                tok_scores.append(best_t)
            token_cov = sum(tok_scores) / len(tok_scores)
        else:
            token_cov = partial_sim

        # 14. ko_jamo_sim: resilient to Jamo/character typos
        jamo_sim = max(
            0.5 * trans_jw + 0.5 * en2ko_sim,
            0.5 * max(cons_sim, cons_acr_sim) + 0.5 * phon_sim,
        )

        # 15. ko_combined_score: unified ranker/retrieval score
        is_short_query = len(q_trans) < int(0.72 * len(c_compact)) and prefix_exact >= 0.80
        partial_bonus = 0.12 * (0.6 * partial_sim + 0.4 * prefix_exact) if is_short_query else 0.04 * prefix_exact

        combined = (
            0.25 * trans_sim
            + 0.12 * trans_jw
            + 0.16 * en2ko_sim
            + 0.12 * max(cons_sim, cons_acr_sim)
            + 0.11 * phon_sim
            + 0.11 * tail_sim
            + 0.05 * token_cov
            + 0.04 * prefix_sim
            + partial_bonus
            + 0.08 * digit_match
        )
        return {
            "ko_trans_sim": float(trans_sim),
            "ko_trans_jw": float(trans_jw),
            "ko_en2ko_sim": float(en2ko_sim),
            "ko_cons_sim": float(cons_sim),
            "ko_cons_acr_sim": float(cons_acr_sim),
            "ko_phon_sim": float(phon_sim),
            "ko_no_eu_sim": float(no_eu_sim),
            "ko_prefix_sim": float(prefix_sim),
            "ko_digit_match": float(digit_match),
            "ko_tail_sim": float(tail_sim),
            "ko_partial_sim": float(partial_sim),
            "ko_prefix_exact": float(prefix_exact),
            "ko_token_cov": float(token_cov),
            "ko_jamo_sim": float(jamo_sim),
            "ko_combined_score": float(combined),
        }

    def score(self, ko_name: str, catalog_idx: int) -> dict[str, float]:
        """Compute phonetic & transliteration similarity features for a single (ko_name, catalog_idx) pair."""
        qinfo = self.translate_ko(ko_name)
        return self._score_from_qinfo(qinfo, int(catalog_idx))

    def score_batch(self, ko_name: str, catalog_indices: list[int]):
        """Compute (len(catalog_indices), 15) feature matrix for a list of candidate indices."""
        qinfo = self.translate_ko(ko_name)
        rows = []
        for idx in catalog_indices:
            d = self._score_from_qinfo(qinfo, int(idx))
            rows.append([d[k] for k in self.FEATURE_NAMES])
        if np is not None:
            return np.asarray(rows, dtype=np.float32)
        return rows

    @staticmethod
    def _prefix_range(sorted_pairs: list[tuple[str, int]], sorted_keys: list[str], prefix: str, limit: int = 120) -> list[int]:
        if len(prefix) < 2:
            return []
        lo = bisect.bisect_left(sorted_keys, prefix)
        out = []
        while lo < len(sorted_pairs) and sorted_keys[lo].startswith(prefix) and len(out) < limit:
            out.append(sorted_pairs[lo][1])
            lo += 1
        return out

    def candidates(self, ko_name: str, k: int = 50) -> list[tuple[int, float]]:
        """Retrieve top-k catalog (index, score) candidates using name only (no tonnage),
        including prefix and first-token lookup for truncated/dropped-word queries."""
        qinfo = self.translate_ko(ko_name)
        q_trans = qinfo["trans"]
        q_rom_no_eu = qinfo["rom_no_eu"]
        q_cons_t = qinfo["cons_trans"]
        q_cons_r = qinfo["cons_rom"]

        shortlist: set[int] = set()
        for idx in self.exact_compact.get(q_trans, ()):
            shortlist.add(idx)
        for idx in self.exact_cons.get(q_cons_t, ()):
            shortlist.add(idx)
        for idx in self.exact_cons.get(q_cons_r, ()):
            shortlist.add(idx)

        # Prefix & first-token lookup for truncated / dropped-word queries
        if qinfo["trans_tokens"]:
            ft = qinfo["trans_tokens"][0]
            for idx in self.first_tok_postings.get(ft, ())[:120]:
                shortlist.add(idx)
        for pref in (q_trans[:6], q_trans[:4], q_rom_no_eu[:5]):
            if len(pref) >= 3:
                for idx in self._prefix_range(self.sorted_compact, self.sorted_compact_keys, pref, limit=80):
                    shortlist.add(idx)
                for idx in self._prefix_range(self.sorted_en2ko, self.sorted_en2ko_keys, pref, limit=80):
                    shortlist.add(idx)
        if len(q_cons_t) >= 3:
            for idx in self._prefix_range(self.sorted_cons, self.sorted_cons_keys, q_cons_t[:4], limit=80):
                shortlist.add(idx)

        votes = Counter()
        for bg in _ngrams(q_trans, 3 if len(q_trans) >= 4 else 2):
            for idx in self.bigram_postings.get(bg, ()):
                votes[idx] += 2
        for bg in _ngrams(q_cons_t, 2) | _ngrams(q_cons_r, 2):
            for idx in self.cons_bigram_postings.get(bg, ()):
                votes[idx] += 1

        for idx, _ in votes.most_common(450):
            shortlist.add(idx)

        if HAS_RAPIDFUZZ and q_trans:
            for _, _, idx in process.extract(q_trans, self.cat_en_compact, scorer=fuzz.ratio, limit=100):
                shortlist.add(int(idx))
            for _, _, idx in process.extract(q_rom_no_eu, self.cat_en2ko, scorer=fuzz.ratio, limit=80):
                shortlist.add(int(idx))
            for _, _, idx in process.extract(q_cons_t, self.cat_cons_acr, scorer=fuzz.ratio, limit=100):
                shortlist.add(int(idx))

        if not shortlist:
            shortlist.update(range(min(k, self.n_cat)))

        scored = [
            (idx, self._score_from_qinfo(qinfo, idx)["ko_combined_score"])
            for idx in shortlist
        ]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]
