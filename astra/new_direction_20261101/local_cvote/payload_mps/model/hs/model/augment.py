"""Training-time data augmentation for the ranker (Ditto-style operators, Li et al. 2020).

train.csv's name noise is narrow: a Latin name differs from the catalog by one dropped, doubled
or swapped letter, and a Hangul name is a full transliteration.  Evaluation records can also
have spacing changes, abbreviated, missing or extra words, more typos, larger tonnage errors and
missing fields.  Each training row gets AUG_COPIES perturbed copies with the same answer; their
features are computed out-of-fold like the original rows, so the ranker learns these cases too.
Only training uses this module; prediction does not change.

Every copy is drawn from a generator seeded by the row's sample_id, so the augmented set is the
same on every run and does not depend on the order of the rows.
"""
import re
import unicodedata
import zlib
from collections import Counter

import numpy as np
import pandas as pd

AUG_SEED = 7
AUG_COPIES = 1
AUG_COPIES_HANGUL = 2   # 10/31: 2 with the wider Hangul changes; copies per Hangul row; 3 gave no gain on 5 folds (Hangul rows are 10% of train, 70% of the score)
# 10/25: extra copies of each Hangul row whose name is shortened (first word only, last word only, one word
# dropped, end cut) and whose tonnage is kept.  Public Hangul rows all carry tonnage (10/24 diagnostic), and
# shortened names are where the ranker loses most with tonnage present.  0 = off (the 10/23 model);
# fold 0 with 2: Hangul +52 rows (with tonnage +28), Latin +94 rows against the 10/23 model.
AUG_PARTIAL_HANGUL = 2
AUG_WEIGHT = 0.5          # sample weight of an augmented row relative to its source row
# 10/28: share of training rows that also get a renamed copy: the name of another train vessel (same script), the
# other fields kept (a vessel renamed between the port record and the catalog).  0 = off.
AUG_RENAME = 0.0
P_NAME, P_TON, P_META, P_EMPTY = 0.75, 0.35, 0.15, 0.03
TAG = "#aug"              # sample_id suffix of augmented rows

QWERTY = ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM"]
_NEIGH = {}
for _r, _row in enumerate(QWERTY):
    for _i, _c in enumerate(_row):
        _NEIGH[_c] = [QWERTY[rr][j] for rr in (_r - 1, _r, _r + 1) if 0 <= rr < 3
                      for j in (_i - 1, _i, _i + 1) if 0 <= j < len(QWERTY[rr]) and (rr, j) != (_r, _i)]
ROMANS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
PREFIXES = ["M.V ", "MV. ", "M.T ", "MT. ", "S.S ", "M/V. "]


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


# ---------------------------------------------------------------- Latin names
def _is_letter(c):
    return c.isascii() and c.isalpha()


def l_typo(name, rng, ctx):
    let = [i for i, c in enumerate(name) if _is_letter(c)]
    if len(let) < 3:
        return None
    kind = _pick(rng, ("drop", "double", "swap", "sub"))
    i = _pick(rng, let)
    if kind == "drop":
        return name[:i] + name[i + 1:]
    if kind == "double":
        return name[:i + 1] + name[i:]
    if kind == "swap":
        if i + 1 >= len(name) or not _is_letter(name[i + 1]) or name[i + 1] == name[i]:
            return None
        return name[:i] + name[i + 1] + name[i] + name[i + 2:]
    return name[:i] + _pick(rng, _NEIGH[name[i].upper()]) + name[i + 1:]


def l_typos(name, rng, ctx):
    out = name
    for _ in range(int(rng.integers(1, 3))):
        out = l_typo(out, rng, ctx) or out
    return out if out != name else None


def l_join(name, rng, ctx):
    pos = [m.start() for m in re.finditer(r"(?<=\w) (?=\w)", name)]
    if not pos:
        return None
    p = _pick(rng, pos)
    return name[:p] + name[p + 1:]


def l_split(name, rng, ctx):
    pos = []
    for m in re.finditer(r"[A-Z]{5,}", name):
        pos += list(range(m.start() + 2, m.end() - 2))
    if not pos:
        return None
    p = _pick(rng, pos)
    return name[:p] + " " + name[p:]


def l_nospace(name, rng, ctx):
    return name.replace(" ", "") if " " in name.strip() else None


def _devowel(w):
    out = w[0] + "".join(c for c in w[1:] if c not in "AEIOU")
    return re.sub(r"(.)\1+", r"\1", out)


def l_abbr(name, rng, ctx):
    toks = name.split()
    cand = [i for i, t in enumerate(toks) if t.isalpha() and len(t) >= 4]
    if not cand or len(toks) < 2:
        return None
    i = _pick(rng, cand)
    w = toks[i]
    kind = _pick(rng, ("prefix", "prefix", "initial", "devowel"))
    if kind == "initial":
        new = w[0]
    elif kind == "prefix":
        new = w[: int(rng.integers(2, min(5, len(w) - 1) + 1))]
    else:
        new = _devowel(w)
    if rng.random() < 0.25:
        new += "."
    if new.rstrip(".") == w:
        return None
    toks[i] = new
    return " ".join(toks)


def l_drop(name, rng, ctx):
    toks = name.split()
    if len(toks) < 2:
        return None
    i = int(rng.integers(len(toks)))
    rest = toks[:i] + toks[i + 1:]
    if not any(_is_letter(c) for t in rest for c in t):
        return None
    return " ".join(rest)


def l_add(name, rng, ctx):
    toks = name.split()
    if rng.random() < 0.5:
        return " ".join([_pick(rng, ctx["first_toks"])] + toks)
    return " ".join(toks + [_pick(rng, ctx["last_toks"])])


def l_swap(name, rng, ctx):
    toks = name.split()
    if len(toks) < 2:
        return None
    i = int(rng.integers(len(toks) - 1))
    if toks[i] == toks[i + 1]:
        return None
    toks[i], toks[i + 1] = toks[i + 1], toks[i]
    return " ".join(toks)


def l_affix(name, rng, ctx):
    toks = name.split()
    num = [i for i, t in enumerate(toks) if t.isdigit() and 1 <= int(t) <= 10]
    rom = [i for i, t in enumerate(toks) if t in ROMANS and i > 0]
    if (num or rom) and rng.random() < 0.5:
        i = _pick(rng, num + rom)
        toks[i] = ROMANS[int(toks[i]) - 1] if toks[i].isdigit() else str(ROMANS.index(toks[i]) + 1)
        return " ".join(toks)
    return _pick(rng, PREFIXES) + name


def l_trunc(name, rng, ctx):
    """Name cut at the end: 50-85% of its letters / digits kept (at least 4), may end mid-word."""
    pos = [i for i, c in enumerate(name) if not c.isspace()]
    if len(pos) < 6:
        return None
    keep = max(4, int(np.ceil(len(pos) * rng.uniform(0.5, 0.85))))
    if keep >= len(pos):
        return None
    new = name[:pos[keep - 1] + 1].rstrip()
    return new if any(_is_letter(c) for c in new) else None


LAT_OPS = (l_typos, l_typos, l_join, l_split, l_nospace, l_abbr, l_abbr, l_drop, l_drop, l_add, l_swap, l_affix,
           l_trunc)


# ---------------------------------------------------------------- Hangul names
def _syl(c):
    return "가" <= c <= "힣"


def _protected(name):
    prot = set()
    s = name.rstrip()
    if s and s[-1] == "호":
        prot.add(len(s) - 1)
    for m in re.finditer(r"제(?=\s*\d)", name):
        prot.add(m.start())
    return prot


V_PAIRS = ["ㅓㅗ", "ㅐㅔ", "ㅡㅜ", "ㅕㅛ", "ㅙㅞ", "ㅚㅞ", "ㅟㅜ", "ㅢㅣ", "ㅒㅖ", "ㅑㅏ"]
C_PAIRS = ["ㄱㅋ", "ㄷㅌ", "ㅂㅍ", "ㅈㅊ", "ㅅㅆ", "ㄱㄲ", "ㄷㄸ", "ㅂㅃ", "ㅈㅉ"]
_L = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_V = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_T = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
      "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
COMMON_T = ["ㄴ", "ㅇ", "ㄹ", "ㅅ", "ㄱ", "ㅁ"]


def _alt(ch, pairs):
    for p in pairs:
        if ch in p:
            return p[1 - p.index(ch)]
    return None


def h_jamo(name, rng, ctx):
    prot = _protected(name)
    pos = [i for i, c in enumerate(name) if _syl(c) and i not in prot]
    if not pos:
        return None
    i = _pick(rng, pos)
    code = ord(name[i]) - 0xAC00
    l, v, t = code // 588, (code % 588) // 28, code % 28
    kind = _pick(rng, ("vowel", "cons", "final", "random"))
    if kind == "vowel":
        a = _alt(_V[v], V_PAIRS)
        if a is None:
            return None
        v = _V.index(a)
    elif kind == "cons":
        a = _alt(_L[l], C_PAIRS)
        if a is None:
            return None
        l = _L.index(a)
    elif kind == "final":
        t = 0 if t else _T.index(_pick(rng, COMMON_T))
    else:
        part = int(rng.integers(3))
        if part == 0:
            l = (l + 1 + int(rng.integers(len(_L) - 1))) % len(_L)
        elif part == 1:
            v = (v + 1 + int(rng.integers(len(_V) - 1))) % len(_V)
        else:
            t = (t + 1 + int(rng.integers(len(_T) - 1))) % len(_T)
    new = chr(0xAC00 + l * 588 + v * 28 + t)
    return None if new == name[i] else name[:i] + new + name[i + 1:]


def h_syl(name, rng, ctx):
    prot = _protected(name)
    pos = [i for i, c in enumerate(name) if _syl(c) and i not in prot]
    if len(pos) < 3:
        return None
    i = _pick(rng, pos)
    kind = _pick(rng, ("drop", "double", "swap"))
    if kind == "drop":
        return name[:i] + name[i + 1:]
    if kind == "double":
        return name[:i + 1] + name[i:]
    if i + 1 < len(name) and _syl(name[i + 1]) and i + 1 not in prot and name[i + 1] != name[i]:
        return name[:i] + name[i + 1] + name[i] + name[i + 2:]
    return None


def h_space(name, rng, ctx):
    prot = _protected(name)
    ins = [i for i in range(len(name) - 1) if _syl(name[i]) and _syl(name[i + 1]) and i not in prot and i + 1 not in prot]
    rem = [i for i in range(1, len(name) - 1) if name[i] == " " and _syl(name[i - 1]) and _syl(name[i + 1])]
    if rem and (not ins or rng.random() < 0.5):
        p = _pick(rng, rem)
        return name[:p] + name[p + 1:]
    if not ins:
        return None
    p = _pick(rng, ins)
    return name[:p + 1] + " " + name[p + 1:]


def h_nospace(name, rng, ctx):
    return name.replace(" ", "") if " " in name.strip() else None


def h_drop(name, rng, ctx):
    toks = name.split()
    if len(toks) < 2:
        return None
    i = int(rng.integers(len(toks)))
    rest = toks[:i] + toks[i + 1:]
    if not any(_syl(c) for t in rest for c in t):
        return None
    return " ".join(rest)


def h_ho(name, rng, ctx):
    s = name.rstrip()
    return None if s.endswith("호") else s + "호"


def h_latin(name, rng, ctx):
    """One Hangul word written with the catalog's own English word (aligned by position)."""
    toks = name.split()
    ct = ctx["cat_clean"][ctx["target"]].split()
    if len(toks) != len(ct) or len(toks) < 2:
        return None
    pos = [i for i, t in enumerate(toks) if any(_syl(c) for c in t) and ct[i].isalpha()]
    if not pos:
        return None
    i = _pick(rng, pos)
    toks[i] = ct[i]
    return " ".join(toks)


def h_swap(name, rng, ctx):
    """Two adjacent words swapped (Hangul names had no word-order operator; Latin ones have l_swap)."""
    toks = name.split()
    pairs = [i for i in range(len(toks) - 1) if toks[i] != toks[i + 1]]
    if not pairs:
        return None
    i = _pick(rng, pairs)
    toks[i], toks[i + 1] = toks[i + 1], toks[i]
    return " ".join(toks)


def h_trunc(name, rng, ctx):
    """Name cut at the end: the last 1..n/2 syllables dropped (at least 2 kept)."""
    pos = [i for i, c in enumerate(name) if _syl(c)]
    if len(pos) < 4:
        return None
    cut = int(rng.integers(1, len(pos) // 2 + 1))
    return name[:pos[len(pos) - cut - 1] + 1].rstrip()


HAN_OPS = (h_jamo, h_jamo, h_syl, h_space, h_nospace, h_drop, h_drop, h_ho, h_latin, h_swap, h_trunc)


# ---------------------------------------------------------------- Hangul keyboard slips (dubeolsik on QWERTY)
_KB_L = dict(zip("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ", "rRseEfaqQtTdwWczxvg"))
_KB_V = {"ㅏ": "k", "ㅐ": "o", "ㅑ": "i", "ㅒ": "O", "ㅓ": "j", "ㅔ": "p", "ㅕ": "u", "ㅖ": "P", "ㅗ": "h", "ㅘ": "hk",
         "ㅙ": "ho", "ㅚ": "hl", "ㅛ": "y", "ㅜ": "n", "ㅝ": "nj", "ㅞ": "np", "ㅟ": "nl", "ㅠ": "b", "ㅡ": "m", "ㅢ": "ml",
         "ㅣ": "l"}
_KB_TC = {"ㄳ": "ㄱㅅ", "ㄵ": "ㄴㅈ", "ㄶ": "ㄴㅎ", "ㄺ": "ㄹㄱ", "ㄻ": "ㄹㅁ", "ㄼ": "ㄹㅂ", "ㄽ": "ㄹㅅ", "ㄾ": "ㄹㅌ",
          "ㄿ": "ㄹㅍ", "ㅀ": "ㄹㅎ", "ㅄ": "ㅂㅅ"}
_KB_K2J = {v: k for k, v in _KB_L.items()}
_KB_K2J.update({v: k for k, v in _KB_V.items() if len(v) == 1})
_KB_VC = {("ㅗ", "ㅏ"): "ㅘ", ("ㅗ", "ㅐ"): "ㅙ", ("ㅗ", "ㅣ"): "ㅚ", ("ㅜ", "ㅓ"): "ㅝ", ("ㅜ", "ㅔ"): "ㅞ",
          ("ㅜ", "ㅣ"): "ㅟ", ("ㅡ", "ㅣ"): "ㅢ"}
_KB_TCK = {tuple(v): k for k, v in _KB_TC.items()}
_KB_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
_KB_NB = {}
for _r, _row in enumerate(_KB_ROWS):
    for _i, _c in enumerate(_row):
        _KB_NB[_c] = [_KB_ROWS[rr][j] for rr in (_r - 1, _r, _r + 1) if 0 <= rr < 3
                      for j in (_i - 1, _i, _i + 1) if 0 <= j < len(_KB_ROWS[rr]) and (rr, j) != (_r, _i)]


def _kb_keys(s):
    out = []
    for ch in s:
        if _syl(ch):
            c = ord(ch) - 0xAC00
            l, v, t = c // 588, (c % 588) // 28, c % 28
            out.append(_KB_L[_L[l]] + _KB_V[_V[v]])
            if t:
                out.append("".join(_KB_L[x] for x in _KB_TC.get(_T[t], _T[t])))
        else:
            out.append(ch)
    return "".join(out)


def _kb_compose(keys):
    """Dubeolsik IME automaton (jamo that cannot compose are left standalone, as an IME shows them)."""
    jam = [_KB_K2J.get(k, _KB_K2J.get(k.lower(), k)) if k.isascii() and k.isalpha() else k for k in keys]
    out, i, n = [], 0, len(jam)
    while i < n:
        j = jam[i]
        if j in _L and i + 1 < n and jam[i + 1] in _V:
            L, V = j, jam[i + 1]
            i += 2
            if i < n and (V, jam[i]) in _KB_VC:
                V = _KB_VC[(V, jam[i])]
                i += 1
            T = ""
            if i < n and jam[i] in _L and jam[i] in _T and not (i + 1 < n and jam[i + 1] in _V):
                T = jam[i]
                i += 1
                if i < n and jam[i] in _L and (T, jam[i]) in _KB_TCK and not (i + 1 < n and jam[i + 1] in _V):
                    T = _KB_TCK[(T, jam[i])]
                    i += 1
            out.append(chr(0xAC00 + _L.index(L) * 588 + _V.index(V) * 28 + _T.index(T)))
        else:
            out.append(j)
            i += 1
    return "".join(out)


def h_kbd(name, rng, ctx):
    """1-3 keyboard slips (drop / double / swap / neighbouring key) in the dubeolsik key sequence of the name."""
    if _protected(name):
        return None
    out = _kb_keys(name)
    for _ in range(1 + int(rng.integers(3))):
        let = [i for i, c in enumerate(out) if c.isascii() and c.isalpha()]
        if len(let) < 4:
            break
        i = _pick(rng, let)
        kind = _pick(rng, ("drop", "double", "swap", "sub", "sub"))
        if kind == "drop":
            out = out[:i] + out[i + 1:]
        elif kind == "double":
            out = out[:i + 1] + out[i] + out[i + 1:]
        elif kind == "swap":
            if i + 1 < len(out) and out[i + 1].isascii() and out[i + 1].isalpha() and out[i + 1] != out[i]:
                out = out[:i] + out[i + 1] + out[i] + out[i + 2:]
        else:
            nb = _KB_NB.get(out[i].lower())
            if nb:
                out = out[:i] + _pick(rng, nb) + out[i + 1:]
    new = _kb_compose(out)
    return new if new != name else None


def h_abbr(name, rng, ctx):
    """One word of 3+ syllables clipped to its first 1-2 syllables (인터내셔널 -> 인터)."""
    toks = name.split()
    cand = [i for i, t in enumerate(toks) if sum(_syl(c) for c in t) >= 3]
    if not cand or len(toks) < 2:
        return None
    i = _pick(rng, cand)
    toks[i] = toks[i][:1 + int(rng.integers(2))]
    return " ".join(toks)


AUG_HAN_EXTRA = True      # keyboard slips, clipped words and two-step changes in the Hangul copies (10/31)
P_HAN_SECOND = 0.35       # chance that a Hangul copy gets a second name change
HAN_OPS_X = HAN_OPS + (h_kbd, h_kbd, h_kbd, h_abbr)


# ---------------------------------------------------------------- tonnage / meta
def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) and x > 0 else np.nan


def t_op(row, rng, ctx):
    gt, igt = _num(row["gross_tonnage"]), _num(row["international_gross_tonnage"])
    if not (np.isfinite(gt) or np.isfinite(igt)):
        return False
    kind = _pick(rng, ("jitter", "jitter", "jitter", "nogt", "noigt", "none", "dwt"))
    if kind == "dwt":
        return t_dwt(row, rng, ctx)
    if kind == "jitter":
        for c, v in (("gross_tonnage", gt), ("international_gross_tonnage", igt)):
            if np.isfinite(v):
                row[c] = float(max(1.0, round(v * float(np.exp(rng.uniform(np.log(0.75), np.log(1.3)))))))
    if kind in ("nogt", "none"):
        row["gross_tonnage"] = np.nan
    if kind in ("noigt", "none"):
        row["international_gross_tonnage"] = np.nan
    return True


def t_dwt(row, rng, ctx):
    """Tonnage written as the catalog deadweight (wrong field)."""
    dwt = ctx["cat_dwt"][ctx["target"]]
    if not (np.isfinite(dwt) and dwt > 0):
        return False
    cols = _pick(rng, (("gross_tonnage",), ("international_gross_tonnage",), ("gross_tonnage", "international_gross_tonnage")))
    for c in cols:
        row[c] = float(round(dwt))
    return True


def m_op(row, rng, ctx):
    kind = _pick(rng, ("noflag", "notype", "randflag", "randtype"))
    if kind == "noflag":
        row["flag_name"] = np.nan
    elif kind == "notype":
        row["vessel_type"] = np.nan
    elif kind == "randflag":
        row["flag_name"] = _pick(rng, ctx["flags"])
    else:
        row["vessel_type"] = _pick(rng, ctx["types"])
    return True


# ---------------------------------------------------------------- driver
def _context(train, cat):
    first, last = Counter(), Counter()
    for c in cat.clean:
        t = c.split()
        if len(t) >= 2:
            if t[0].isalpha() and len(t[0]) >= 2:
                first[t[0]] += 1
            if t[-1].isalpha() and len(t[-1]) >= 2:
                last[t[-1]] += 1
    return {
        "first_toks": [t for t, _ in first.most_common(300)],
        "last_toks": [t for t, _ in last.most_common(300)],
        "flags": sorted(train["flag_name"].dropna().astype(str).unique()),
        "types": sorted(train["vessel_type"].dropna().astype(str).unique()),
        "cat_clean": cat.clean,
        "cat_dwt": cat.dwt,
    }


def _is_hangul(v):
    return isinstance(v, str) and any(_syl(c) for c in v)


def augment(train: pd.DataFrame, cat) -> pd.DataFrame:
    """AUG_COPIES perturbed copies of every training row (same target), sample_id + '#augN'."""
    ctx = _context(train, cat)
    out = []
    for rec in train.to_dict("records"):
        name = rec["vessel_name"] if isinstance(rec["vessel_name"], str) else ""
        ctx["target"] = cat.id_to_index[str(rec["target_vessel_id"])]
        ops = HAN_OPS if _is_hangul(name) else LAT_OPS
        for copy_no in range(AUG_COPIES_HANGUL if _is_hangul(name) else AUG_COPIES):
            sid = str(rec["sample_id"])
            rng = np.random.default_rng([AUG_SEED, zlib.crc32(sid.encode("utf-8")), copy_no])
            row = dict(rec)
            changed = False
            if name and rng.random() < P_EMPTY:  # name missing altogether
                row["vessel_name"] = np.nan
                changed = True
            elif name and rng.random() < P_NAME:
                if AUG_HAN_EXTRA and ops is HAN_OPS:
                    ops = HAN_OPS_X
                for _ in range(4):
                    new = _pick(rng, ops)(unicodedata.normalize("NFC", name), rng, ctx)
                    if new and new != name:
                        row["vessel_name"] = new
                        changed = True
                        break
                if changed and ops is HAN_OPS_X and rng.random() < P_HAN_SECOND:
                    for _ in range(4):
                        new2 = _pick(rng, ops)(row["vessel_name"], rng, ctx)
                        if new2 and new2 != row["vessel_name"] and any(_syl(c) for c in new2):
                            row["vessel_name"] = new2
                            break
            if rng.random() < P_TON:
                changed = t_op(row, rng, ctx) or changed
            if rng.random() < P_META:
                changed = m_op(row, rng, ctx) or changed
            if not changed:
                changed = t_op(row, rng, ctx)
            if changed:
                row["sample_id"] = f"{sid}{TAG}{copy_no}"
                out.append(row)
    if AUG_PARTIAL_HANGUL:
        out += _partial_hangul(train, cat, ctx)
    if AUG_RENAME:
        out += _renamed(train)
    return pd.DataFrame(out, columns=train.columns)


def _renamed(train) -> list:
    """AUG_RENAME of the rows (at random) get a copy named like another train vessel of the same script."""
    names = train["vessel_name"].where(train["vessel_name"].map(lambda v: isinstance(v, str) and bool(v.strip())))
    pairs = sorted(set(zip(names.dropna().map(lambda v: unicodedata.normalize("NFC", v)),
                           train.loc[names.notna(), "target_vessel_id"].astype(str))))
    pools = {True: [p for p in pairs if _is_hangul(p[0])], False: [p for p in pairs if not _is_hangul(p[0])]}
    out = []
    for rec in train.to_dict("records"):
        name = rec["vessel_name"] if isinstance(rec["vessel_name"], str) else ""
        if not name.strip():
            continue
        sid = str(rec["sample_id"])
        rng = np.random.default_rng([AUG_SEED + 202, zlib.crc32(sid.encode("utf-8"))])
        if rng.random() >= AUG_RENAME:
            continue
        pool = pools[_is_hangul(name)]
        vid = str(rec["target_vessel_id"])
        for _ in range(8):
            new, v = pool[int(rng.integers(len(pool)))]
            if v != vid:
                row = dict(rec)
                row["vessel_name"] = new
                row["sample_id"] = f"{sid}{TAG}R"
                out.append(row)
                break
    return out


def h_keepfirst(name, rng, ctx):
    toks = name.split()
    return toks[0] if len(toks) >= 2 else None


def h_keeplast(name, rng, ctx):
    toks = name.split()
    return toks[-1] if len(toks) >= 2 else None


PARTIAL_OPS = (h_keepfirst, h_keeplast, h_drop, h_trunc)


def _partial_hangul(train, cat, ctx) -> list:
    """AUG_PARTIAL_HANGUL shortened-name copies of every Hangul row (tonnage, flag and type kept)."""
    out = []
    for rec in train.to_dict("records"):
        name = rec["vessel_name"] if isinstance(rec["vessel_name"], str) else ""
        if not _is_hangul(name):
            continue
        ctx["target"] = cat.id_to_index[str(rec["target_vessel_id"])]
        sid = str(rec["sample_id"])
        nfc = unicodedata.normalize("NFC", name)
        for k in range(AUG_PARTIAL_HANGUL):
            rng = np.random.default_rng([AUG_SEED + 101, zlib.crc32(sid.encode("utf-8")), k])
            new = None
            for _ in range(4):
                cand = PARTIAL_OPS[int(rng.integers(len(PARTIAL_OPS)))](nfc, rng, ctx)
                if cand and cand != nfc:
                    new = cand
                    break
            if new is None:
                continue
            row = dict(rec)
            row["vessel_name"] = new
            row["sample_id"] = f"{sid}{TAG}P{k}"
            out.append(row)
    return out


def is_aug(sample_ids) -> np.ndarray:
    return np.array([TAG in str(s) for s in sample_ids], bool)
