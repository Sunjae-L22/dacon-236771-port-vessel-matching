# author: opus
"""Wider corruption families seen in real port records, vessel-split fold 0."""
import random
import re
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cv import folds, load  # noqa: E402
import opus_lib as L  # noqa: E402
from opus_cv import PARAMS, fold_data  # noqa: E402

R = random.Random(0)


def op(fn):
    def g(r, cat_name):
        r = dict(r)
        r["vessel_name"] = fn(str(r["vessel_name"]), cat_name)
        return r
    return g


def trunc_at(n):
    return op(lambda s, c: s[:n].rstrip())


def reorder(s, c):
    w = s.split()
    return " ".join(w[1:] + w[:1]) if len(w) > 1 else s


def mixed(s, c):
    """Korean row with one token left in the catalog's English spelling."""
    w, e = s.split(), c.upper().split()
    if L.is_hangul(s) and len(w) == len(e) and len(w) > 1:
        i = R.randrange(len(w))
        w[i] = e[i]
        return " ".join(w)
    return s


def abbrev(s, c):
    w = s.split()
    if len(w) < 2:
        return s
    i = max(range(len(w)), key=lambda j: len(w[j]))
    if w[i].isascii() and len(w[i]) > 4:
        w[i] = w[i][0] + re.sub(r"[AEIOUaeiou]", "", w[i][1:])
    return " ".join(w)


def initial(s, c):
    w = s.split()
    return " ".join([w[0][0] + "."] + w[1:]) if len(w) > 1 and w[0].isascii() else s


def numfmt(s, c):
    s2 = re.sub(r"\bNO\.?\s*(\d+)", r"#\1", s, flags=re.I)
    if s2 != s:
        return s2
    return re.sub(r"제(\d+)", r"\1", s)


_VOWEL_SWAP = {0: 4, 4: 0, 8: 13, 13: 8, 18: 20, 20: 18, 5: 1, 1: 5}  # ㅏ↔ㅓ ㅗ↔ㅜ ㅡ↔ㅣ ㅔ↔ㅐ
_CHO_SWAP = {0: 15, 15: 0, 3: 16, 16: 3, 7: 17, 17: 7, 12: 14, 14: 12, 9: 10, 10: 9}  # ㄱ↔ㅋ ㄷ↔ㅌ ㅂ↔ㅍ ㅈ↔ㅊ ㅅ↔ㅆ


def jamo_typo(s, c, n=1):
    """Korean typo inside a syllable: confusable vowel or initial consonant swapped."""
    chars = list(s)
    idx = [i for i, ch in enumerate(chars) if "가" <= ch <= "힣"]
    for _ in range(n):
        if not idx:
            break
        i = R.choice(idx)
        o = ord(chars[i]) - 0xAC00
        cho, jung, jong = o // 588, (o % 588) // 28, o % 28
        if R.random() < 0.5 and jung in _VOWEL_SWAP:
            jung = _VOWEL_SWAP[jung]
        elif cho in _CHO_SWAP:
            cho = _CHO_SWAP[cho]
        else:
            jong = 0 if jong else 4  # add or drop a final ㄴ
        chars[i] = chr(0xAC00 + cho * 588 + jung * 28 + jong)
    return "".join(chars)


def punct(s, c):
    return s.replace(" ", "-") if R.random() < 0.5 else s.replace(" ", "  ") + "."


V = {
    "orig": op(lambda s, c: s),
    "trunc8": trunc_at(8), "trunc10": trunc_at(10), "trunc12": trunc_at(12), "trunc15": trunc_at(15),
    "reorder": op(reorder), "mixed_script": op(mixed), "abbrev_vowels": op(abbrev),
    "initial": op(initial), "jamo1": op(lambda s, c: jamo_typo(s, c, 1)), "jamo2": op(lambda s, c: jamo_typo(s, c, 2)), "num_format": op(numfmt), "punct": op(punct),
}


def main(tag, k=0, n_en=400):
    train, catalog = load()
    f = folds(train)
    Xtr, ytr, gtr, *_ = fold_data(train, catalog, f, k, tag)
    m = lgb.train(PARAMS, lgb.Dataset(Xtr, ytr, group=gtr, feature_name=L.FEATURES), 600)
    tr, va = train[f != k], train[f == k]
    ix = L.Index(catalog, tr)
    sets = {"ko": va[va.vessel_name.map(L.is_hangul)],
            "en": va[~va.vessel_name.map(L.is_hangul)].sample(n_en, random_state=0)}
    for name, fn in V.items():
        out = []
        for lang, sub in sets.items():
            ok = rec = changed = 0
            for row in sub.to_dict("records"):
                t = ix.id2i[row["target_vessel_id"]]
                new = fn(row, ix.names_raw[t])
                changed += new["vessel_name"] != row["vessel_name"]
                idx, F = L.query_features(ix, new, L.query_candidates(ix, new))
                rec += t in idx
                ok += idx[int(np.argmax(m.predict(F)))] == t
            out.append(f"{lang} acc={ok / len(sub):.3f} rec={rec / len(sub):.3f} chg={changed / len(sub):.2f}")
        print(f"{tag} {name:13s} " + " | ".join(out), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
