# author: opus
"""Milder, realistic Korean noise than opus_lib._typo (which deletes/duplicates/swaps whole syllables):
jamo-level variants (vowel ㅐ/ㅔ, ㅓ/ㅗ, ㅡ/ㅜ; plain/aspirated/tense consonants; final consonant dropped
or added), spacing changes, and combinations with tonnage x U(0.85,1.15). Seed-0 79-recipe fold ranker,
held-out Korean rows, same fingerprint numbers as regime_fingerprint.py.

usage: python3 jamo_fingerprint.py FOLD   -> jamo_fp_foldK.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "astra"))
import incumbent97 as I  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

B = I.B
BG = B.FEATURES.index("best_gap")
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
V = {"ㅐ": "ㅔ", "ㅔ": "ㅐ", "ㅓ": "ㅗ", "ㅗ": "ㅓ", "ㅡ": "ㅜ", "ㅜ": "ㅡ", "ㅒ": "ㅖ", "ㅖ": "ㅒ", "ㅙ": "ㅞ", "ㅞ": "ㅙ", "ㅚ": "ㅞ"}
C = {"ㄱ": "ㅋ", "ㅋ": "ㄱ", "ㄷ": "ㅌ", "ㅌ": "ㄷ", "ㅂ": "ㅍ", "ㅍ": "ㅂ", "ㅈ": "ㅊ", "ㅊ": "ㅈ", "ㅅ": "ㅆ", "ㅆ": "ㅅ",
     "ㄲ": "ㄱ", "ㄸ": "ㄷ", "ㅃ": "ㅂ", "ㅉ": "ㅈ"}


def split(ch):
    o = ord(ch) - 0xAC00
    return o // 588, (o % 588) // 28, o % 28


def join(a, b, c):
    return chr(0xAC00 + a * 588 + b * 28 + c)


def jamo_typo(s, rng):
    pos = [i for i, ch in enumerate(s) if "가" <= ch <= "힣"]
    for _ in range(10):
        if not pos:
            return s
        i = pos[int(rng.integers(len(pos)))]
        a, b, c = split(s[i])
        op = int(rng.integers(3))
        if op == 0 and JUNG[b] in V:
            b = JUNG.index(V[JUNG[b]])
        elif op == 1 and CHO[a] in C:
            a = CHO.index(C[CHO[a]])
        elif op == 2:
            c = 0 if c else JONG.index(["ㄴ", "ㅇ", "ㄹ", "ㅅ"][int(rng.integers(4))])
        else:
            continue
        return s[:i] + join(a, b, c) + s[i + 1:]
    return s


def spacing(s, rng):
    if " " in s and rng.random() < 0.7:
        return s.replace(" ", "")
    i = int(rng.integers(1, max(2, len(s))))
    return s[:i] + " " + s[i:]


def name(kind, s, rng):
    if kind == "jamo":
        return jamo_typo(s, rng)
    if kind == "jamo2":
        return jamo_typo(jamo_typo(s, rng), rng)
    if kind == "space":
        return spacing(s, rng)
    if kind == "mix":  # one of: jamo, space, syllable typo
        k = int(rng.integers(3))
        return jamo_typo(s, rng) if k == 0 else spacing(s, rng) if k == 1 else B._typo(s, 1, rng)
    return s


REGIMES = [(n, t) for n in ("jamo", "jamo2", "space", "mix") for t in ("keep", "s15")]


def main(fold):
    tr, cat, fit, va, ix, llm, out = I.context(fold)
    model = lgb.Booster(model_file=str(out / "seed0" / "ranker.txt"))
    fit_targets = set(ix.cs_map.values())
    rows = va[va.vessel_name.map(B.is_hangul)]
    rng = np.random.default_rng(100 + fold)
    res = {f"{a}+{b}": [] for a, b in REGIMES}
    for r in rows.to_dict("records"):
        base = {k: r.get(k) for k in I.MODEL_FIELDS}
        base["call_sign"] = "UNSEEN"
        t = ix.id2i[r["target_vessel_id"]]
        for nk, tk in REGIMES:
            q = dict(base, vessel_name=name(nk, str(r["vessel_name"]), rng))
            if tk == "s15":
                s = rng.uniform(0.85, 1.15)
                q["gross_tonnage"] = r["gross_tonnage"] * s
                q["international_gross_tonnage"] = r["international_gross_tonnage"] * s
            idx, F = B.query_features(ix, q, B.query_candidates(ix, q))
            sc = model.predict(F, num_threads=1)
            order = sorted(range(len(idx)), key=lambda j: (-float(sc[j]), idx[j]))
            allowed = [j for j in order if idx[j] not in fit_targets] or order
            gap = float(sc[allowed[0]] - sc[allowed[1]]) if len(allowed) > 1 else 99.0
            pick = allowed[0]
            if gap < 4:
                top = allowed[:3]
                bg = np.nan_to_num(F[top, BG].astype(float), nan=9.0)
                i = int(np.argmin(bg))
                if bg[i] < 0.02 and bg[0] > 0.05:
                    pick = top[i]
            res[f"{nk}+{tk}"].append([int(idx[allowed[0]] == t), int(gap < 4), int(idx[pick] == t)])
    (Path(__file__).parent / f"jamo_fp_fold{fold}.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main(int(sys.argv[1]))
