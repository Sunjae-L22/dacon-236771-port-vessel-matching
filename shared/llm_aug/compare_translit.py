# author: opus
"""Compare LLM Korean transliterations against the real Hangul names in train.

Reference: train_korean_ref.csv (382 vessels with a real Korean record), minus the 30 few-shot
vessels shown in the prompt. For each model file (vessel_id, ko_greedy, ko_sample1, ko_sample2):
  exact      greedy == real (spaces ignored)
  any3       real is among greedy/sample1/sample2
  sim        mean character similarity of greedy (difflib ratio, spaces ignored)
  cer        mean character error rate of greedy (Levenshtein / len(real))

usage: python3 shared/llm_aug/compare_translit.py llm_korean_aug.csv llm_korean_aug_kanana.csv ...
"""
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def nos(s):
    return str(s if isinstance(s, str) else "").replace(" ", "")


def evaluate(path, ref):
    g = pd.read_csv(path, encoding="utf-8-sig").drop(columns=["en_name", "ko_name"], errors="ignore")
    m = ref.merge(g, on="vessel_id", how="left")
    real = m.ko_name.map(nos)
    cols = [c for c in ("ko_greedy", "ko_sample1", "ko_sample2") if c in m]
    gr = m.ko_greedy.map(nos)
    res = dict(
        file=Path(path).name, n=len(m), missing=int(m.ko_greedy.isna().sum()),
        exact=(gr == real).mean(),
        any3=pd.concat([m[c].map(nos) == real for c in cols], axis=1).any(axis=1).mean(),
        sim=pd.Series([SequenceMatcher(None, a, b).ratio() for a, b in zip(real, gr)]).mean(),
        cer=pd.Series([lev(a, b) / max(1, len(a)) for a, b in zip(real, gr)]).mean(),
    )
    m["sim"] = [SequenceMatcher(None, a, b).ratio() for a, b in zip(real, gr)]
    return res, m


def main(paths):
    ref = pd.read_csv(HERE / "train_korean_ref.csv")
    fs = pd.read_csv(HERE / "llm_fewshot.csv")
    ref = ref[~ref.vessel_id.isin(fs.vessel_id)]
    en = pd.read_csv(HERE / "llm_aug_input.csv")
    ref = ref.merge(en, on="vessel_id")
    rows, frames = [], {}
    for p in paths:
        r, m = evaluate(p, ref)
        rows.append(r)
        frames[Path(p).stem] = m.set_index("vessel_id")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    if len(frames) >= 2:  # where the models disagree on correctness
        a, b = list(frames)[:2]
        A, B = frames[a], frames[b]
        ok_a = A.ko_greedy.map(nos) == A.ko_name.map(nos)
        ok_b = B.ko_greedy.map(nos) == B.ko_name.map(nos)
        print(f"\n{a} only right: {int((ok_a & ~ok_b).sum())}   {b} only right: {int((~ok_a & ok_b).sum())}")
        d = pd.DataFrame({"en": A.en_name, "real": A.ko_name, a: A.ko_greedy, b: B.ko_greedy})
        print(d[ok_a != ok_b].head(25).to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or [str(HERE / "llm_korean_aug.csv")])
