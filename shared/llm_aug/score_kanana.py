# author: opus
"""Score Kanana-2 Korean spellings exactly like the v1 Qwen file (llm_korean_scored.csv):
greedy/sample1/sample2 in that order, Hangul-only, no Latin, one row per (vessel, spelling),
KoMatcher ko_combined_score with the matcher built from train (KO_DICT=1).

usage (KO_DICT=1, cwd /tmp):
    python3 shared/llm_aug/score_kanana.py shared/llm_aug/llm_korean_aug_kanana.csv \
        shared/experiments/pinned/build70k/llm_korean_scored.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared/experiments/pinned/build70"))
import cv  # noqa: E402
import opus_lib as L  # noqa: E402


def main(src, out):
    tr, ca = cv.load()
    ix = L.Index(ca, tr)
    w = pd.read_csv(src, encoding="utf-8-sig")
    g = pd.concat([w[["vessel_id", c]].rename(columns={c: "ko"}).assign(src=c)
                   for c in ("ko_greedy", "ko_sample1", "ko_sample2")], ignore_index=True)
    g["ko"] = g["ko"].fillna("").astype(str).str.strip()
    g = g[g["ko"].map(lambda n: bool(re.search("[가-힣]", n)) and not re.search("[A-Za-z]", n))]
    g = g.drop_duplicates(["vessel_id", "ko"]).reset_index(drop=True)
    g["score"] = [float(ix.ko.score(n, ix.id2i[v])["ko_combined_score"]) for n, v in zip(g["ko"], g["vessel_id"])]
    print("rows", len(g), "quantiles", np.quantile(g["score"], [.05, .1, .25, .5]).round(3))
    print(g.groupby("src").score.agg(["size", "mean"]).round(3).to_string())
    keep = g[g.score >= 0.70]
    print("keep >= .70:", len(keep), "rows,", keep.vessel_id.nunique(), "vessels")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(out, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main(*sys.argv[1:3])
