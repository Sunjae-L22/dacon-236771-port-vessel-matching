# author: opus
"""Score LLM Korean spellings with KoMatcher (learned from real train pairs only) and keep one row per
(vessel, spelling). Same scoring as the v1 run that produced llm_korean_scored.csv.

usage (from shared/experiments/pinned/build27, KO_DICT=1):
    python3 ../../../llm_aug/score_llm.py <v1_wide.csv> <v2_long.csv> <out.csv>
"""
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import cv  # noqa: E402
import opus_lib as L  # noqa: E402


def main(v1_csv, v2_csv, out):
    tr, ca = cv.load()
    ix = L.Index(ca, tr)
    v1 = pd.read_csv(v1_csv, encoding="utf-8-sig")
    gen = [v1[["vessel_id", c]].rename(columns={c: "ko"}).assign(src=f"v1_{c}")
           for c in ("ko_greedy", "ko_sample1", "ko_sample2")]
    v2 = pd.read_csv(v2_csv, encoding="utf-8-sig")
    gen.append(v2.assign(src="v2_p" + v2["prompt_id"].astype(str) + "_" + v2["kind"])[["vessel_id", "ko", "src"]])
    g = pd.concat(gen, ignore_index=True)
    g["ko"] = g["ko"].fillna("").astype(str).str.strip()
    g = g[g["ko"].map(lambda n: bool(re.search("[가-힣]", n)) and not re.search("[A-Za-z]", n))]
    g = g.drop_duplicates(["vessel_id", "ko"]).reset_index(drop=True)  # v1 rows come first and win
    g["score"] = [float(ix.ko.score(n, ix.id2i[v])["ko_combined_score"]) for n, v in zip(g["ko"], g["vessel_id"])]
    print("hangul-only, deduped:", len(g), "quantiles", np.quantile(g["score"], [.05, .1, .25, .5]).round(3))
    for th in (0.6, 0.65, 0.7, 0.75):
        k = g[g["score"] >= th]
        print(f"keep >= {th}: {len(k)} rows ({k.src.str.startswith('v2').sum()} from v2), {k.vessel_id.nunique()} vessels")
    print(g.groupby(g["src"].str[:5])["score"].agg(["size", "mean"]).round(3).to_string())
    g.to_csv(out, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main(*sys.argv[1:4])
