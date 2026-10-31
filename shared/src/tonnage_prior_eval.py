# author: opus
"""Does the ranker under-use tonnage? Re-score held-out top-20 candidates with
    score = ranker_score + lam * log p(ratio)
where ratio = (query GT or IGT, whichever is closer) / candidate catalog GT and p is the empirical
density of that ratio for true pairs in train-role rows only (terra/byt5_pilot/split_v1).
lam is chosen on tune (mixed objective 0.5*orig + 0.5*corrupted mean), reported on eval.

usage: python3 shared/src/tonnage_prior_eval.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "open"
EDGES = np.round(np.arange(0.88, 1.1201, 0.005), 3)  # ratio bins; outside -> floor
LAMS = (0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)


def closest_ratio(gt, igt, cgt):
    r1, r2 = gt / cgt, igt / cgt
    r = np.where(np.abs(r1 - 1) <= np.abs(r2 - 1), r1, r2)
    return np.where(np.isfinite(r1) | np.isfinite(r2), np.where(np.isfinite(r), r, np.where(np.isfinite(r1), r1, r2)), np.nan)


def main():
    tr = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig")
    ca = pd.read_csv(DATA / "vessel_catalog.csv", encoding="utf-8-sig").set_index("vessel_id")
    role = pd.read_csv(ROOT / "terra" / "byt5_pilot" / "split_v1" / "rows.csv").set_index("sample_id")["role"]
    tr["role"] = tr["sample_id"].map(role)

    fit = tr[tr["role"] == "train"].join(ca[["gross_tonnage"]].rename(columns={"gross_tonnage": "cgt"}), on="target_vessel_id")
    r = closest_ratio(fit["gross_tonnage"].to_numpy(float), fit["international_gross_tonnage"].to_numpy(float), fit["cgt"].to_numpy(float))
    r = r[np.isfinite(r)]
    exact = np.abs(r - 1) < 0.0005
    hist, _ = np.histogram(r[~exact], bins=EDGES)
    width = np.diff(EDGES)
    dens = (hist + 0.5) / (len(r) * width)  # smoothed density of non-exact ratios
    p_exact = exact.mean()
    floor = 0.5 / (len(r) * 0.1)
    print(f"density from {len(r)} train-role pairs; exact share {p_exact:.3f}")

    def logp(ratio):
        out = np.full(len(ratio), np.log(floor))
        ok = np.isfinite(ratio)
        ex = ok & (np.abs(ratio - 1) < 0.0005)
        out[ex] = np.log(p_exact / 0.001)  # point mass spread over the exact tolerance
        inb = ok & ~ex & (ratio >= EDGES[0]) & (ratio < EDGES[-1])
        idx = np.clip(np.searchsorted(EDGES, ratio[inb], side="right") - 1, 0, len(dens) - 1)
        out[inb] = np.log(dens[idx])
        out[~ok] = 0.0  # no query tonnage: no evidence
        return out

    c = pd.read_csv(ROOT / "shared" / "experiments" / "llm_rerank" / "candidates_top20.csv", encoding="utf-8-sig")
    q = tr.set_index("sample_id")[["gross_tonnage", "international_gross_tonnage"]]
    c = c.join(q, on="orig_sample_id").join(ca[["gross_tonnage"]].rename(columns={"gross_tonnage": "cgt"}), on="candidate_vessel_id")
    c["lp"] = logp(closest_ratio(c["gross_tonnage"].to_numpy(float), c["international_gross_tonnage"].to_numpy(float),
                                 c["cgt"].to_numpy(float)))
    c["lp"] -= c.groupby("query_sample_id")["lp"].transform("max")  # relative within query

    def acc(frame, lam):
        s = frame["ranker_score"] + lam * frame["lp"]
        top = frame.assign(s=s).sort_values(["query_sample_id", "s", "rank"], ascending=[True, False, True]).groupby("query_sample_id").head(1)
        by = top.groupby("variant")["is_target"].mean()
        return by

    rows = []
    for lam in LAMS:
        for rl in ("tune", "eval"):
            by = acc(c[c["role"] == rl], lam)
            cor = by.drop("orig").mean()
            rows.append(dict(lam=lam, role=rl, orig=by["orig"], corrupted=cor, mixed=0.5 * by["orig"] + 0.5 * cor,
                             **{f"v_{k}": v for k, v in by.items() if k != "orig"}))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(t[["lam", "role", "orig", "corrupted", "mixed"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    tune = t[t["role"] == "tune"]
    best = tune.loc[tune["mixed"].idxmax(), "lam"]
    ev = t[t["role"] == "eval"].set_index("lam")
    print(f"\ntune-selected lam={best}; eval mixed {ev.loc[0, 'mixed']:.4f} -> {ev.loc[best, 'mixed']:.4f}, "
          f"corrupted {ev.loc[0, 'corrupted']:.4f} -> {ev.loc[best, 'corrupted']:.4f}, orig {ev.loc[best, 'orig']:.4f}")
    vcols = [k for k in t.columns if k.startswith("v_")]
    print(ev.loc[[0, best], vcols].T.to_string(float_format=lambda x: f"{x:.3f}"))
    t.to_csv(ROOT / "shared" / "experiments" / "tonnage_prior_grid.csv", index=False)


if __name__ == "__main__":
    main()
