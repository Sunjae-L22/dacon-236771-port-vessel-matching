# author: opus
"""Candidate packets for an LLM judge on Korean rows (offline evaluation on Colab).

Held-out Korean rows of fold FOLD (astra/incumbent97 context, seed-0 79-recipe ranker, train targets of the
fit split excluded as at inference) under several name/tonnage conditions. For each query: the record
fields, the top-K allowed candidates with catalog fields and ranker scores, the answer's catalog index and
the leader/runner-up gap (hard = gap < 4, the 102 split). Several unlike conditions on purpose: a judge is
only worth submitting if it helps on all of them, since none is known to match the test.

usage: python3 judge_packets.py FOLD   -> judge/packets_foldK.jsonl
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "astra"))
sys.path.insert(0, str(Path(__file__).parent))
import incumbent97 as I  # noqa: E402
from jamo_fingerprint import name  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

B = I.B
K = 10
CONDS = [("clean", "keep"), ("jamo2", "s15"), ("mix", "s15"), ("corrupt", "keep"), ("corrupt", "s15")]
QF = ["vessel_name", "gross_tonnage", "international_gross_tonnage", "flag_name", "vessel_type", "arrival_year"]
CF = ["vessel_id", "vessel_name", "vessel_type", "flag_name", "gross_tonnage", "deadweight_tonnage"]


def clean(v):
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def main(fold):
    tr, cat, fit, va, ix, llm, out = I.context(fold)
    model = lgb.Booster(model_file=str(out / "seed0" / "ranker.txt"))
    fit_targets = set(ix.cs_map.values())
    rows = va[va.vessel_name.map(B.is_hangul)]
    rng = np.random.default_rng(700 + fold)
    crec = cat[CF].to_dict("records")
    dst = Path(__file__).parent / "judge"
    dst.mkdir(exist_ok=True)
    with open(dst / f"packets_fold{fold}.jsonl", "w") as f:
        for r in rows.to_dict("records"):
            base = {k: r.get(k) for k in I.MODEL_FIELDS}
            base["call_sign"] = "UNSEEN"
            t = ix.id2i[r["target_vessel_id"]]
            for nk, tk in CONDS:
                s0 = str(r["vessel_name"])
                nm = B._corrupt_name(s0, rng) if nk == "corrupt" else name(nk, s0, rng)
                q = dict(base, vessel_name=nm)
                if tk == "s15":
                    s = rng.uniform(0.85, 1.15)
                    q["gross_tonnage"] = r["gross_tonnage"] * s
                    q["international_gross_tonnage"] = r["international_gross_tonnage"] * s
                idx, F = B.query_features(ix, q, B.query_candidates(ix, q))
                sc = model.predict(F, num_threads=1)
                order = sorted(range(len(idx)), key=lambda j: (-float(sc[j]), idx[j]))
                allowed = [j for j in order if idx[j] not in fit_targets] or order
                gap = float(sc[allowed[0]] - sc[allowed[1]]) if len(allowed) > 1 else 99.0
                top = allowed[:K]
                f.write(json.dumps({
                    "fold": fold, "sample_id": r["sample_id"], "cond": f"{nk}+{tk}",
                    "query": {k: clean(q.get(k, r.get(k))) for k in QF},
                    "orig_name": s0, "gap": gap, "target": int(t),
                    "rank_of_target": next((i for i, j in enumerate(allowed) if idx[j] == t), -1),
                    "cands": [dict({k: clean(v) for k, v in crec[idx[j]].items()}, i=int(idx[j]), score=float(sc[j]))
                              for j in top],
                }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]))
