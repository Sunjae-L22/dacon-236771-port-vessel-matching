"""DACON entry point (93 + LLM judge on hard Korean rows; settings in model/judge_config.json).

Rows go through 93 exactly. A Korean row whose leader / runner-up gap (mean of the six Korean rankers,
train targets excluded) is below GAP is re-decided among its top-K allowed candidates by an LLM
(model/llm, read from next-token logits, both candidate orderings), score = llm_logprob + BETA * ranker.
Each row is scored alone, so the prediction of a row never depends on the other test rows.
"""
from __future__ import annotations
from pathlib import Path
import json
import os
import pickle
import sys
import time
import unicodedata

if os.environ.get("PYTHONHASHSEED") is None:
    os.execve(sys.executable, [sys.executable] + sys.argv, dict(os.environ, PYTHONHASHSEED="0"))

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "model"))
import opus_lib as L
import opus_lib_65 as L65

CF = ["vessel_name", "vessel_type", "flag_name", "gross_tonnage", "deadweight_tonnage"]


def run(test_path="data/test.csv", catalog_path="data/vessel_catalog.csv", out="output/submission.csv"):
    t0 = time.time()
    cfg = json.loads((HERE / "model" / "judge_config.json").read_text())
    GAP, K, BETA = float(cfg["gap"]), int(cfg["k"]), float(cfg["beta"])
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    crec = catalog[CF].to_dict("records")
    with open(HERE / "model" / "stats.pkl", "rb") as f:
        stats = pickle.load(f)
    ix = L.Index(catalog, None)
    ix.cs_map = {k: ix.id2i[v] for k, v in stats["cs_map"].items() if v in ix.id2i}
    ix.flag_p, ix.type_p = stats["flag_p"], stats["type_p"]
    ix.train_targets = set(ix.cs_map.values())
    ix.ko = L.KA.KoMatcher(catalog, stats["ko_pairs"])
    ix.fit_alias(stats["alias_rows"])
    load = lambda fam: [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob(fam + "_*.txt"))]
    ko, en65, s62 = load("ranker"), load("en65"), load("s62")
    train_targets = {ix.id2i[v] for v in stats["cs_map"].values() if v in ix.id2i}

    import torch
    from judge_llm import Judge
    judge = Judge(str(HERE / "model" / "llm"), dtype=torch.bfloat16, device_map=os.environ.get("JUDGE_DEVICE", "cuda"))
    print(f"setup {time.time() - t0:.0f}s", flush=True)

    preds, n_judged, n_changed = [], 0, 0
    for row in test.to_dict("records"):
        cs = ix.cs_map.get(row.get("call_sign"))
        if cs is not None:
            preds.append(ix.ids[cs])
            continue
        q = unicodedata.normalize("NFC", str(row.get("vessel_name") or ""))
        if L.is_hangul(q):
            idx, F = L.query_features(ix, row, L.query_candidates(ix, row))
            s = sum(m.predict(F) for m in ko) / len(ko)
            order = sorted(range(len(idx)), key=lambda j: (-float(s[j]), idx[j]))
            allowed = [j for j in order if idx[j] not in train_targets] or order
            gap = float(s[allowed[0]] - s[allowed[1]]) if len(allowed) > 1 else 99.0
            pick = allowed[0]
            if gap < GAP and len(allowed) > 1:
                top = allowed[:K]
                query = dict(row, vessel_name=q)
                lp = judge.score(query, [crec[idx[j]] for j in top])
                comb = lp + BETA * np.array([float(s[j]) for j in top])
                pick = top[int(np.argmax(comb))]
                n_judged += 1
                n_changed += pick != allowed[0]
            preds.append(ix.ids[idx[pick]])
            continue
        idx, F65 = L65.query_features(ix, row, L65.query_candidates(ix, row))
        score = 1.5 * sum(m.predict(F65) for m in en65) + 1.0 * sum(m.predict(F65[:, :60]) for m in s62)
        order = sorted(range(len(idx)), key=lambda j: (-float(score[j]), idx[j]))
        preds.append(ix.ids[next((idx[j] for j in order if idx[j] not in train_targets), idx[order[0]])])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": test["sample_id"], "target_vessel_id": preds}).to_csv(out, index=False)
    print(f"Saved {len(preds)} predictions to {out}; judged {n_judged}, changed {n_changed}; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    run()
