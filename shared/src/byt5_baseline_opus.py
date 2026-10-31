# author: opus
"""Single-seed baseline ranker for Sol's ByT5 3-way pilot (terra/byt5_pilot/split_v1).

Everything is learned from role=train rows only: call-sign map, flag/type stats, KoMatcher dictionary,
augmentation, and the LLM rows (train-role vessels only, 0.70 filter re-scored by a matcher that saw
train-role real pairs only). Recipe is 31/33's: pinned build31 code, KO_DICT=1, LLM v1, aug seed 0, 600 rounds.
For every Korean tune/eval row it writes the natural top-20 candidates with ranker scores (no forced target).

usage: KO_DICT=1 PYTHONHASHSEED=0 python3 shared/src/byt5_baseline_opus.py
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "shared" / "experiments" / "pinned" / "build31"
sys.path.insert(0, str(CODE))
assert os.environ.get("KO_DICT") == "1", "33 recipe uses KO_DICT=1"
import cv  # noqa: E402
import opus_lib as L  # noqa: E402
from opus_cv import PARAMS  # noqa: E402

SPLIT = ROOT / "terra" / "byt5_pilot" / "split_v1" / "rows.csv"
OUT = ROOT / "shared" / "experiments" / "byt5_baseline"
TOP = 20


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    roles = pd.read_csv(SPLIT).set_index("sample_id")["role"]
    train, catalog = cv.load()
    train = train.assign(role=train["sample_id"].map(roles))
    assert train["role"].notna().all()
    tr = train[train["role"] == "train"].drop(columns="role").reset_index(drop=True)

    ix = L.Index(catalog, tr)
    gen = pd.read_csv(ROOT / "shared" / "llm_aug" / "llm_korean_scored.csv", encoding="utf-8-sig")
    gen = gen[gen["vessel_id"].isin(set(tr["target_vessel_id"]))].copy()
    gen["score"] = [float(ix.ko.score(n, ix.id2i[v])["ko_combined_score"]) for n, v in zip(gen["ko"], gen["vessel_id"])]
    llm_csv = OUT / "llm_scored_trainrole.csv"
    gen.to_csv(llm_csv, index=False, encoding="utf-8-sig")
    llm = L.llm_korean_rows(tr, llm_csv)
    ix.ko = L.KA.KoMatcher(catalog, pd.concat([tr, llm], ignore_index=True))
    print(f"train rows {len(tr)}, LLM rows {len(llm)}", flush=True)

    X, y, g, _ = L.build(ix, pd.concat([L.augment_korean(tr, 0), L.augment_corrupt(tr, 0), L.augment_rename(tr, 0), llm],
                                       ignore_index=True), True)
    model = lgb.train(PARAMS, lgb.Dataset(X, y, group=g, feature_name=L.FEATURES), 600)
    model.save_model(str(OUT / "ranker.txt"))
    print(f"trained in {time.time() - t0:.0f}s", flush=True)

    train_targets = {ix.id2i[v] for v in tr["target_vessel_id"] if v in ix.id2i}
    queries = train[train["role"].isin(["tune", "eval"]) & train["vessel_name"].map(L.is_hangul)]
    rows = []
    for r in queries.to_dict("records"):
        idx, F = L.query_features(ix, r, L.query_candidates(ix, r))
        s = model.predict(F)
        order = sorted(range(len(idx)), key=lambda j: (-s[j], idx[j]))[:TOP]
        for rank, j in enumerate(order, 1):
            rows.append(dict(query_sample_id=r["sample_id"], orig_sample_id=r["sample_id"], variant="orig",
                             role=r["role"], vessel_name=r["vessel_name"], target_vessel_id=r["target_vessel_id"],
                             rank=rank, candidate_vessel_id=ix.ids[idx[j]], ranker_score=float(s[j]),
                             is_target=ix.ids[idx[j]] == r["target_vessel_id"],
                             cand_is_train_role_target=idx[j] in train_targets, n_candidates=len(idx)))
    cand = pd.DataFrame(rows)
    cand.to_csv(OUT / "candidates_top20_orig.csv", index=False, encoding="utf-8-sig")

    summary = {}
    for role, d in cand.groupby("role"):
        top1 = d[d["rank"] == 1]
        excl = d[~d["cand_is_train_role_target"]].sort_values(["query_sample_id", "rank"]).groupby("query_sample_id").head(1)
        summary[role] = {"queries": int(d["query_sample_id"].nunique()), "acc@1": round(float(top1["is_target"].mean()), 4),
                         "acc@1_excluding_train_role_targets": round(float(excl["is_target"].mean()), 4),
                         "recall@20": round(float(d.groupby("query_sample_id")["is_target"].any().mean()), 4)}
    manifest = {"status": "BASELINE_ORIG_ONLY", "recipe": "build31 code, KO_DICT=1, LLM v1 (train-role re-scored >=0.70), aug seed 0, 600 rounds, 1 model",
                "summary": summary, "llm_rows": len(llm), "train_rows": len(tr), "seconds": round(time.time() - t0),
                "sha256": {"split rows.csv": sha(SPLIT), "opus_lib.py": sha(CODE / "opus_lib.py"),
                           "ko_translit_argon.py": sha(CODE / "ko_translit_argon.py"), "this script": sha(__file__),
                           "ranker.txt": sha(OUT / "ranker.txt"), "llm_scored_trainrole.csv": sha(llm_csv),
                           "candidates_top20_orig.csv": sha(OUT / "candidates_top20_orig.csv")}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
