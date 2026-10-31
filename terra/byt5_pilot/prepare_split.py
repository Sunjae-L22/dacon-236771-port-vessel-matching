# author: terra (Sol)
"""Freeze public-train-only pilot roles; no model training or test access."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "split_v1"
CV = ROOT / "shared/src/cv.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location("pilot_cv", CV)
    cv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cv)
    train, catalog = cv.load()
    inputs = [ROOT / "data/open/train.csv", ROOT / "data/open/vessel_catalog.csv", CV,
              ROOT / "shared/llm_aug/llm_fewshot.csv",
              ROOT / "shared/llm_aug/llm_fewshot_sets.csv"]
    prompt_ids = set().union(*(set(pd.read_csv(p).vessel_id) for p in inputs[-2:]))
    roles = train[["sample_id", "target_vessel_id"]].copy()
    roles["original_fold"] = cv.folds(train, seed=42)
    roles["prompt_example"] = roles.target_vessel_id.isin(prompt_ids)
    roles["role"] = roles.original_fold.map({0: "eval", 1: "tune", 2: "train", 3: "train", 4: "train"})
    roles["moved_to_train"] = roles.prompt_example & roles.role.ne("train")
    roles.loc[roles.prompt_example, "role"] = "train"
    roles["is_hangul"] = train.vessel_name.map(cv.is_hangul)
    ids = {r: set(roles.loc[roles.role.eq(r), "target_vessel_id"]) for r in ("train", "tune", "eval")}
    assert all(not ids[a] & ids[b] for a, b in (("train", "tune"), ("train", "eval"), ("tune", "eval")))
    assert not (ids["tune"] | ids["eval"]) & prompt_ids
    assert train.sample_id.is_unique
    assert set(train.target_vessel_id) <= set(catalog.vessel_id)
    OUT.mkdir(exist_ok=True)
    roles.to_csv(OUT / "rows.csv", index=False)
    vessels = roles[["target_vessel_id", "original_fold", "prompt_example", "role", "moved_to_train"]].drop_duplicates()
    assert vessels.target_vessel_id.is_unique
    vessels.to_csv(OUT / "vessels.csv", index=False)
    report = {
        "status": "SPLIT_ONLY_NO_TRAINING_NO_GPU_NO_ACCURACY",
        "scheme": "seed42 vessel-group folds 2/3/4 train, 1 tune, 0 eval; all v1/v2 prompt-example vessels forced into train",
        "prompt_example_vessels_total": len(prompt_ids),
        "prompt_example_vessels_in_train_csv": len(prompt_ids & set(train.target_vessel_id)),
        "moved_vessels": int(vessels.moved_to_train.sum()),
        "counts": {r: {"rows": int(roles.role.eq(r).sum()), "vessels": len(ids[r]),
            "hangul_rows": int((roles.role.eq(r) & roles.is_hangul).sum()),
            "hangul_vessels": int(roles.loc[roles.role.eq(r) & roles.is_hangul, "target_vessel_id"].nunique())}
            for r in ids},
        "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
        "artifacts_sha256": {p.name: sha(p) for p in (OUT / "rows.csv", OUT / "vessels.csv")},
        "preparation_code_sha256": sha(Path(__file__)),
        "limitations": ["3-way holdout, not 5-fold OOF", "Prompt-example vessels excluded from evaluation population",
            "A conventional fold0 OOF ranker trains on tune fold1, so cannot be used for this fusion pilot",
            "LLM generation scores must be recomputed using only role=train; this script does not select LLM pairs"],
    }
    (OUT / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
