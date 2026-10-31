# author: opus
"""Stage-1 out-of-fold scores for the Korean top-k reranker (stage 2).

For each outer fold (vessel-disjoint, cv.folds) this reuses Astra's matched-recipe 79 Korean
pipeline (astra/incumbent97.py: fit-only dictionary, LLM rows and augmentation blocks), trains the
seed-0 ranker on the fit vessels, then scores Korean queries built from the held-out vessels.
For every query it keeps the top-K allowed candidates (fit targets excluded, as production does)
with their 60 stage-1 features and the stage-1 score.

usage: python3 opus/stage2/oof.py --fold 0      (prepare + train + score; fold2 reuses Astra's model)
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "astra"))
import incumbent97 as I  # noqa: E402  (pins opus_lib/KoMatcher bytes to the 79 ZIP)

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

B = I.B
OUT = ROOT / "opus" / "stage2" / "oof"
TOPK = 20
FIELDS = I.MODEL_FIELDS


def train_seed(fold, seed):
    """Astra's incumbent97.train without its fold2-only evaluation step."""
    start = time.time()
    tr, cat, fit, va, ix, llm, out = I.context(fold)
    seedout = out / f"seed{seed}"
    seedout.mkdir(exist_ok=True)
    modelpath = seedout / "ranker.txt"
    if modelpath.exists():
        return lgb.Booster(model_file=str(modelpath))
    order = json.loads((out / "plan.json").read_text())["seeds"][str(seed)]["order"]
    for name in order:
        I.cache_block(ix, out, name, start)
    paths = sum(([out / "features" / name / f"{lo:06d}.npz"
                  for lo in range(0, len(json.loads((out / "blocks" / f"{name}.json").read_text())), 500)]
                 for name in order), [])
    groups, ys, count = [], [], 0
    for p in paths:
        with np.load(p) as z:
            groups.extend(z["groups"].tolist())
            ys.append(z["y"])
            count += len(z["y"])
    X = np.lib.format.open_memmap(seedout / "matrix.npy", mode="w+", dtype=np.float32,
                                  shape=(count, len(B.FEATURES)))
    off = 0
    for p in paths:
        with np.load(p) as z:
            X[off:off + len(z["y"])] = z["X"]
            off += len(z["y"])
    y = np.concatenate(ys)
    print("TRAIN", fold, seed, len(groups), len(y), round(time.time() - start), flush=True)
    m = lgb.train(I.PARAMS, lgb.Dataset(X, y, group=groups, feature_name=B.FEATURES), 600)
    m.save_model(str(modelpath.with_suffix(".tmp")))
    modelpath.with_suffix(".tmp").replace(modelpath)
    del X, y
    gc.collect()
    (seedout / "matrix.npy").unlink()
    print("TRAINED", fold, seed, round(time.time() - start), flush=True)
    return m


def heldout_queries(va, fit):
    """Korean queries from held-out vessels: native rows, Astra's old9 stress set and the
    training augmentations (noisy tonnage, corrupt names, renames) applied to held-out rows."""
    ko = va[va.vessel_name.map(B.is_hangul)].copy()
    rows = []
    for r in ko.to_dict("records"):
        rows.append(dict(r, variant="native"))
    reps = ko.drop_duplicates("target_vessel_id")
    tree = I.ast.parse((ROOT / "astra/ranker84.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, I.ast.FunctionDef) and n.name == "queries")
    ns = {"B": B, "np": np}
    exec(compile(I.ast.Module(body=[fn], type_ignores=[]), "old_queries", "exec"), ns)
    rows += ns["queries"](reps, fit)
    for seed in (0, 1):
        aug = B.augment_korean(ko, seed).iloc[len(ko):]
        rows += [dict(r, variant=f"aug_tonnage{seed}") for r in aug.to_dict("records")]
        rows += [dict(r, variant=f"aug_corrupt{seed}") for r in B.augment_corrupt(ko, seed).to_dict("records")]
        rows += [dict(r, variant=f"aug_rename{seed}") for r in B.augment_rename(ko, seed).to_dict("records")]
    out = []
    for i, r in enumerate(rows):
        q = {k: r.get(k) for k in FIELDS}
        q["call_sign"] = "UNSEEN"
        if not B.is_hangul(str(q.get("vessel_name") or "")):
            continue
        q["target_vessel_id"] = r["target_vessel_id"]
        q["variant"] = r["variant"]
        q["qid"] = f"{r['variant']}::{r['sample_id']}::{i}"
        out.append(q)
    return out


def score_fold(fold, seeds):
    tr, cat, fit, va, ix, llm, out = I.context(fold)
    models = [train_seed(fold, s) for s in seeds]
    rows = heldout_queries(va, fit)
    fit_targets = set(ix.cs_map.values())
    X, S, cand, qi, tgt, meta = [], [], [], [], [], []
    for j, r in enumerate(rows):
        q = {k: r[k] for k in FIELDS}
        idx, F = B.query_features(ix, q, B.query_candidates(ix, q))
        s = sum(m.predict(F, num_threads=2) for m in models) / len(models)
        order = sorted(range(len(idx)), key=lambda k: (-float(s[k]), idx[k]))
        allowed = [k for k in order if idx[k] not in fit_targets] or order
        top = allowed[:TOPK]
        X.append(F[top])
        S.append(s[top])
        cand.extend(idx[k] for k in top)
        qi.extend([j] * len(top))
        t = ix.id2i[r["target_vessel_id"]]
        tgt.append(t)
        meta.append(dict(qid=r["qid"], variant=r["variant"], vessel=r["target_vessel_id"],
                         vessel_name=r["vessel_name"], in_cands=bool(t in idx)))
        if (j + 1) % 1000 == 0:
            print("SCORED", fold, j + 1, len(rows), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"fold{fold}.npz", X=np.concatenate(X).astype(np.float32),
                        s1=np.concatenate(S).astype(np.float32), cand=np.asarray(cand, np.int32),
                        q=np.asarray(qi, np.int32), target=np.asarray(tgt, np.int32))
    pd.DataFrame(meta).to_csv(OUT / f"fold{fold}_queries.csv", index=False)
    print("DONE", fold, len(rows), "queries", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seeds", default="0")
    a = p.parse_args()
    if not (I.BASE / f"fold{a.fold}" / "plan.json").exists():
        I.prepare(a.fold)
    score_fold(a.fold, [int(s) for s in a.seeds.split(",")])
