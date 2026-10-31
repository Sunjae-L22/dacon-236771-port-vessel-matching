"""Super98 matched training & evaluation on outer_fold=2 over immutable incumbent97 caches.
Appends 24 additive features (total 84 features) without changing candidate pools or the first 60 features.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import fcntl
import gc
import hashlib
import json
import os
import sys
import time

os.environ["KO_DICT"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE_DIR = ROOT / "astra" / "incumbent97"
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(ROOT / "astra"))
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(HERE))

import incumbent97 as BASE
import cv
import opus_lib as B
from features import FEATURE_NAMES, SuperFeatureExtractor

SEEDS = [0, 5000, 1000, 2000, 6000, 7000]
KINDS = ["old9", "benchmark", "llm_holdout"]
N_EXTRA = len(FEATURE_NAMES)
N_TOTAL = 60 + N_EXTRA


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def build_context(fold: int):
    out = BASE_DIR / f"fold{fold}"
    plan = json.loads((out / "plan.json").read_text())
    assert len(plan["features"]) == 60 and plan["features"] == B.FEATURES
    for name, sha in plan["source"]["sources"].items():
        assert file_hash(SRC_DIR / name) == sha, f"Source changed: {name}"
    tr, cat = cv.load()
    fit = tr[cv.folds(tr) != fold]
    va = tr[cv.folds(tr) == fold]
    assert not (set(fit.target_vessel_id) & set(va.target_vessel_id))
    support = json.loads((out / "dictionary_support.json").read_text())
    B.KA._UNIVERSAL_KO_SUBWORDS = [(r["ko"], r["en"]) for r in support["retained"]]
    assert len(B.KA._UNIVERSAL_KO_SUBWORDS) == plan["dictionary_retained"]
    ix = B.Index(cat, fit)
    llm = B.llm_korean_rows(fit, out / "llm_fit.csv")
    assert len(llm) == plan["llm_rows"]
    ix.ko = B.KA.KoMatcher(cat, pd.concat([fit, llm], ignore_index=True))
    sfe = SuperFeatureExtractor(ix, cat, fit, B.FEATURES)
    provenance = {
        "baseline_plan_sha256": file_hash(out / "plan.json"),
        "feature_code_sha256": file_hash(HERE / "features.py"),
        "feature_names": FEATURE_NAMES,
    }
    return ix, sfe, out, plan, provenance, set(fit.target_vessel_id), set(va.target_vessel_id)


def ordered_chunks(out: Path, plan: dict, seed: int):
    paths = []
    order = plan["seeds"][str(seed)]["order"]
    for name in order:
        rowpath = out / "blocks" / f"{name}.json"
        rows = json.loads(rowpath.read_text())
        for lo in range(0, len(rows), 500):
            path = out / "features" / name / f"{lo:06d}.npz"
            assert path.exists(), f"Missing baseline cache: {path}"
            paths.append((name, path, rowpath, lo, min(500, len(rows) - lo)))
    return paths


def ensure_extra(sfe: SuperFeatureExtractor, out: Path, fold: int, mode: str, category: str, rows: list[dict], paths: list[Path], provenance: dict):
    outputs = []
    for path in paths:
        dst = HERE / f"fold{fold}" / mode / category / path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.with_suffix(".json").exists():
            rep = json.loads(dst.with_suffix(".json").read_text())
            if rep.get("provenance") == provenance and rep.get("source_cache_sha256") == file_hash(path):
                outputs.append(dst)
                continue
        with np.load(path) as z:
            X = z["X"]
            idx = z["idx"]
            groups = z["groups"]
            q_idx = z["query_index"] if "query_index" in z.files else np.arange(len(groups))
            E = np.empty((len(X), N_EXTRA), dtype=np.float32)
            off = 0
            for local, (qi, g) in enumerate(zip(q_idx, groups)):
                E[off:off + g] = sfe.compute_query(rows[int(qi)], idx[off:off + g], X[off:off + g])
                off += g
            assert off == len(X) and np.isfinite(E).all()
            arrays = {k: z[k] for k in ["idx", "groups", "target", "query_index", "allowed"] if k in z.files}
            arrays["extra_features"] = E
            tmp = dst.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **arrays)
            tmp.replace(dst)
            dump(dst.with_suffix(".json"), {
                "provenance": provenance,
                "source_cache_sha256": file_hash(path),
                "output_extra_cache_sha256": file_hash(dst),
                "candidate_rows": int(len(E)),
            })
            print(f"SUPER98_EXTRA {category} {path.name} rows={len(E)}", flush=True)
        outputs.append(dst)
    return outputs


def construct_matrix(fold: int, seed: int, ordered: list, seedout: Path, expected_queries: int):
    groups, ys, count = [], [], 0
    for _, path, _, _, _ in ordered:
        with np.load(path) as z:
            groups.extend(z["groups"].tolist())
            ys.append(z["y"])
            count += len(z["y"])
    assert len(groups) == expected_queries
    matrix_path = seedout / f"matrix{N_TOTAL}.npy"
    X = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float32, shape=(count, N_TOTAL))
    offset = 0
    for block, path, _, lo, size in ordered:
        extra_path = HERE / f"fold{fold}" / "train-block" / block / path.name
        with np.load(path) as z, np.load(extra_path) as e:
            F = z["X"]
            E = e["extra_features"]
            n = len(F)
            assert F.shape == (n, 60) and E.shape == (n, N_EXTRA)
            assert np.array_equal(z["idx"], e["idx"])
            X[offset:offset + n, :60] = F
            X[offset:offset + n, 60:] = E
            offset += n
    X.flush()
    y = np.concatenate(ys)
    assert offset == len(X) == len(y) == sum(groups)
    return X, y, groups, matrix_path


def evaluate_seed(fold: int, seed: int, ix, sfe: SuperFeatureExtractor, out: Path, provenance: dict, model: lgb.Booster):
    seedout = HERE / f"fold{fold}" / f"seed{seed}"
    for kind in KINDS:
        paths = [p for p in out.glob(f"eval_{kind}_*.npz") if ".tmp." not in p.name]
        assert len(paths) == 1
        rows = json.loads((out / f"eval_{kind}_queries.json").read_text())
        extra = ensure_extra(sfe, out, fold, "eval", kind, rows, paths, provenance)[0]
        with np.load(paths[0]) as z, np.load(extra) as e:
            F = z["X"]
            E = e["extra_features"]
            scores = np.empty(len(F), dtype=np.float64)
            for lo in range(0, len(F), 100000):
                scores[lo:lo + 100000] = model.predict(np.column_stack((F[lo:lo + 100000], E[lo:lo + 100000])), num_threads=4)
            dst = seedout / f"{kind}_scores.npy"
            tmp = dst.with_suffix(".tmp.npy")
            np.save(tmp, scores)
            tmp.replace(dst)
            d = BASE.predict_rows(scores, z, rows, ix.ids, f"super98_aug{seed}")
            d.to_csv(seedout / f"{kind}_predictions.csv", index=False)
        print(f"SUPER98_EVAL seed={seed} {kind} acc={d['correct'].mean():.4f} ({int(d['correct'].sum())}/{len(d)})", flush=True)


def aggregate(fold: int):
    out = BASE_DIR / f"fold{fold}"
    own = HERE / f"fold{fold}"
    ids = pd.read_csv(ROOT / "data/open/vessel_catalog.csv", usecols=["vessel_id"]).vessel_id.to_numpy()
    avail_seeds = [s for s in SEEDS if all((own / f"seed{s}" / f"{k}_scores.npy").exists() for k in KINDS)]
    print(f"=== SUPER98 AGGREGATE (available seeds: {avail_seeds}) ===")
    summary = {}
    for kind in KINDS:
        paths = [p for p in out.glob(f"eval_{kind}_*.npz") if ".tmp." not in p.name]
        rows = json.loads((out / f"eval_{kind}_queries.json").read_text())
        score = sum(np.load(own / f"seed{s}" / f"{kind}_scores.npy") for s in avail_seeds) / len(avail_seeds)
        base_score = sum(np.load(out / f"seed{s}" / f"{kind}_scores.npy") for s in avail_seeds) / len(avail_seeds)
        with np.load(paths[0]) as z:
            sup_df = BASE.predict_rows(score, z, rows, ids, f"super98_{len(avail_seeds)}models")
            base_df = BASE.predict_rows(base_score, z, rows, ids, f"base_{len(avail_seeds)}models")
        plus = int((~base_df["correct"] & sup_df["correct"]).sum())
        minus = int((base_df["correct"] & ~sup_df["correct"]).sum())
        print(f"[{kind:11s}] ({len(avail_seeds)} seeds): BASE={int(base_df['correct'].sum())}/{len(base_df)} ({base_df['correct'].mean():.4f}) -> SUPER98={int(sup_df['correct'].sum())}/{len(sup_df)} ({sup_df['correct'].mean():.4f}) | +{plus} -{minus} (net {plus - minus:+d})")
        # Also test 50/50 blend of BASE + SUPER98!
        blend_score = 0.5 * base_score + 0.5 * score
        with np.load(paths[0]) as z:
            blend_df = BASE.predict_rows(blend_score, z, rows, ids, f"blend_{len(avail_seeds)}models")
        b_plus = int((~base_df["correct"] & blend_df["correct"]).sum())
        b_minus = int((base_df["correct"] & ~blend_df["correct"]).sum())
        print(f"[{kind:11s}] (50/50 blend): BASE={int(base_df['correct'].sum())}/{len(base_df)} -> BLEND={int(blend_df['correct'].sum())}/{len(blend_df)} ({blend_df['correct'].mean():.4f}) | +{b_plus} -{b_minus} (net {b_plus - b_minus:+d})")
        summary[kind] = {
            "seeds": avail_seeds,
            "base_correct": int(base_df["correct"].sum()),
            "super98_correct": int(sup_df["correct"].sum()),
            "blend_correct": int(blend_df["correct"].sum()),
            "total": len(base_df),
        }
    dump(own / "summary_report.json", summary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=2)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 5000])
    p.add_argument("--mode", choices=["train", "aggregate"], default="train")
    args = p.parse_args()

    if args.mode == "aggregate":
        aggregate(args.fold)
        return

    t0 = time.time()
    ix, sfe, out, plan, provenance, fit_ids, heldout_ids = build_context(args.fold)
    for seed in args.seeds:
        seedout = HERE / f"fold{args.fold}" / f"seed{seed}"
        seedout.mkdir(parents=True, exist_ok=True)
        modelpath = seedout / "ranker.txt"
        if not modelpath.exists():
            ordered = ordered_chunks(out, plan, seed)
            for block in plan["seeds"][str(seed)]["order"]:
                rows = json.loads((out / "blocks" / f"{block}.json").read_text())
                paths = [p for name, p, _, _, _ in ordered if name == block]
                ensure_extra(sfe, out, args.fold, "train-block", block, rows, paths, provenance)
            X, y, groups, matrix_path = construct_matrix(args.fold, seed, ordered, seedout, plan["seeds"][str(seed)]["queries"])
            params = dict(BASE.PARAMS)
            params["num_threads"] = 4
            print(f"SUPER98_TRAIN seed={seed} queries={len(groups)} cands={len(y)} features={N_TOTAL}", flush=True)
            t_tr = time.time()
            model = lgb.train(params, lgb.Dataset(X, y, group=groups, feature_name=plan["features"] + FEATURE_NAMES), 600)
            print(f"SUPER98_TRAIN_DONE seed={seed} in {time.time() - t_tr:.1f}s", flush=True)
            tmp = modelpath.with_suffix(".tmp")
            model.save_model(str(tmp))
            tmp.replace(modelpath)
            del X, y, groups
            gc.collect()
            matrix_path.unlink(missing_ok=True)
        else:
            model = lgb.Booster(model_file=str(modelpath))
        evaluate_seed(args.fold, seed, ix, sfe, out, provenance, model)
    aggregate(args.fold)
    print(f"Completed in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
