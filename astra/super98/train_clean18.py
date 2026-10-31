"""Train Super98-Clean18 (78 features) directly from already-cached Super98 extra_features npz files
by dropping the 6 root-dominating/over-splitting columns:
  12: unexplained_q_ratio
  15: gt_win_zscore
  16: gt_win_broad_best
  17: gt_win_broad_margin
  19: within_gt_rank
  20: meta_compat_score
"""
from __future__ import annotations
from pathlib import Path
import gc
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
sys.path.insert(0, str(ROOT / "astra"))
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(HERE))

import incumbent97 as BASE
from features import FEATURE_NAMES

DROP_COLS = {
    "unexplained_q_ratio",
    "gt_win_zscore",
    "gt_win_broad_best",
    "gt_win_broad_margin",
    "within_gt_rank",
    "meta_compat_score",
}
KEEP_IDX = [i for i, f in enumerate(FEATURE_NAMES) if f not in DROP_COLS]
KEEP_NAMES = [FEATURE_NAMES[i] for i in KEEP_IDX]
N_KEEP = len(KEEP_IDX)
N_TOTAL = 60 + N_KEEP
assert N_KEEP == 18 and N_TOTAL == 78

KINDS = ["old9", "benchmark", "llm_holdout"]


def ordered_chunks(out: Path, plan: dict, seed: int):
    paths = []
    for name in plan["seeds"][str(seed)]["order"]:
        rowpath = out / "blocks" / f"{name}.json"
        rows = json.loads(rowpath.read_text())
        for lo in range(0, len(rows), 500):
            path = out / "features" / name / f"{lo:06d}.npz"
            paths.append((name, path, rowpath, lo, min(500, len(rows) - lo)))
    return paths


def main():
    fold = 2
    out = BASE_DIR / f"fold{fold}"
    plan = json.loads((out / "plan.json").read_text())
    ids = pd.read_csv(ROOT / "data/open/vessel_catalog.csv", usecols=["vessel_id"]).vessel_id.to_numpy()

    seeds = [0, 5000, 1000, 2000, 6000, 7000]
    for seed in seeds:
        seedout = HERE / f"fold{fold}_clean18" / f"seed{seed}"
        seedout.mkdir(parents=True, exist_ok=True)
        modelpath = seedout / "ranker.txt"
        if not modelpath.exists():
            ordered = ordered_chunks(out, plan, seed)
            groups, ys, count = [], [], 0
            for _, path, _, _, _ in ordered:
                with np.load(path) as z:
                    groups.extend(z["groups"].tolist())
                    ys.append(z["y"])
                    count += len(z["y"])
            matrix_path = seedout / f"matrix{N_TOTAL}.npy"
            X = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float32, shape=(count, N_TOTAL))
            offset = 0
            for block, path, _, lo, size in ordered:
                extra_path = HERE / f"fold{fold}" / "train-block" / block / path.name
                with np.load(path) as z, np.load(extra_path) as e:
                    F = z["X"]
                    E = e["extra_features"][:, KEEP_IDX]
                    n = len(F)
                    X[offset:offset + n, :60] = F
                    X[offset:offset + n, 60:] = E
                    offset += n
            X.flush()
            y = np.concatenate(ys)
            params = dict(BASE.PARAMS)
            params["num_threads"] = 4
            t0 = time.time()
            print(f"CLEAN18_TRAIN seed={seed} queries={len(groups)} cands={len(y)} features={N_TOTAL}", flush=True)
            model = lgb.train(params, lgb.Dataset(X, y, group=groups, feature_name=plan["features"] + KEEP_NAMES), 600)
            print(f"CLEAN18_TRAIN_DONE seed={seed} in {time.time() - t0:.1f}s", flush=True)
            tmp = modelpath.with_suffix(".tmp")
            model.save_model(str(tmp))
            tmp.replace(modelpath)
            del X, y, groups
            gc.collect()
            matrix_path.unlink(missing_ok=True)
        else:
            model = lgb.Booster(model_file=str(modelpath))

        for kind in KINDS:
            paths = [p for p in out.glob(f"eval_{kind}_*.npz") if ".tmp." not in p.name]
            extra_path = HERE / f"fold{fold}" / "eval" / kind / paths[0].name
            rows = json.loads((out / f"eval_{kind}_queries.json").read_text())
            with np.load(paths[0]) as z, np.load(extra_path) as e:
                F = z["X"]
                E = e["extra_features"][:, KEEP_IDX]
                scores = np.empty(len(F), dtype=np.float64)
                for lo in range(0, len(F), 100000):
                    scores[lo:lo + 100000] = model.predict(np.column_stack((F[lo:lo + 100000], E[lo:lo + 100000])), num_threads=4)
                np.save(seedout / f"{kind}_scores.npy", scores)
                d = BASE.predict_rows(scores, z, rows, ids, f"clean18_aug{seed}")
                print(f"CLEAN18_EVAL seed={seed} {kind} acc={d['correct'].mean():.4f} ({int(d['correct'].sum())}/{len(d)})", flush=True)

    print("\n=== SUMMARY COMPARISON (all 6 seeds) ===")
    for kind in KINDS:
        paths = [p for p in out.glob(f"eval_{kind}_*.npz") if ".tmp." not in p.name]
        rows = json.loads((out / f"eval_{kind}_queries.json").read_text())
        base6 = sum(np.load(out / f"seed{s}" / f"{kind}_scores.npy") for s in seeds) / 6.0
        sup24 = sum(np.load(HERE / f"fold{fold}" / f"seed{s}" / f"{kind}_scores.npy") for s in seeds) / 6.0
        cln18 = sum(np.load(HERE / f"fold{fold}_clean18" / f"seed{s}" / f"{kind}_scores.npy") for s in seeds) / 6.0
        with np.load(paths[0]) as z:
            b6_df = BASE.predict_rows(base6, z, rows, ids, "b6")
            b6 = int(b6_df["correct"].sum())
            s24 = int(BASE.predict_rows(sup24, z, rows, ids, "s24")["correct"].sum())
            c18 = int(BASE.predict_rows(cln18, z, rows, ids, "c18")["correct"].sum())
            bl_c18 = int(BASE.predict_rows(0.5 * base6 + 0.5 * cln18, z, rows, ids, "bl_c18")["correct"].sum())
            bl_3way = int(BASE.predict_rows(0.34 * base6 + 0.33 * cln18 + 0.33 * sup24, z, rows, ids, "bl_3way")["correct"].sum())
            print(f"[{kind:11s}] N={len(rows):4d} | base6={b6} | sup24={s24} | clean18={c18} | base6+clean18={bl_c18} ({bl_c18-b6:+d}) | 3way={bl_3way} ({bl_3way-b6:+d})")


if __name__ == "__main__":
    main()
