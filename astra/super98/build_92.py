"""Full-data production builder for submissions/92_submit.zip.
Supports multi-process lock-coordinated feature chunk extraction and full 6-seed training
for both Super98 (84 features) and Clean18 (78 features).
"""
from __future__ import annotations
from pathlib import Path
import argparse
import fcntl
import gc
import json
import os
import pickle
import shutil
import sys
import time
import zipfile

os.environ["KO_DICT"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FULL_DIR = HERE / "full_train"
SRC_DIR = ROOT / "astra" / "incumbent97" / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(HERE))

import cv
import opus_lib as B
from super_features_92 import (
    FEATURE_NAMES,
    CLEAN18_KEEP_IDX,
    CLEAN18_KEEP_NAMES,
    SuperFeatureExtractor,
)

SEEDS = [0, 5000, 1000, 2000, 6000, 7000]
PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    eval_at=[1],
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambdarank_truncation_level=10,
    verbose=-1,
    num_threads=4,
)

SCRIPT_92 = '''"""DACON inference entry point (92_submit: 18-model Korean [72 + Super98 + Clean18] + Compat-Jamo Composer + 12-model English [65 + 62])."""
from __future__ import annotations
from pathlib import Path
import os
import pickle
import sys
import unicodedata

if os.environ.get("PYTHONHASHSEED") is None:
    os.execve(sys.executable, [sys.executable] + sys.argv, dict(os.environ, PYTHONHASHSEED="0"))

os.environ["KO_DICT"] = "1"
import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "model"))
import opus_lib as L
import opus_lib_65 as L65
from super_features import (
    CLEAN18_KEEP_IDX,
    SuperFeatureExtractor,
    compose_compat_jamo,
)


def run(test_path="data/test.csv", catalog_path="data/vessel_catalog.csv", out="output/submission.csv"):
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    with open(HERE / "model" / "stats.pkl", "rb") as f:
        stats = pickle.load(f)

    ix = L.Index(catalog, None)
    ix.cs_map = {k: ix.id2i[v] for k, v in stats["cs_map"].items() if v in ix.id2i}
    ix.flag_p, ix.type_p = stats["flag_p"], stats["type_p"]
    ix.train_targets = set(ix.cs_map.values())
    ix.ko = L.KA.KoMatcher(catalog, stats["ko_pairs"])
    ix.fit_alias(stats["alias_rows"])

    sfe = SuperFeatureExtractor(
        ix,
        catalog,
        None,
        L.FEATURES,
        fit_vids=set(stats["fit_vids"]),
        type_unit_counts=stats["type_unit_counts"],
        stateless=True,
    )

    models_72 = [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob("ranker_*.txt"))]
    models_sup98 = [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob("sup98_*.txt"))]
    models_cln18 = [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob("cln18_*.txt"))]
    models_65 = [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob("en65_*.txt"))]
    models_62 = [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob("s62_*.txt"))]

    train_targets = {ix.id2i[v] for v in stats["cs_map"].values() if v in ix.id2i}
    keep_idx = np.asarray(CLEAN18_KEEP_IDX, dtype=np.int32)
    preds = []

    for row in test.to_dict("records"):
        cs = ix.cs_map.get(row.get("call_sign"))
        if cs is not None:
            preds.append(ix.ids[cs])
            continue
        raw_q = unicodedata.normalize("NFC", str(row.get("vessel_name") or ""))
        q = compose_compat_jamo(raw_q)
        if q != raw_q:
            row = dict(row)
            row["vessel_name"] = q
        if L.is_hangul(q):
            cand = L.query_candidates(ix, row)
            idx, F60 = L.query_features(ix, row, cand)
            E24 = sfe.compute_query(row, idx, F60)
            F84 = np.column_stack((F60, E24))
            F78 = np.column_stack((F60, E24[:, keep_idx]))
            s_72 = sum(m.predict(F60) for m in models_72)
            s_sup = sum(m.predict(F84) for m in models_sup98)
            s_cln = sum(m.predict(F78) for m in models_cln18)
            score = 0.34 * s_72 + 0.33 * s_sup + 0.33 * s_cln
        else:
            idx, F65 = L65.query_features(ix, row, L65.query_candidates(ix, row))
            F60 = F65[:, :60]
            score = 1.5 * sum(m.predict(F65) for m in models_65) + 1.0 * sum(m.predict(F60) for m in models_62)
        order = sorted(range(len(idx)), key=lambda j: (-float(score[j]), idx[j]))
        best = next((idx[j] for j in order if idx[j] not in train_targets), idx[order[0]])
        preds.append(ix.ids[best])

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": test["sample_id"], "target_vessel_id": preds}).to_csv(out, index=False)
    print(f"Saved {len(preds)} predictions to {out}")


if __name__ == "__main__":
    run()
'''


def dump_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str))
    tmp.replace(p)


def try_lock(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    f = p.open("a")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return f
    except BlockingIOError:
        f.close()
        return None


def build_full_context():
    tr, cat = cv.load()
    ix = B.Index(cat, tr)
    llm_csv = ROOT / "shared" / "experiments" / "pinned" / "build70" / "llm_korean_scored.csv"
    llm = B.llm_korean_rows(tr, llm_csv)
    ix.ko = B.KA.KoMatcher(cat, pd.concat([tr, llm], ignore_index=True))
    sfe = SuperFeatureExtractor(ix, cat, tr, B.FEATURES, stateless=False)
    return tr, cat, ix, llm, sfe


def prepare_blocks():
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = FULL_DIR / "plan.json"
    if plan_path.exists():
        print("FULL_BLOCKS_ALREADY_PREPARED", flush=True)
        return
    tr, cat, ix, llm, _ = build_full_context()
    ko = tr[tr["vessel_name"].map(B.is_hangul)]
    blocks_dir = FULL_DIR / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)

    fixed = {
        "real": tr,
        "nogt": ko.assign(gross_tonnage=np.nan, international_gross_tonnage=np.nan),
        "llm": llm,
    }
    for name, df in fixed.items():
        dump_json(blocks_dir / f"{name}.json", df.to_dict("records"))

    plans = {}
    for seed in SEEDS:
        aug = B.augment_korean(tr, seed)
        noisy = aug.iloc[len(tr) + len(ko):].copy()
        corrupt = B.augment_corrupt(tr, seed)
        rename = B.augment_rename(tr, seed)
        for kind, df in [("noisy", noisy), ("corrupt", corrupt), ("rename", rename)]:
            dump_json(blocks_dir / f"seed{seed}_{kind}.json", df.to_dict("records"))
        order = ["real", "nogt", f"seed{seed}_noisy", f"seed{seed}_corrupt", f"seed{seed}_rename", "llm"]
        tot_q = sum(len(json.loads((blocks_dir / f"{b}.json").read_text())) for b in order)
        plans[str(seed)] = {"order": order, "queries": tot_q}

    dump_json(plan_path, {"seeds": plans, "features": B.FEATURES + FEATURE_NAMES})
    print(f"PREPARED_FULL_BLOCKS queries_per_seed={plans['0']['queries']}", flush=True)


def all_block_names() -> list[str]:
    names = ["real", "nogt", "llm"]
    for s in SEEDS:
        names.extend([f"seed{s}_noisy", f"seed{s}_corrupt", f"seed{s}_rename"])
    return names


def extract_worker():
    prepare_blocks()
    tr, cat, ix, llm, sfe = build_full_context()
    blocks_dir = FULL_DIR / "blocks"
    feat_dir = FULL_DIR / "features84"
    feat_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for name in all_block_names():
        rows = json.loads((blocks_dir / f"{name}.json").read_text())
        bdir = feat_dir / name
        bdir.mkdir(parents=True, exist_ok=True)
        for lo in range(0, len(rows), 500):
            dst = bdir / f"{lo:06d}.npz"
            if dst.exists():
                continue
            lk = try_lock(dst.with_suffix(".lock"))
            if lk is None:
                continue
            try:
                if dst.exists():
                    continue
                X84_list, y_list, g_list = [], [], []
                chunk = rows[lo:lo + 500]
                for row in chunk:
                    idx, F60 = B.query_features(ix, row, B.query_candidates(ix, row))
                    E24 = sfe.compute_query(row, idx, F60)
                    t = ix.id2i[row["target_vessel_id"]]
                    X84_list.append(np.column_stack((F60, E24)))
                    y_list.append(np.array([i == t for i in idx], dtype=np.int8))
                    g_list.append(len(idx))
                tmp = dst.with_suffix(".tmp.npz")
                np.savez_compressed(
                    tmp,
                    X84=np.concatenate(X84_list),
                    y=np.concatenate(y_list),
                    groups=np.asarray(g_list, dtype=np.int32),
                )
                tmp.replace(dst)
                print(f"EXTRACTED {name} {lo + len(chunk)}/{len(rows)} elapsed={time.time() - t0:.1f}s", flush=True)
            finally:
                lk.close()


def verify_all_extracted():
    blocks_dir = FULL_DIR / "blocks"
    feat_dir = FULL_DIR / "features84"
    for name in all_block_names():
        rows = json.loads((blocks_dir / f"{name}.json").read_text())
        for lo in range(0, len(rows), 500):
            p = feat_dir / name / f"{lo:06d}.npz"
            assert p.exists(), f"Missing chunk: {p}"


def train_seed(seed_idx: int, seed: int):
    verify_all_extracted()
    plan = json.loads((FULL_DIR / "plan.json").read_text())
    order = plan["seeds"][str(seed)]["order"]
    blocks_dir = FULL_DIR / "blocks"
    feat_dir = FULL_DIR / "features84"
    models_dir = FULL_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    sup_path = models_dir / f"sup98_{seed_idx}.txt"
    cln_path = models_dir / f"cln18_{seed_idx}.txt"
    if sup_path.exists() and cln_path.exists():
        print(f"SEED_ALREADY_TRAINED idx={seed_idx} seed={seed}", flush=True)
        return

    paths = []
    for name in order:
        rows = json.loads((blocks_dir / f"{name}.json").read_text())
        for lo in range(0, len(rows), 500):
            paths.append(feat_dir / name / f"{lo:06d}.npz")

    groups, ys, count = [], [], 0
    for p in paths:
        with np.load(p) as z:
            groups.extend(z["groups"].tolist())
            ys.append(z["y"])
            count += len(z["y"])
    y = np.concatenate(ys)

    if not sup_path.exists():
        t0 = time.time()
        mat84_path = FULL_DIR / f"mat84_seed{seed}.npy"
        X84 = np.lib.format.open_memmap(mat84_path, mode="w+", dtype=np.float32, shape=(count, 84))
        off = 0
        for p in paths:
            with np.load(p) as z:
                arr = z["X84"]
                n = len(arr)
                X84[off:off + n] = arr
                off += n
        X84.flush()
        print(f"TRAIN_SUP98 idx={seed_idx} seed={seed} queries={len(groups)} cands={count}", flush=True)
        m_sup = lgb.train(PARAMS, lgb.Dataset(X84, y, group=groups, feature_name=B.FEATURES + FEATURE_NAMES), 600)
        tmp = sup_path.with_suffix(".tmp")
        m_sup.save_model(str(tmp))
        tmp.replace(sup_path)
        del X84
        gc.collect()
        mat84_path.unlink(missing_ok=True)
        print(f"DONE_SUP98 idx={seed_idx} seed={seed} in {time.time() - t0:.1f}s", flush=True)

    if not cln_path.exists():
        t0 = time.time()
        keep_cols = np.asarray(list(range(60)) + [60 + i for i in CLEAN18_KEEP_IDX], dtype=np.int32)
        mat78_path = FULL_DIR / f"mat78_seed{seed}.npy"
        X78 = np.lib.format.open_memmap(mat78_path, mode="w+", dtype=np.float32, shape=(count, 78))
        off = 0
        for p in paths:
            with np.load(p) as z:
                arr = z["X84"][:, keep_cols]
                n = len(arr)
                X78[off:off + n] = arr
                off += n
        X78.flush()
        print(f"TRAIN_CLN18 idx={seed_idx} seed={seed} queries={len(groups)} cands={count}", flush=True)
        m_cln = lgb.train(PARAMS, lgb.Dataset(X78, y, group=groups, feature_name=B.FEATURES + CLEAN18_KEEP_NAMES), 600)
        tmp = cln_path.with_suffix(".tmp")
        m_cln.save_model(str(tmp))
        tmp.replace(cln_path)
        del X78
        gc.collect()
        mat78_path.unlink(missing_ok=True)
        print(f"DONE_CLN18 idx={seed_idx} seed={seed} in {time.time() - t0:.1f}s", flush=True)


def package_92():
    models_dir = FULL_DIR / "models"
    for i in range(6):
        assert (models_dir / f"sup98_{i}.txt").exists(), f"Missing sup98_{i}.txt"
        assert (models_dir / f"cln18_{i}.txt").exists(), f"Missing cln18_{i}.txt"

    out_dir = ROOT / "submit_92_build"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "model").mkdir(parents=True)

    with zipfile.ZipFile(ROOT / "submissions" / "79_submit.zip") as z79:
        for name in z79.namelist():
            if name.startswith("model/"):
                (out_dir / name).write_bytes(z79.read(name))

    with zipfile.ZipFile(ROOT / "submissions" / "80_submit.zip") as z80:
        for i in range(6):
            name = f"model/s62_{i}.txt"
            (out_dir / name).write_bytes(z80.read(name))

    for i in range(6):
        shutil.copy2(models_dir / f"sup98_{i}.txt", out_dir / "model" / f"sup98_{i}.txt")
        shutil.copy2(models_dir / f"cln18_{i}.txt", out_dir / "model" / f"cln18_{i}.txt")

    shutil.copy2(HERE / "super_features_92.py", out_dir / "model" / "super_features.py")

    tr, cat, ix, llm, sfe = build_full_context()
    with open(out_dir / "model" / "stats.pkl", "rb") as f:
        stats = pickle.load(f)
    extra_stats = sfe.export_stats()
    stats["fit_vids"] = extra_stats["fit_vids"]
    stats["type_unit_counts"] = extra_stats["type_unit_counts"]
    with open(out_dir / "model" / "stats.pkl", "wb") as f:
        pickle.dump(stats, f, protocol=4)

    (out_dir / "script.py").write_text(SCRIPT_92, encoding="utf-8")
    (out_dir / "requirements.txt").write_text("# uses preinstalled lightgbm, rapidfuzz, numpy, pandas\n", encoding="utf-8")

    zp = ROOT / "submissions" / "92_submit.zip"
    assert not zp.exists(), f"Refusing to overwrite {zp}"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                zf.write(p, p.relative_to(out_dir))
    shutil.rmtree(out_dir)
    print(f"PACKAGED {zp} ({zp.stat().st_size / 1e6:.1f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prepare", "extract", "train", "package"], required=True)
    ap.add_argument("--seed-idx", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    args = ap.parse_args()

    if args.mode == "prepare":
        prepare_blocks()
    elif args.mode == "extract":
        extract_worker()
    elif args.mode == "train":
        for idx in args.seed_idx:
            train_seed(idx, SEEDS[idx])
    elif args.mode == "package":
        package_92()


if __name__ == "__main__":
    main()
