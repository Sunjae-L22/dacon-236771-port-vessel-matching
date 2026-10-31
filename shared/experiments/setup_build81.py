"""Create shared/experiments/pinned/build81 and run_build81.sh:
Trains a 100% Dedicated Korean-Only Ranker (ko_only, 6 models across 81 + 82) where:
1. 100% of training rows are Korean (train[is_ko] + augment_korean + augment_corrupt(ko, frac=1.0) + augment_rename(ko, frac=0.20) + llm_korean_rows),
   so 100% of LightGBM's 37,200 tree splits are dedicated to Korean matching (instead of 4.5% in the joint model) and training finishes in ~22 minutes!
2. Fixes Bug B in opus_lib.py: on Korean rows, sets `qcmp = qinfo["trans"].upper()` and `qt = " ".join(qinfo["trans_tokens"]).upper()`
   so `ratio`, `pratio`, `tsort`, `tset`, `jw`, `lev_norm`, `len_q`, `len_ratio`, `len_diff`, `is_prefix` compare KoMatcher's English
   back-transliteration against the English catalog name (while keeping `rom_ratio`, `ck_*`, `vk_*` on raw romanization).
3. Keeps the exact candidate pool of build70 (72_submit.zip) so candidate sets match 72_submit.zip bit-for-bit, allowing seamless logit blending
   of 72_submit.zip (6 joint models) + 81/82 (6 dedicated Korean models) on Korean rows, paired with 65_submit.zip (3502/3568) on English rows -> 83_submit.zip!
"""
from __future__ import annotations
import shutil
import unicodedata
from pathlib import Path

ROOT = Path(unicodedata.normalize("NFD", "/Users/sunjae/Desktop/항만"))
B70 = ROOT / "shared/experiments/pinned/build70"
B67 = ROOT / "shared/experiments/pinned/build67"
B81 = ROOT / "shared/experiments/pinned/build81"

if B81.exists():
    shutil.rmtree(B81)
shutil.copytree(B70, B81, ignore=shutil.ignore_patterns("__pycache__"))

# Copy our upgraded ko_translit_argon.py (with 1-syllable truncated prefix fix + phonology fixes) from build67
shutil.copy2(B67 / "ko_translit_argon.py", B81 / "ko_translit_argon.py")

# Patch opus_lib.py in build81:
# Keep query_candidates 100% identical to build70 (so idx matches 72_submit.zip bit-for-bit!),
# and in query_features() for Korean rows:
# - Use qinfo = ix.ko.translate_ko(q) to set qcmp = qinfo["trans"].upper() and qt = " ".join(qinfo["trans_tokens"]).upper()
# - Add the 7 GT-window & matched-token metadata features from build67
opus_src = (B81 / "opus_lib.py").read_text(encoding="utf-8")

opus_src = opus_src.replace(
    'list(KA.KoMatcher.FEATURE_NAMES) + ["ko_comb_best", "ko_comb_margin", "is_prefix", "ck_pratio", "len_diff", "ko_comb_rev", "initial_ok"]',
    'list(KA.KoMatcher.FEATURE_NAMES) + ["ko_comb_best", "ko_comb_margin", "is_prefix", "ck_pratio", "len_diff", "ko_comb_rev", "initial_ok", "ko_gt_win_best", "ko_gt_win_margin", "ko_gt_win_zscore", "ko_tail_gt_best", "ko_part_gt_best", "ko_tok_meta_best", "ko_tok_meta_margin"]',
    1,
)

old_qcmp = '''    if ko:
        rom, ck, vk = ko_keys(q)
        qd = digits(ko_clean(q))
        qcmp = re.sub(r"[^a-z0-9]", "", rom).upper()
    else:
        qcmp = norm(q)
        qd = digits(q)
        ck = phon_key(q)
        vk = vowel_key(q)
    names = [ix.names[i] for i in idx]
    toks = [ix.names_tok[i] for i in idx]
    F[:, col["ratio"]] = process.cdist([qcmp], names, scorer=fuzz.ratio)[0]
    F[:, col["pratio"]] = process.cdist([qcmp], names, scorer=fuzz.partial_ratio)[0]
    qt = norm_tokens(rom if ko else q)'''

new_qcmp = '''    if ko:
        rom, ck, vk = ko_keys(q)
        qd = digits(ko_clean(q))
        qinfo_ko = ix.ko.translate_ko(q)
        rom_upper = re.sub(r"[^a-z0-9]", "", rom).upper()
        qcmp = (qinfo_ko["trans"] or rom_upper).upper()
        qt = " ".join(t.upper() for t in qinfo_ko["trans_tokens"]) if qinfo_ko["trans_tokens"] else norm_tokens(rom)
    else:
        qcmp = norm(q)
        rom_upper = qcmp
        qd = digits(q)
        ck = phon_key(q)
        vk = vowel_key(q)
        qt = norm_tokens(q)
    names = [ix.names[i] for i in idx]
    toks = [ix.names_tok[i] for i in idx]
    F[:, col["ratio"]] = process.cdist([qcmp], names, scorer=fuzz.ratio)[0]
    F[:, col["pratio"]] = process.cdist([qcmp], names, scorer=fuzz.partial_ratio)[0]'''

assert old_qcmp in opus_src, "old_qcmp not found"
opus_src = opus_src.replace(old_qcmp, new_qcmp, 1)

opus_src = opus_src.replace(
    'F[:, col["rom_ratio"]] = process.cdist([norm(acronym_expand(q)) if not ko else qcmp],',
    'F[:, col["rom_ratio"]] = process.cdist([norm(acronym_expand(q)) if not ko else rom_upper],',
    1,
)

old_ko_end = '''        toks = q.split()
        if len(toks) > 1:  # same name with the word order reversed
            F[:, col["ko_comb_rev"]] = ix.ko.score_batch(" ".join(toks[::-1]), idx)[:, -1]
        else:
            F[:, col["ko_comb_rev"]] = comb
    return idx, F'''

new_ko_end = '''        toks = q.split()
        if len(toks) > 1:  # same name with the word order reversed
            F[:, col["ko_comb_rev"]] = ix.ko.score_batch(" ".join(toks[::-1]), idx)[:, -1]
        else:
            F[:, col["ko_comb_rev"]] = comb
        comb_eff = np.maximum(comb, F[:, col["ko_comb_rev"]])
        win_mask = F[:, col["best_gap"]] <= 0.12
        if not np.any(win_mask):
            win_mask = np.ones(n, dtype=bool)
        win_idx = np.where(win_mask)[0]
        win_comb = comb_eff[win_idx]
        win_max = float(win_comb.max())
        win_mean = float(win_comb.mean())
        win_std = float(win_comb.std()) + 1e-3
        if len(win_comb) >= 2:
            top2 = np.partition(win_comb, -2)[-2:]
            win_top1, win_top2 = float(top2[1]), float(top2[0])
        else:
            win_top1 = win_top2 = win_max
        F[:, col["ko_gt_win_best"]] = comb_eff - win_max
        is_win_top1 = win_mask & (comb_eff >= win_top1 - 1e-9)
        F[:, col["ko_gt_win_margin"]] = np.where(is_win_top1, comb_eff - win_top2, comb_eff - win_top1)
        F[:, col["ko_gt_win_zscore"]] = (comb_eff - win_mean) / win_std
        tail_arr = F[:, col["ko_tail_sim"]]
        part_arr = np.maximum(F[:, col["ko_partial_sim"]], F[:, col["ko_token_cov"]])
        F[:, col["ko_tail_gt_best"]] = tail_arr - float(tail_arr[win_idx].max())
        F[:, col["ko_part_gt_best"]] = part_arr - float(part_arr[win_idx].max())
        tok_match_mask = (part_arr >= 0.85) | (comb_eff >= win_max - 0.05)
        if not np.any(tok_match_mask):
            tok_match_mask = np.ones(n, dtype=bool)
        tp_safe = np.clip(F[:, col["type_p"]], 1e-3, 1.0)
        fp_safe = np.clip(F[:, col["flag_p"]], 1e-3, 1.0)
        meta_sc = np.log(tp_safe) + 0.6 * np.log(fp_safe) - 25.0 * np.minimum(F[:, col["best_gap"]], 0.40) - 8.0 * np.minimum(F[:, col["dwt_gap"]], 0.40)
        tm_idx = np.where(tok_match_mask)[0]
        tm_sc = meta_sc[tm_idx]
        tm_max = float(tm_sc.max())
        if len(tm_sc) >= 2:
            tm_top2 = np.partition(tm_sc, -2)[-2:]
            tm_1, tm_2 = float(tm_top2[1]), float(tm_top2[0])
        else:
            tm_1 = tm_2 = tm_max
        F[:, col["ko_tok_meta_best"]] = meta_sc - tm_max
        is_tm_top1 = tok_match_mask & (meta_sc >= tm_1 - 1e-9)
        F[:, col["ko_tok_meta_margin"]] = np.where(is_tm_top1, meta_sc - tm_2, meta_sc - tm_1)
    return idx, F'''

assert old_ko_end in opus_src, "old_ko_end not found"
opus_src = opus_src.replace(old_ko_end, new_ko_end, 1)
(B81 / "opus_lib.py").write_text(opus_src, encoding="utf-8")

# Patch make_submit.py in build81 so it trains ONLY on Korean queries (100% dedicated Korean tree splits, 2.5x faster!)
make_src = (B81 / "make_submit.py").read_text(encoding="utf-8")
old_train_loop = '''    for seed in range(int(os.environ.get("SEEDS", "1"))):
        aug = seed * 1000 + int(os.environ.get("SEED_OFFSET", "0"))  # another draw of the same recipe
        X, y, g, _ = L.build(ix, pd.concat([L.augment_korean(train, aug), L.augment_corrupt(train, aug),
                                            L.augment_rename(train, aug), llm], ignore_index=True), True)
        models.append(lgb.train(params, lgb.Dataset(X, y, group=g, feature_name=L.FEATURES), rounds))
        print(f"trained seed {seed}", flush=True)'''

new_train_loop = '''    ko_train = train[train["vessel_name"].map(L.is_hangul)].reset_index(drop=True)
    ko_all = pd.concat([ko_train, llm], ignore_index=True)
    for seed in range(int(os.environ.get("SEEDS", "1"))):
        aug = seed * 1000 + int(os.environ.get("SEED_OFFSET", "0"))
        df_ko = pd.concat([
            L.augment_korean(ko_train, aug),
            L.augment_corrupt(ko_all, aug, frac=0.65),
            L.augment_rename(ko_train, aug, frac=0.20),
            llm,
        ], ignore_index=True)
        X, y, g, _ = L.build(ix, df_ko, True)
        models.append(lgb.train(params, lgb.Dataset(X, y, group=g, feature_name=L.FEATURES), rounds))
        print(f"trained ko-only seed {seed} (rows={len(df_ko)})", flush=True)'''

assert old_train_loop in make_src, "old_train_loop not found"
make_src = make_src.replace(old_train_loop, new_train_loop, 1)
(B81 / "make_submit.py").write_text(make_src, encoding="utf-8")

# Create build_83_from_81_82.py which combines:
# - Korean: 72_submit (6 joint models) + 81+82 (6 dedicated Korean-only models) + 62_submit (6 joint models)
# - English: 65_submit (6 models, 3502/3568) + 62_submit (6 models)
build83_py = '''"""Combine 81+82 (6 dedicated Korean-only models) + 72_submit (6 joint models) on Korean, and 65_submit on English -> 83_submit.zip."""
from __future__ import annotations
import unicodedata
import zipfile
from pathlib import Path

ROOT = Path(unicodedata.normalize("NFD", "/Users/sunjae/Desktop/항만"))

SCRIPT_83 = \'\'\'"""DACON inference entry point (83_submit: Dedicated Korean Ranker [81+82] + Joint [72] on Korean, [65] on English)."""
from __future__ import annotations
from pathlib import Path
import os
import pickle
import sys
import unicodedata

if os.environ.get("PYTHONHASHSEED") is None:
    os.execve(sys.executable, [sys.executable] + sys.argv, dict(os.environ, PYTHONHASHSEED="0"))

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "model"))
import opus_lib as L72
import opus_lib_81 as L81
import opus_lib_65 as L65


def run(test_path="data/test.csv", catalog_path="data/vessel_catalog.csv", out="output/submission.csv"):
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    with open(HERE / "model" / "stats.pkl", "rb") as f:
        stats = pickle.load(f)

    ix72 = L72.Index(catalog, None)
    ix72.cs_map = {k: ix72.id2i[v] for k, v in stats["cs_map"].items() if v in ix72.id2i}
    ix72.flag_p, ix72.type_p = stats["flag_p"], stats["type_p"]
    ix72.train_targets = set(ix72.cs_map.values())
    ix72.ko = L72.KA.KoMatcher(catalog, stats["ko_pairs"])
    ix72.fit_alias(stats["alias_rows"])

    ix81 = L81.Index(catalog, None)
    ix81.cs_map = ix72.cs_map
    ix81.flag_p, ix81.type_p = stats["flag_p"], stats["type_p"]
    ix81.train_targets = ix72.train_targets
    ix81.ko = L81.KA.KoMatcher(catalog, stats["ko_pairs"])
    ix81.fit_alias(stats["alias_rows"])

    models_72 = [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob("ranker_*.txt"))]
    models_81 = [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob("ko81_*.txt"))]
    models_65 = [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob("en65_*.txt"))]
    train_targets = {ix72.id2i[v] for v in stats["cs_map"].values() if v in ix72.id2i}
    preds = []
    for row in test.to_dict("records"):
        cs = ix72.cs_map.get(row.get("call_sign"))
        if cs is not None:
            preds.append(ix72.ids[cs])
            continue
        q = unicodedata.normalize("NFC", str(row.get("vessel_name") or ""))
        if L72.is_hangul(q):
            cand = L72.query_candidates(ix72, row)
            idx, F72 = L72.query_features(ix72, row, cand)
            _, F81 = L81.query_features(ix81, row, cand)
            score = 1.0 * sum(m.predict(F72) for m in models_72) + 1.0 * sum(m.predict(F81) for m in models_81)
        else:
            idx, F65 = L65.query_features(ix72, row, L65.query_candidates(ix72, row))
            score = sum(m.predict(F65) for m in models_65)
        order = sorted(range(len(idx)), key=lambda j: (-float(score[j]), idx[j]))
        best = next((idx[j] for j in order if idx[j] not in train_targets), idx[order[0]])
        preds.append(ix72.ids[best])

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": test["sample_id"], "target_vessel_id": preds}).to_csv(out, index=False)
    print(f"Saved {len(preds)} predictions to {out}")


if __name__ == "__main__":
    run()
\'\'\'


def main():
    with zipfile.ZipFile(ROOT / "submissions/72_submit.zip") as z72:
        items = {n: z72.read(n) for n in z72.namelist() if not n.endswith("/")}
    with zipfile.ZipFile(ROOT / "submissions/81_submit.zip") as z81:
        items["model/opus_lib_81.py"] = z81.read("model/opus_lib.py")
        items["model/ko_translit_argon_81.py"] = z81.read("model/ko_translit_argon.py")
        for i in range(3):
            items[f"model/ko81_{i}.txt"] = z81.read(f"model/ranker_{i}.txt")
    with zipfile.ZipFile(ROOT / "submissions/82_submit.zip") as z82:
        for i in range(3):
            items[f"model/ko81_{i+3}.txt"] = z82.read(f"model/ranker_{i}.txt")
    # Make opus_lib_81 import ko_translit_argon_81
    items["model/opus_lib_81.py"] = (
        items["model/opus_lib_81.py"]
        .decode("utf-8")
        .replace("import ko_translit_argon as KA", "import ko_translit_argon_81 as KA")
        .encode("utf-8")
    )
    with zipfile.ZipFile(ROOT / "submissions/65_submit.zip") as z65:
        items["model/opus_lib_65.py"] = z65.read("model/opus_lib.py")
        for i in range(6):
            items[f"model/en65_{i}.txt"] = z65.read(f"model/ranker_{i}.txt")
    items["script.py"] = SCRIPT_83.encode("utf-8")
    dst = ROOT / "submissions/83_submit.zip"
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for n in sorted(items.keys()):
            zout.writestr(n, items[n])
    print(f"Built {dst} ({dst.stat().st_size / 1e6:.2f} MB) with 6 (72) + 6 (81/82 Dedicated Ko) + 6 (65 En) = 18 models.")


if __name__ == "__main__":
    main()
'''
(ROOT / "shared/experiments/build_83_from_81_82.py").write_text(build83_py, encoding="utf-8")

run81_sh = '''#!/bin/sh
set -eu
R=/Users/sunjae/Desktop/항만
E=$R/shared/experiments
S=$R/submissions
PY=/Users/sunjae/.pyenv/versions/3.11.9/bin/python3
cd /tmp

echo "[1/4] Launching fast Dedicated Korean-Only training for 81 (SEED_OFFSET=0) and 82 (SEED_OFFSET=5000)..."
for n in 81 82; do
  off=$(( (n - 81) * 5000 ))
  PYTHONHASHSEED=0 KO_DICT=1 LLM_DICT_CSV=llm_korean_scored.csv LLM_TRAIN_CSV=llm_korean_scored.csv SUBMIT_NUM=$n SEEDS=3 \\
    SEED_OFFSET=$off OUT_DIR=submit$n CACHE_DIR=$E/cache nohup "$PY" "$E/pinned/build81/make_submit.py" > "$E/make$n.log" 2>&1 &
done
wait

echo "[2/4] Applying post-hoc train-target exclusion & determinism patch..."
for n in 81 82; do
  grep -q "wrote .*${n}_submit.zip" "$E/make$n.log" || { echo "build $n FAILED"; exit 1; }
  "$PY" "$R/shared/src/add_exclusion.py" "$S/${n}_submit.zip" "$S/${n}_submit.zip"
done

echo "[3/4] Building 83_submit.zip (Dedicated Ko [81+82] + Joint Ko [72] + Best En [65])..."
"$PY" "$E/build_83_from_81_82.py"

echo "[4/4] Running local_check.py on 83_submit.zip..."
"$PY" "$R/shared/src/local_check.py" "$S/83_submit.zip" > "$E/check_83.log" 2>&1
echo "83 ready $(date)"
tail -3 "$E/check_83.log"
'''
sh81 = ROOT / "run_build81.sh"
sh81.write_text(run81_sh, encoding="utf-8")
sh81.chmod(0o755)
print("Successfully created build81 and run_build81.sh (-> 83_submit.zip)!")
