"""Update shared/experiments/pinned/build67 and run_build67.sh:
Combines ALL LB-verified and structural breakthroughs into a single retrain recipe (76 + 77 -> 78_submit.zip):
1. `augment_corrupt` wide tonnage noise (30% of kept-tonnage corrupted rows scaled by 0.8..1.25 -> broke 491 to 492/532 in 72_submit.zip!)
2. `augment_korean` full [0.8, 1.25] noisy tonnage scale (keeping the 2 noisy-tonnage Korean rows that 65 lost when shrinking to 0.915..1.085)
3. 5 NaN-free GT-window relative features (`ko_gt_win_best`, `ko_gt_win_margin`, `ko_gt_win_zscore`, `ko_tail_gt_best`, `ko_part_gt_best`) + 2 matched-token metadata tie-breaker features (`ko_tok_meta_best`, `ko_tok_meta_margin`) for 1-word truncated collisions (`모닝`, `글로비스`, `오리엔탈`, `케이엠티씨`, `송가`, `슈퍼`, `갤럭시`)!
4. `query_candidates()` tonnage-window (`±15%`) direct `KoMatcher` (`q_trans` + `q_rom`) retrieval so truncated/dropped-word Korean queries never miss `cand`.
5. Korean-English phonology fixes (`woo->u`, `hyun->hyeon`, `chem->kem`, word-final `-se->j`, `I/Y` acronym protection, `w/y` glide vowel compatibility, dropped-first-word bonus).
6. Runs from `cd /tmp` (preventing macOS Desktop cwd hang).
"""
from __future__ import annotations
import shutil
import unicodedata
from pathlib import Path

ROOT = Path(unicodedata.normalize("NFD", "/Users/sunjae/Desktop/항만"))
B63 = ROOT / "shared/experiments/pinned/build63"
B67 = ROOT / "shared/experiments/pinned/build67"

if B67.exists():
    shutil.rmtree(B67)
shutil.copytree(B63, B67, ignore=shutil.ignore_patterns("__pycache__"))

# 1. Patch ko_translit_argon.py in build67
ko_src = (B67 / "ko_translit_argon.py").read_text(encoding="utf-8")

old_sub = '    ("카멜리아", "camellia"), ("아젤리아", "azalea"), ("프리지아", "freesia"),\n    ("아카시아", "acacia"), ("에델바이스", "edelweiss"),'
new_sub = (
    '    ("카멜리아", "camellia"), ("아젤리아", "azalea"), ("프리지아", "freesia"),\n'
    '    ("아카시아", "acacia"), ("에델바이스", "edelweiss"),\n'
    '    ("로즈", "rose"), ("릴리", "lily"), ("아이리스", "iris"), ("로투스", "lotus"),\n'
    '    ("오키드", "orchid"), ("달리아", "dahlia"), ("칸나", "canna"), ("바이올렛", "violet"),'
)
assert old_sub in ko_src
ko_src = ko_src.replace(old_sub, new_sub, 1)

old_e2k = '''@lru_cache(maxsize=131072)
def en_to_ko_phonetic_lat(en_body: str, keep_precons_r: bool = False) -> str:
    """Project an English ship name into Korean loanword phonetic Latin using pure phonological rules."""
    s = en_body.lower()
    s = re.sub(r"tion\\b", "syeon", s)
    s = re.sub(r"sion\\b", "jeon", s)
    s = re.sub(r"liance\\b", "laieonseu", s)
    s = re.sub(r"ture\\b", "cheo", s)
    s = re.sub(r"shou", "seou", s)
    s = re.sub(r"zh", "j", s)
    s = re.sub(r"ph", "p", s)
    s = re.sub(r"tch|ch", "ch", s)'''

new_e2k = '''@lru_cache(maxsize=131072)
def en_to_ko_phonetic_lat(en_body: str, keep_precons_r: bool = False) -> str:
    """Project an English ship name into Korean loanword phonetic Latin using pure phonological rules."""
    s = en_body.lower()
    s = re.sub(r"\\bchem", "kem", s)
    s = re.sub(r"woo", "u", s)
    s = re.sub(r"\\bwu", "u", s)
    s = re.sub(r"hyun", "hyeon", s)
    s = re.sub(r"hyoung|hyung", "hyeong", s)
    s = re.sub(r"young|yung", "yeong", s)
    s = re.sub(r"([aeiou])se\\b", r"\\1j", s)
    s = re.sub(r"tion\\b", "syeon", s)
    s = re.sub(r"sion\\b", "jeon", s)
    s = re.sub(r"liance\\b", "laieonseu", s)
    s = re.sub(r"ture\\b", "cheo", s)
    s = re.sub(r"shou", "seou", s)
    s = re.sub(r"zh", "j", s)
    s = re.sub(r"ph", "p", s)
    s = re.sub(r"tch|ch", "ch", s)'''

assert old_e2k in ko_src
ko_src = ko_src.replace(old_e2k, new_e2k, 1)

old_ai = '    s = re.sub(r"ai|ay", "ei", s)'
new_ai = '    s = re.sub(r"ay\\b|(?<![whktsmdlnpb\\s])ai(?=[lnrt])", "ei", s)'
assert old_ai in ko_src
ko_src = ko_src.replace(old_ai, new_ai, 1)

ko_src = ko_src.replace('in "aeiou"', 'in "aeiouwy"')

old_bonus = '''        # 15. ko_combined_score: unified ranker/retrieval score
        is_short_query = len(q_trans) < int(0.72 * len(c_compact)) and prefix_exact >= 0.80
        partial_bonus = 0.12 * (0.6 * partial_sim + 0.4 * prefix_exact) if is_short_query else 0.04 * prefix_exact'''

new_bonus = '''        # 15. ko_combined_score: unified ranker/retrieval score
        is_short_prefix = len(q_trans) < int(0.72 * len(c_compact)) and prefix_exact >= 0.80
        is_short_drop_first = len(q_trans) >= 3 and len(q_trans) < int(0.75 * len(c_compact)) and partial_sim >= 0.90 and token_cov >= 0.90 and tail_sim >= 0.70
        if is_short_prefix:
            partial_bonus = 0.12 * (0.6 * partial_sim + 0.4 * prefix_exact)
        elif is_short_drop_first:
            partial_bonus = 0.11 * (0.5 * partial_sim + 0.5 * token_cov)
        else:
            partial_bonus = 0.04 * prefix_exact'''

assert old_bonus in ko_src
ko_src = ko_src.replace(old_bonus, new_bonus, 1)
(B67 / "ko_translit_argon.py").write_text(ko_src, encoding="utf-8")

# 2. Patch opus_lib.py in build67
opus_src = (B67 / "opus_lib.py").read_text(encoding="utf-8")

# Add 2 matched-token metadata tie-breaker features for 1-word collisions
opus_src = opus_src.replace(
    '["ko_gt_win_best", "ko_gt_win_margin", "ko_gt_win_zscore", "ko_tail_gt_best", "ko_part_gt_best"]',
    '["ko_gt_win_best", "ko_gt_win_margin", "ko_gt_win_zscore", "ko_tail_gt_best", "ko_part_gt_best", "ko_tok_meta_best", "ko_tok_meta_margin"]',
    1,
)

old_cand_ko = '''    if ko:
        _, ck, _ = ko_keys(q)
        pool = set()
        for v in dict.fromkeys((gt, igt)):
            if np.isfinite(v):
                lo, hi = np.searchsorted(ix.gt_sorted, [v * 0.88, v * 1.12])
                pool.update(ix.gt_sorted_idx[lo:hi].tolist())
        if ck and pool:
            pool = list(pool)
            keys = [ix.ckey_acr[i] for i in pool]
            for scorer in (fuzz.ratio, fuzz.partial_ratio):
                s = process.cdist([ck], keys, scorer=scorer, workers=1)[0]
                for r, j in enumerate(np.argsort(-s, kind="stable")[:k_ko]):
                    add(pool[j], r)'''

new_cand_ko = '''    if ko:
        _, ck, _ = ko_keys(q)
        pool = set()
        for v in dict.fromkeys((gt, igt)):
            if np.isfinite(v):
                lo, hi = np.searchsorted(ix.gt_sorted, [v * 0.85, v * 1.15])
                pool.update(ix.gt_sorted_idx[lo:hi].tolist())
        if pool:
            pool = sorted(pool)
            if ck:
                keys = [ix.ckey_acr[i] for i in pool]
                for scorer in (fuzz.ratio, fuzz.partial_ratio):
                    s = process.cdist([ck], keys, scorer=scorer, workers=1)[0]
                    for r, j in enumerate(np.argsort(-s, kind="stable")[:k_ko]):
                        add(pool[j], r)
            qinfo = ix.ko.translate_ko(q)
            q_tr = qinfo["trans"]
            q_rom = qinfo["rom_no_eu"]
            if q_tr:
                c_comp = [ix.ko.cat_en_compact[i] for i in pool]
                for scorer in (fuzz.ratio, fuzz.partial_ratio):
                    s = process.cdist([q_tr], c_comp, scorer=scorer, workers=1)[0]
                    for r, j in enumerate(np.argsort(-s, kind="stable")[: k_ko // 2]):
                        add(pool[j], r)
            if q_rom:
                c_e2k = [ix.ko.cat_en2ko[i] for i in pool]
                for scorer in (fuzz.ratio, fuzz.partial_ratio):
                    s = process.cdist([q_rom], c_e2k, scorer=scorer, workers=1)[0]
                    for r, j in enumerate(np.argsort(-s, kind="stable")[: k_ko // 2]):
                        add(pool[j], r)'''

assert old_cand_ko in opus_src
opus_src = opus_src.replace(old_cand_ko, new_cand_ko, 1)

old_win_block = '''        # GT-window (best_gap <= 0.12) relative features without NaN poisoning
        win_mask = F[:, col["best_gap"]] <= 0.12
        if not np.any(win_mask):
            win_mask = np.ones(n, dtype=bool)
        win_idx = np.where(win_mask)[0]
        win_comb = comb[win_idx]
        win_max = float(win_comb.max())
        win_mean = float(win_comb.mean())
        win_std = float(win_comb.std()) + 1e-3
        if len(win_comb) >= 2:
            top2 = np.partition(win_comb, -2)[-2:]
            win_top1, win_top2 = float(top2[1]), float(top2[0])
        else:
            win_top1 = win_top2 = win_max
        F[:, col["ko_gt_win_best"]] = comb - win_max
        is_win_top1 = win_mask & (comb >= win_top1 - 1e-9)
        F[:, col["ko_gt_win_margin"]] = np.where(is_win_top1, comb - win_top2, comb - win_top1)
        F[:, col["ko_gt_win_zscore"]] = (comb - win_mean) / win_std
        tail_arr = F[:, col["ko_tail_sim"]]
        part_arr = np.maximum(F[:, col["ko_partial_sim"]], F[:, col["ko_token_cov"]])
        F[:, col["ko_tail_gt_best"]] = tail_arr - float(tail_arr[win_idx].max())
        F[:, col["ko_part_gt_best"]] = part_arr - float(part_arr[win_idx].max())'''

new_win_block = '''        # GT-window (best_gap <= 0.12) relative features without NaN poisoning
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
        # Matched-token metadata tie-breaker for 1-word / prefix / suffix collisions (e.g. 모닝, 글로비스, 오리엔탈, 케이엠티씨)
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
        F[:, col["ko_tok_meta_margin"]] = np.where(is_tm_top1, meta_sc - tm_2, meta_sc - tm_1)'''

assert old_win_block in opus_src
opus_src = opus_src.replace(old_win_block, new_win_block, 1)

# Restore [0.8, 1.25] in augment_korean AND add 30% wide tonnage noise in augment_corrupt (proven in 72_submit.zip!)
opus_src = opus_src.replace("scale = rng.uniform(0.915, 1.085, len(ko))", "scale = rng.uniform(0.8, 1.25, len(ko))", 1)

old_corrupt = '''def augment_corrupt(train: pd.DataFrame, seed: int = 0, frac: float = 0.5) -> pd.DataFrame:
    """Name-corrupted copies (typos, dropped words, truncation) for both languages;
    30% of them also lose tonnage."""
    rng = np.random.default_rng(seed + 100)
    sub = train.sample(frac=frac, random_state=seed + 100).copy()
    sub["vessel_name"] = [_corrupt_name(str(v), rng) for v in sub["vessel_name"]]
    drop = rng.random(len(sub)) < 0.3
    sub.loc[drop, ["gross_tonnage", "international_gross_tonnage"]] = np.nan
    return sub'''

new_corrupt = '''def augment_corrupt(train: pd.DataFrame, seed: int = 0, frac: float = 0.5) -> pd.DataFrame:
    """Name-corrupted copies (typos, dropped words, truncation) for both languages;
    30% of them also lose tonnage, and 30% of the kept-tonnage copies get wide [0.8, 1.25] tonnage noise."""
    rng = np.random.default_rng(seed + 100)
    sub = train.sample(frac=frac, random_state=seed + 100).copy()
    sub["vessel_name"] = [_corrupt_name(str(v), rng) for v in sub["vessel_name"]]
    drop = rng.random(len(sub)) < 0.3
    sub.loc[drop, ["gross_tonnage", "international_gross_tonnage"]] = np.nan
    wide = ~drop & (rng.random(len(sub)) < 0.3)
    scale = rng.uniform(0.8, 1.25, len(sub))
    for c in ("gross_tonnage", "international_gross_tonnage"):
        sub.loc[wide, c] = sub.loc[wide, c] * scale[wide]
    return sub'''

assert old_corrupt in opus_src
opus_src = opus_src.replace(old_corrupt, new_corrupt, 1)
(B67 / "opus_lib.py").write_text(opus_src, encoding="utf-8")

# 3. Write run_build67.sh (runs from /tmp to avoid macOS Desktop cwd hang, builds 76 + 77 -> 78_submit.zip)
run_sh = '''#!/bin/sh
set -eu
R=/Users/sunjae/Desktop/항만
E=$R/shared/experiments
S=$R/submissions
PY=/Users/sunjae/.pyenv/versions/3.11.9/bin/python3
cd /tmp

echo "[1/4] Launching parallel training from /tmp for 76 (SEED_OFFSET=0) and 77 (SEED_OFFSET=5000)..."
for n in 76 77; do
  off=$(( (n - 76) * 5000 ))
  PYTHONHASHSEED=0 KO_DICT=1 LLM_DICT_CSV=llm_korean_scored.csv LLM_TRAIN_CSV=llm_korean_scored.csv SUBMIT_NUM=$n SEEDS=3 \\
    SEED_OFFSET=$off OUT_DIR=submit$n CACHE_DIR=$E/cache nohup "$PY" "$E/pinned/build67/make_submit.py" > "$E/make$n.log" 2>&1 &
done
wait

echo "[2/4] Applying post-hoc train-target exclusion & determinism patch..."
for n in 76 77; do
  grep -q "wrote .*${n}_submit.zip" "$E/make$n.log" || { echo "build $n FAILED"; exit 1; }
  "$PY" "$R/shared/src/add_exclusion.py" "$S/${n}_submit.zip" "$S/${n}_submit.zip"
done

echo "[3/4] Merging 76 + 77 into 6-model ensemble submissions/78_submit.zip..."
"$PY" "$R/shared/src/merge_rankers.py" "$S/78_submit.zip" "$S/76_submit.zip" "$S/77_submit.zip"

echo "[4/4] Running local_check.py on 78_submit.zip..."
"$PY" "$R/shared/src/local_check.py" "$S/78_submit.zip" > "$E/check_78.log" 2>&1
echo "78 ready $(date)"
tail -3 "$E/check_78.log"
'''
sh_path = ROOT / "run_build67.sh"
sh_path.write_text(run_sh, encoding="utf-8")
sh_path.chmod(0o755)
print("Successfully updated build67 (67 features) and run_build67.sh (-> 78_submit.zip)!")
