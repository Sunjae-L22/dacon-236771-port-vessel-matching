# author: argon
"""Surgical candidate-retrieval and feature-alignment fixes for v18/v19 residual Korean errors.

Addresses the two largest solvable error categories identified on v18 fold 0 (vessel-split):
1. Category (a) — True vessel missing from candidates (151 misses across 29 variants):
   - `query_candidates_v18_fix`:
     (i) Inside the ±12% GT window (`pool`), ranks candidates not only by `opus_lib`'s crude
         consonant skeleton `ck`, but also by `KoMatcher`'s translated `cat_en_compact` and
         phonetic `cat_en2ko` (`fuzz.ratio` and `fuzz.partial_ratio`). This recovers ships whose
         first word was replaced, dropped, or corrupted (`디에스 오션 -> 우양 오션`,
         `돈카를로스 -> 카를로스`, `코랄젬 -> 코랄`) where the true ship is within ±12% GT
         but fell past rank 40 in global name-only search due to common tokens (`Ocean`, `Coral`).
     (ii) Widens `meta_candidates` from `(k=40, window=0.06)` to `(k=60, window=0.085)` so
          renamed ships with 6.0%~8.5% domestic/international GT discrepancy are retained.

2. Category (d) — True vessel has higher KoMatcher name score but ranker picks another ship (202 misses):
   - `query_features_v18_fix` / `align_ko_group_features_inplace`:
     (i) Sets `comb_eff = np.maximum(ko_combined_score, ko_comb_rev)` and computes
         `ko_comb_best` and `ko_comb_margin` from `comb_eff` (fixes word-order swap inversions
         where `ko_comb_rev` is 0.95+ but `ko_comb_best` was negative).
     (ii) For Korean queries (`is_ko == 1`), computes `name_best` and `name_margin` from
          `comb_eff * 100.0` instead of `opus_lib`'s crude `ck_acr_ratio`.
          Previously, `ck_acr_ratio` gave `펭슈하이` (`Feng Xiu Hai`, ko=0.896) `name_best = -14.3`
          vs `Feng Shou Hai` (ko=0.866, `name_best = 0.0`), and gave `엠에스 프리지아` (`MS Freesia`,
          ko=0.968) `name_best = -7.3` with a tiny `name_margin`, causing the ranker to prefer
          `Serene Monaco` (ko=0.389, `gt_exact = 1.0`).

Rules compliance:
- Zero meaning translations (`대양->ocean` etc. excluded), zero shipping-company alias tables,
  zero external data.
- All set iterations wrapped in `sorted(...)` for strict run-to-run determinism.
"""
from __future__ import annotations

import unicodedata
import numpy as np
from rapidfuzz import fuzz, process


def query_candidates_v18_fix(ix, row: dict, opus_lib_mod, k_name: int = 30, k_gt: int = 60, k_ko: int = 40) -> dict:
    """Drop-in replacement for `opus_lib.query_candidates` that closes Korean Category (a) recall gaps."""
    cand = opus_lib_mod.query_candidates(ix, row, k_name=k_name, k_gt=k_gt, k_ko=k_ko)
    q = unicodedata.normalize("NFC", str(row.get("vessel_name") or ""))
    if not opus_lib_mod.is_hangul(q):
        return cand

    gt = opus_lib_mod.positive(row.get("gross_tonnage"))
    igt = opus_lib_mod.positive(row.get("international_gross_tonnage"))
    own = ix.id2i.get(row.get("target_vessel_id"))

    def add(i, src_rank=None, gt_rank=None, ko_rank=None):
        i = int(i)
        if opus_lib_mod.EXCLUDE_TRAIN and i in ix.train_targets and i != own:
            return
        e = cand.setdefault(i, [999, 999, 999])
        if src_rank is not None:
            e[0] = min(e[0], src_rank)
        if gt_rank is not None:
            e[1] = min(e[1], gt_rank)
        if ko_rank is not None:
            e[2] = min(e[2], ko_rank)

    # 1. Rank the ±12% GT window using KoMatcher's translated compact & phonetic keys
    pool = set()
    for v in dict.fromkeys((gt, igt)):
        if np.isfinite(v):
            lo, hi = np.searchsorted(ix.gt_sorted, [v * 0.88, v * 1.12])
            pool.update(ix.gt_sorted_idx[lo:hi].tolist())
    if pool:
        pool_l = sorted(pool)
        qinfo = ix.ko.translate_ko(q)
        qt = qinfo["trans"]
        qr = qinfo["rom_no_eu"]
        if qt:
            c_comp = [ix.ko.cat_en_compact[i] for i in pool_l]
            for scorer in (fuzz.ratio, fuzz.partial_ratio):
                s = process.cdist([qt], c_comp, scorer=scorer, workers=1)[0]
                for r, j in enumerate(np.argsort(-s, kind="stable")[:k_ko]):
                    add(pool_l[j], r)
        if qr:
            c_e2k = [ix.ko.cat_en2ko[i] for i in pool_l]
            s = process.cdist([qr], c_e2k, scorer=fuzz.partial_ratio, workers=1)[0]
            for r, j in enumerate(np.argsort(-s, kind="stable")[: k_ko // 2]):
                add(pool_l[j], r)

    # 2. Widen meta_candidates window from 6.0% (k=40) to 8.5% (k=60)
    for r, i in enumerate(
        ix.meta_candidates(
            gt,
            igt,
            str(row.get("vessel_type")),
            str(row.get("flag_name")),
            k=60,
            window=0.085,
        )
    ):
        add(i, None, r)

    return cand


def align_ko_group_features_inplace(F: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Align Korean group-relative features (`ko_comb_best`, `ko_comb_margin`, `name_best`, `name_margin`)
    with `np.maximum(ko_combined_score, ko_comb_rev)` for a single query's feature matrix `F`."""
    n = len(F)
    if n == 0:
        return F
    col = {f: j for j, f in enumerate(feature_names)}
    if F[0, col["is_ko"]] < 0.5:
        return F

    comb = F[:, col["ko_combined_score"]]
    rev = F[:, col["ko_comb_rev"]]
    comb_eff = np.maximum(comb, rev)
    F[:, col["ko_combined_score"]] = comb_eff

    cmax = float(comb_eff.max())
    csec = float(np.partition(comb_eff, -2)[-2]) if n > 1 else 0.0
    F[:, col["ko_comb_best"]] = comb_eff - cmax
    F[:, col["ko_comb_margin"]] = cmax - csec

    best_ko = comb_eff * 100.0
    bmax = float(best_ko.max())
    bsec = float(np.partition(best_ko, -2)[-2]) if n > 1 else 0.0
    F[:, col["name_best"]] = best_ko - bmax
    F[:, col["name_margin"]] = bmax - bsec
    return F


def query_features_v18_fix(ix, row: dict, cand: dict, opus_lib_mod) -> tuple[list[int], np.ndarray]:
    """Drop-in replacement for `opus_lib.query_features` with Korean group-feature alignment."""
    idx, F = opus_lib_mod.query_features(ix, row, cand)
    align_ko_group_features_inplace(F, opus_lib_mod.FEATURES)
    return idx, F
