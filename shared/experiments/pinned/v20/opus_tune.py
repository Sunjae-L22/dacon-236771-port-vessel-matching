# author: opus
"""Cache fold-0 validation features under many corruptions once, then compare LightGBM
settings quickly. Score = mean accuracy over variants, weighted 0.7 ko / 0.3 en."""
import os
import pickle
import random
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cv import folds, load  # noqa: E402
import opus_lib as L  # noqa: E402
from opus_cv import CACHE, PARAMS, fold_data  # noqa: E402
import opus_stress2 as S2  # noqa: E402
import opus_stress3 as S3  # noqa: E402
import opus_stress4 as S4  # noqa: E402


def val_cache(tag, k=0, n_en=400):
    # VAL_TAG lets several training caches share one validation cache when their features match
    p = CACHE / f"{os.environ.get('VAL_TAG', tag)}_valvariants_fold{k}.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    train, catalog = load()
    f = folds(train)
    tr, va = train[f != k], train[f == k]
    ix = L.Index(catalog, tr)
    pool_en = [n.upper() for n in catalog["vessel_name"].sample(5000, random_state=1)]
    pool_ko = sorted(set(tr.loc[tr.vessel_name.map(L.is_hangul), "vessel_name"]))
    v3 = S3.variants(pool_en, pool_ko, seed=0)
    fns = {n: (lambda fn: lambda r, c: fn(r))(v3[n])
           for n in ("orig", "edit1", "edit2", "edit4", "rename", "rename_brand", "rename_new", "edit2+nogt")}
    fns.update({n: S4.V[n] for n in ("reorder", "initial", "trunc8", "jamo1")})
    fns.update({n: (lambda fn: lambda r, c: fn(r))(S2.V[n]) for n in ("drop_last_word", "trunc60%")})
    sets = {"ko": va[va.vessel_name.map(L.is_hangul)],
            "en": va[~va.vessel_name.map(L.is_hangul)].sample(n_en, random_state=0)}
    out = {}
    for name, fn in fns.items():
        for lang, sub in sets.items():
            groups = []
            for row in sub.to_dict("records"):
                t = ix.id2i[row["target_vessel_id"]]
                new = fn(row, ix.names_raw[t])
                idx, F = L.query_features(ix, new, L.query_candidates(ix, new))
                groups.append((F, np.array(idx) == t))
            out[(name, lang)] = groups
        print("cached", name, flush=True)
    pickle.dump(out, open(p, "wb"))
    return out


def evaluate(models, vc):
    res = {}
    for (name, lang), groups in vc.items():
        ok = 0
        for F, y in groups:
            s = sum(m.predict(F) for m in models)
            ok += bool(y[int(np.argmax(s))])
        res[(name, lang)] = ok / len(groups)
    ko = np.mean([v for (n, l), v in res.items() if l == "ko"])
    en = np.mean([v for (n, l), v in res.items() if l == "en"])
    return 0.7 * ko + 0.3 * en, ko, en, res


NAME_UP = ["exact", "ratio", "pratio", "tsort", "tset", "jw", "lev_norm", "ck_ratio", "ck_acr_ratio", "vk_ratio",
           "ck_jw", "rom_ratio", "digit_match", "name_best", "ko_trans_sim", "ko_trans_jw", "ko_en2ko_sim",
           "ko_cons_sim", "ko_cons_acr_sim", "ko_phon_sim", "ko_no_eu_sim", "ko_prefix_sim", "ko_digit_match",
           "ko_tail_sim", "ko_partial_sim", "ko_prefix_exact", "ko_token_cov", "ko_jamo_sim", "ko_combined_score",
           "ko_comb_best", "is_prefix", "ck_pratio", "ko_comb_rev", "initial_ok"]
META_UP = ["gt_exact", "flag_p", "type_p"]
DOWN = ["digit_conflict", "gt_gap", "igt_gap", "best_gap", "rank_name", "rank_gt"]


def mono(up, down):
    """A better name (or tonnage) match may never lower a candidate's score."""
    return [1 if f in up else -1 if f in down else 0 for f in L.FEATURES]


CONFIGS = {
    "mono_names": dict(PARAMS, monotone_constraints=mono(NAME_UP, []), monotone_constraints_method="advanced"),
    "mono_all": dict(PARAMS, monotone_constraints=mono(NAME_UP + META_UP, DOWN), monotone_constraints_method="advanced"),
    "base": dict(PARAMS),
    "leaves31": dict(PARAMS, num_leaves=31),
    "leaves127": dict(PARAMS, num_leaves=127),
    "lr03_r1200": dict(PARAMS, learning_rate=0.03, _rounds=1200),
    "mindata100": dict(PARAMS, min_data_in_leaf=100),
    "trunc_all": dict(PARAMS, lambdarank_truncation_level=40),
    "binary": dict(PARAMS, objective="binary", metric="binary_logloss"),
}

def ensemble_check(tags):
    """Average the scores of models trained on differently-seeded augmentation caches."""
    train, catalog = load()
    f = folds(train)
    vc = val_cache(tags[0])
    models = []
    for t in tags:
        Xtr, ytr, gtr, *_ = fold_data(train, catalog, f, 0, t)
        models.append(lgb.train(PARAMS, lgb.Dataset(Xtr, ytr, group=gtr, feature_name=L.FEATURES), 600))
        w, ko, en, _ = evaluate([models[-1]], vc)
        print(f"single {t}: weighted={w:.4f} ko={ko:.4f} en={en:.4f}", flush=True)
    w, ko, en, _ = evaluate(models, vc)
    print(f"ensemble {'+'.join(tags)}: weighted={w:.4f} ko={ko:.4f} en={en:.4f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "ensemble":
        ensemble_check(sys.argv[2:])
        sys.exit()
    tag = sys.argv[1]
    names = sys.argv[2:] or list(CONFIGS)
    train, catalog = load()
    Xtr, ytr, gtr, *_ = fold_data(train, catalog, folds(train), 0, tag)
    vc = val_cache(tag)
    for cname in names:
        p = dict(CONFIGS[cname])
        rounds = p.pop("_rounds", 600)
        m = lgb.train(p, lgb.Dataset(Xtr, ytr, group=gtr, feature_name=L.FEATURES), rounds)
        w, ko, en, res = evaluate([m], vc)
        print(f"{tag} {cname:12s} weighted={w:.4f} ko={ko:.4f} en={en:.4f} | "
              + " ".join(f"{n}:{res[(n, 'ko')]:.2f}/{res[(n, 'en')]:.2f}" for n in dict.fromkeys(n for n, _ in res)), flush=True)
