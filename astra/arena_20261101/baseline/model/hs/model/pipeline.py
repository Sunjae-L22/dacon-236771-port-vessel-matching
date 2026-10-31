"""Train / predict helpers shared by train.py and script.py."""
import copy
import unicodedata
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import matcher
from matcher import Catalog, Priors, QueryBatch, retrieve, features, FEATURES

HANGUL_WEIGHT = 3.0
# 'auto' (as for the 10/23 model.pkl, 397 trees): early stopping switches itself on above 10,000 pairs and
# holds out 10% of the training pairs at random.  False = all pairs, 400 trees (tried in the 10/24 retrain).
EARLY_STOPPING = "auto"
# The 10/24 retrieval changes (matcher.V2_CHANNEL, matcher.NOTON_TYPE_BAND) are used at prediction.
# False (default): the training pairs are built without them, i.e. exactly as for the 10/23 model.pkl.
# fold 0, shortened Hangul names: V2 at prediction only +16 rows; V2 also in training (retrained) +4 rows.
RETRIEVAL_1024_IN_TRAIN = False
# 10/24: store the trained trees as plain numpy arrays (TreeEnsemble) instead of the scikit-learn object.
# model.pkl then loads with any scikit-learn version and with numpy 1 or 2, and gives the same scores.
EXPORT_TREES = True
AUGMENT = True     # train the ranker on augmented copies too (augment.py)
# Monotonic constraints (Potharst & Feelders 2002): the match probability may only rise with a
# name-similarity feature and only fall with a tonnage gap or a name-rank/distance feature.
MONOTONIC = True
MONO_UP = ("exact_gt", "exact_igt", "key_exact", "key_lev", "key_cos", "roman_lev", "roman_cos", "skel_exact",
           "skel_lev", "skel_cos", "skelv_lev", "skelv_cos", "tok_jacc", "bigram_jacc", "prefix", "xl_pl", "xl_lx",
           "ng2_pl", "ng2_lo", "w_me_q", "w_hit_q", "w_cov_c", "w_me_c", "w_hit_c", "w_cov_q", "w_comb", "ngt_lo")
MONO_DOWN = ("gap_gt", "gap_igt", "gap_min", "r_gap", "d_gap", "gap_dwt", "r_name", "d_name", "r_lev", "d_lev",
             "xl_d", "xl_dpl", "ng2_dlo", "ng2_dpl", "w_dcomb", "w_miss_q", "ngt_dlo")
# 10/26: Latin records are ranked by a listwise model, Hangul records keep the pointwise classifier.
# The pointwise classifier fits every (record, candidate) pair on its own, so most of its loss comes from the ~100
# easy candidates of a record; the answer is the best of a record's candidates, so the listwise model fits the
# softmax cross-entropy of the true candidate within its record's list (ListNet top-1 loss, Cao et al. 2007; it
# bounds MRR and NDCG when one candidate is right, Bruch et al. 2019; for entity linking Zheng et al. 2010 report
# 0.905 listwise vs 0.764 classification).  fold 0, same cached features, only the loss differs: Latin sets
# 12,373 -> 12,624 of 14,649 rows right (word dropped +64, first word dropped +84, cut +40, initial +18,
# abbreviation +7, typos +2, unchanged records +1); Hangul sets: no gain outside sister-ship ties (same name part,
# same tonnage), which the pointwise model breaks better, so Hangul stays pointwise.
# "pointwise" = one classifier for every record (submit_8).
LATIN_RANKER = "listwise"
LIST_PARAMS = dict(max_iter=900, learning_rate=0.06, max_leaf_nodes=48, min_samples_leaf=40, l2_regularization=1.0,
                   n_iter_no_change=40, validation_fraction=0.1, random_state=0)
# 10/26 2nd: the evaluation vessels are not train vessels (vessel-disjoint split), so a record whose best candidate is
# a train answer vessel takes its best candidate that is not one.  Only train.csv's answer list is used (stored in
# model.pkl by fit()), never other test rows.  Public: +3 Hangul / +5 Latin rows on submit_4 (10/24), +0.0056 for a
# teammate's model; the teammate's call_sign check found no train vessel in the test.  Skipping train vessels before
# training instead lost 0.008 (teammate), so it is applied after scoring only.  False = submit_10.
EXCLUDE_TRAIN_VESSELS = True
# 10/27: Hangul spellings of the train vessels' catalog names written by a public pretrained LLM (Qwen2.5-7B-Instruct,
# llm_hangul_colab notebook: inputs are vessel_catalog.csv names and train.csv examples only, greedy decoding).
# fit() adds them as Hangul training records of their vessels (other fields from that vessel's first train record with
# a tonnage), so they also enter the dictionary / transliteration tables (Priors) like real Hangul records.  Their
# training features use the out-of-fold tables of their vessel group, as every train row's do.  Only real train rows
# get augmented copies.  Prediction never runs the LLM.  None = off (submit_11).
LLM_HANGUL = None        # file in the model folder, e.g. "llm_hangul_train.csv" (vessel_id, name_en, variant, name_ko)
LLM_WEIGHT = 0.5         # pair weight of an LLM record relative to a real record
LLM_TAG = "#llm"
# 10/27: fit() trains one model per seed and averages their raw scores (RawAverage) for each script.  Seed s uses
# augmentation seed augment.AUG_SEED + 1000 s and tree random_state s (the early-stopping hold-out); the features of
# the train and LLM records are computed once and shared.  (0,) = one model, exactly as before (submit_11).
# Public: a teammate's average of 6 same-recipe models was +0.001 (29) and, with LLM data, +0.0085 (33) over one
# model; retraining one model with another seed moves Public by about +-0.004.
SEEDS = (0, 1, 2)
# fold 0 (cached pairs, random_state 0/1/2): averaging the three pointwise models gave 2,501 of 3,637 Hangul set rows
# vs 2,464-2,488 for single seeds; averaging the three Latin listwise models gave 12,601 of 14,649 vs 12,603-12,624.
# So the Hangul ranker is the seed average and the Latin ranker is the first seed's model (LATIN_AVERAGE = False).
LATIN_AVERAGE = False
# 10/28: tonnage gate at pick time.  No train record's tonnage is more than 8.75% off its vessel's catalog tonnage
# (rel_gap of the features, best of gross / international, 11,429 records with a tonnage; none above 9%), so a
# candidate further off than TON_GATE is a non-match (a comparison never seen for true matches, Fellegi & Sunter
# 1969).  When a record has a candidate within TON_GATE, pick() skips the candidates further off; candidates without a
# catalog tonnage stay.  Uses the row's own tonnage and the catalog only.  CV folds 0-2 (vessel-disjoint, submit_13's
# rankers single-seed): Hangul sets 4,252 -> 4,259 of 4,591 rows (5 vessels better, 0 worse), renamed Hangul records
# 84 -> 99 of 704 (renamed to a held-out vessel's name 44 -> 105), Latin sets 37,264 -> 37,350 of 40,127 (45 / 0).
# None = off (submit_13).
# 10/29: off.  With the gate on, Public fell to 0.91970 (Hangul 478 / 532, Latin 3,458 / 3,568; submit_12 486 / 3,481)
# where CV expected a gain.  A record whose tonnage is more than TON_GATE off loses its answer to any candidate within
# TON_GATE, and the nearest-tonnage channel always supplies one: fold 0 held-out records with the real name and the
# tonnage scaled by 1.3 or 1 / 1.3 (no gate -> gate): Hangul 255 -> 10 of 257, Latin 2,080 -> 5 of 2,111; tonnage off
# by 10-25%: Latin 2,104 -> 186; a word dropped from the Hangul name as well: 57 -> 4 of 184.  Train never has such a
# record (54% exact, 36% off by 3-5%, none above 8.75%), so CV could not see the risk; test names are much noisier than
# train's, and a softer rule of the same kind also lost Public Hangul rows (10/24, tonnage-matching candidate first:
# -3).  Without the gate this file's rankers keep the tonnage as one piece of evidence: same stress 255 / 2,086 /
# 2,104 / 57.
TON_GATE = None
GAP_MIN, CGT_MISSING = FEATURES.index("gap_min"), FEATURES.index("c_gt_missing")
# 10/28: a LightGBM lambdarank model (LambdaMART, Burges 2010; LightGBM, Ke et al. 2017) on the first seed's pairs, used
# per record (ListRouter).  Hangul: the pointwise models pick unless none of the record's allowed candidates (allowed())
# holds every word of the record's name (w_hit_q < 1, the name was changed or rewritten), then lambdarank alone; it
# ranks the candidates with the record's tonnage, type and flag better when the name says little.  Latin: the mean of
# the listwise model's and lambdarank's scores, each standardised over the record's list (Yahoo LTR 2010 winners
# combined LambdaMART with other rankers, Burges et al. 2011).  CV folds 0-2 (single-seed models, TON_GATE on in both):
# Hangul sets 4,259 -> 4,276 rows (95% vessel bootstrap +1..+34), renamed Hangul records 99 -> 146 (held-out vessel's
# name 105 -> 141); Latin sets 37,350 -> 37,401 (130 vessels better, 82 worse).  Lambdarank for every Hangul record:
# +27 rows but -40..+96 (27 / 21 vessels); for Latin alone -53.  False = submit_13's rankers (ScriptRouter).
# 10/29, TON_GATE None (vs submit_13's rankers, no gate either): Hangul sets 4,252 -> 4,260 (-2..+19), renamed Hangul
# records 84 -> 122, Latin sets 37,264 -> 37,333 (+6..+132; 141 vessels better, 83 worse).
RANK_LGB = True
RANK_PARAMS = dict(objective="lambdarank", learning_rate=0.05, num_leaves=48, min_data_in_leaf=40, lambda_l2=1.0,
                   feature_fraction=1.0, bagging_fraction=1.0, seed=0, verbose=-1, deterministic=True,
                   force_row_wise=True, eval_at=[1], lambdarank_truncation_level=30,
                   monotone_constraints_method="intermediate")
RANK_TREES = 600
HANGUL_SWITCH = ("w_hit_q", 1.0)
# 10/28 2nd: a second pass for records whose name matches none of their candidates (a renamed vessel: the record carries
# an old name or another vessel's).  Every candidate channel starts from the name, so such a vessel is rarely
# retrieved (CV renamed Hangul records: the answer is among the candidates for 32%).  A record none of whose allowed
# candidates (allowed()) reaches RENAME_PASS on that feature (w_me_q: the mean best word similarity of the record's
# words; 0.8) is retrieved again with matcher.EXACT_CHANNEL (also the catalog vessels of exactly the record's tonnage,
# a second blocking key, Hernandez & Stolfo 1995, Christen 2012; answer among the candidates 73%) and ranked by
# "rename_model", a ListRouter trained the same way on pairs retrieved with that channel plus renamed training copies
# (augment.AUG_RENAME = RENAME_AUG: 15% of the train rows also get another train vessel's name).  The record takes the
# second pass only when its best allowed candidate there is a vessel the first list did not have; every other record
# keeps submit_14's candidates and scores.  CV folds 0-2 (single-seed models, vs submit_14's picks): Hangul sets
# 4,276 -> 4,276 of 4,591 rows (no pick changed), renamed Hangul records 146 -> 290 of 704 (95% vessel bootstrap
# +116..+174; renamed to a held-out vessel's name 141 -> 217), Latin sets 37,401 -> 37,393 of 40,127 (-27..+9),
# renamed Latin records 904 -> 1,597 of 6,639.  0.7 / 0.9: renamed Hangul +112 / +153, Latin sets -8 / -21; the
# second pass for every weak record (without the new-vessel rule): Latin sets -61 (-105..-21).  None = off (submit_14).
# 10/29, TON_GATE None in both passes (vs the first pass alone, no gate): Hangul sets 4,260 -> 4,260 (no pick changed),
# renamed Hangul records 122 -> 208 (+66..+107; renamed to a held-out vessel's name 43 -> 43: without the gate that
# vessel's name keeps the record strong), Latin sets 37,333 -> 37,323 (-26..+2), renamed Latin records 860 -> 1,365.
# 10/29 2nd: 0.9 (vs 0.8, no gate either; the models are the same): Hangul sets 4,260 -> 4,260 (no pick changed),
# renamed Hangul records 208 -> 240 (+21..+44; renamed to a held-out vessel's name 43 -> 51), Latin sets 37,323 ->
# 37,313 (-20..-3; 0 vessels better, 7 worse), renamed Latin records 1,365 -> 1,474 (+87..+132).  Public with the gate
# on: the second pass at 0.8 changed Hangul 478 -> 478 and Latin 3,459 -> 3,458 (submit_14 -> submit_15).
RENAME_PASS = ("w_me_q", 0.9)
RENAME_AUG = 0.15
# Out-of-fold priors for the training features.  The tables in Priors are learned from
# train labels; a training row translated / scored with tables that contain its own
# vessel looks much easier than an unseen test vessel (e.g. ~81% of training Hangul names
# translate to exactly the catalog name in-sample vs ~16% for unseen vessels).  So the
# training features of each vessel group are computed with Priors built from the other
# groups; the bundle keeps the full-train Priors for prediction.
OOF_K = 20                              # number of vessel groups (<= 1 disables OOF)
OOF_TABLES = ("tok", "xlit", "tng")  # which Priors tables are taken out-of-fold (type/flag OOF hurt in CV);
                                     # tng is built on the xlit units, so the two stay out-of-fold together

# Text columns are always read as strings.  Otherwise pandas infers a column's type from the
# rows of the file, so a digit-only name such as "801" became int 801 (and lost its name
# features) when it was alone in a file but stayed "801" next to other names.
TEXT_COLUMNS = ("sample_id", "vessel_name", "call_sign", "flag_name", "vessel_type", "target_vessel_id")


def read_records(path) -> pd.DataFrame:
    """Read train.csv / test.csv with the text columns as str, so a row's text values parse the
    same way whatever file it is in (tonnage columns stay numeric; positive() accepts either)."""
    return pd.read_csv(path, encoding="utf-8-sig", dtype={c: str for c in TEXT_COLUMNS})


def _oof_groups(train: pd.DataFrame, k: int) -> np.ndarray:
    """Deterministic vessel-level group (all rows of a vessel share a group)."""
    vid = train["target_vessel_id"].astype(str)
    return np.array([zlib.crc32(v.encode("utf-8")) % k for v in vid])


def _train_features(train: pd.DataFrame, cat: Catalog, pri: Priors, extra: pd.DataFrame | None = None,
                    base_rows: bool = True):
    """Pair features + candidates for every training row, in train order, then for every row of
    extra (augmented copies), each scored with the out-of-fold Priors of its vessel group.
    The Priors are always learned from the original train rows only.  base_rows=False: extra rows only
    (every row's features depend on that row alone, so the two parts can be computed apart)."""
    extra = train.iloc[:0] if extra is None else extra.reset_index(drop=True)
    if OOF_K <= 1 or not OOF_TABLES:
        out = []
        for df in ((train, extra) if base_rows else (extra,)):
            if len(df):
                q = QueryBatch(df, cat, pri)
                out.append(features(q, cat, pri, retrieve(q, cat)))
        return [f for o in out for f in o[0]], [c for o in out for c in o[1]]
    groups, xgroups = _oof_groups(train, OOF_K), _oof_groups(extra, OOF_K)
    n = len(train)
    feats, cands = [None] * (n + len(extra)), [None] * (n + len(extra))
    for k in range(OOF_K):
        rows = np.flatnonzero(groups == k) if base_rows else np.empty(0, dtype=np.int64)
        xrows = np.flatnonzero(xgroups == k)
        if not len(rows) and not len(xrows):
            continue
        oof = Priors(train.iloc[np.flatnonzero(groups != k)], cat)
        p = copy.copy(pri)
        for name in OOF_TABLES:
            setattr(p, name, getattr(oof, name))
        for df, pos in ((train.iloc[rows], rows), (extra.iloc[xrows], n + xrows)):
            if not len(pos):
                continue
            q = QueryBatch(df, cat, p)
            f, c = features(q, cat, p, retrieve(q, cat))
            for r, fr, cr in zip(pos, f, c):
                feats[r], cands[r] = fr, cr
    return (feats, cands) if base_rows else (feats[n:], cands[n:])


def train_pairs(train: pd.DataFrame, cat: Catalog, pri: Priors, aug_source: pd.DataFrame | None = None):
    """Features of the training rows plus the augmented copies (AUGMENT) of aug_source (default: train), and those
    rows."""
    train = train.reset_index(drop=True)
    if AUGMENT:
        import augment
        extra = augment.augment(train if aug_source is None else aug_source.reset_index(drop=True), cat)
    else:
        extra = train.iloc[:0]
    feats, cands = _train_features(train, cat, pri, extra)
    return feats, cands, pd.concat([train, extra], ignore_index=True)


def pair_weights(rows: pd.DataFrame, lens, X) -> np.ndarray:
    """Hangul pairs HANGUL_WEIGHT, pairs of augmented rows x AUG_WEIGHT."""
    w = np.where(X[:, 0] > 0, HANGUL_WEIGHT, 1.0)
    if LLM_HANGUL:
        w = w * np.repeat(np.where(is_llm(rows["sample_id"]), LLM_WEIGHT, 1.0), lens)
    if AUGMENT:
        import augment
        w = w * np.repeat(np.where(augment.is_aug(rows["sample_id"]), augment.AUG_WEIGHT, 1.0), lens)
    return w


def is_llm(sample_ids) -> np.ndarray:
    return np.array([LLM_TAG in str(s) for s in sample_ids], bool)


def llm_records(train: pd.DataFrame, path) -> pd.DataFrame:
    """Hangul training records from the LLM spellings file: one per (train vessel, spelling).  The other fields come
    from the vessel's first train record that has a tonnage (vessels without one are skipped); spellings equal to a
    real Hangul record of the vessel (ignoring spaces) are skipped."""
    llm = pd.read_csv(path, dtype=str, encoding="utf-8").dropna(subset=["vessel_id", "name_ko"])
    train = train.reset_index(drop=True)
    vid = train["target_vessel_id"].astype(str)
    has_ton = np.array([np.isfinite(matcher.positive(a)) or np.isfinite(matcher.positive(b))
                        for a, b in zip(train["gross_tonnage"], train["international_gross_tonnage"])])
    first = {}
    for i in np.flatnonzero(has_ton):
        first.setdefault(vid[i], i)
    real = {}
    for v, q in zip(vid, train["vessel_name"]):
        if isinstance(q, str) and matcher.is_hangul(matcher.clean(q)):
            real.setdefault(v, set()).add(unicodedata.normalize("NFC", q).replace(" ", ""))
    rows, seen = [], set()
    for v, var, ko in zip(llm["vessel_id"].astype(str), llm.get("variant", pd.Series(["1"] * len(llm))), llm["name_ko"]):
        ko = " ".join(unicodedata.normalize("NFC", str(ko)).split())
        key = (v, ko.replace(" ", ""))
        if v not in first or not matcher.is_hangul(matcher.clean(ko)) or key in seen or key[1] in real.get(v, ()):
            continue
        seen.add(key)
        r = train.iloc[first[v]].to_dict()
        r["sample_id"] = f"{r['sample_id']}{LLM_TAG}{var}"
        r["vessel_name"] = ko
        rows.append(r)
    return pd.DataFrame(rows, columns=train.columns)


class TreeEnsemble:
    """A fitted binary HistGradientBoostingClassifier as numpy arrays.

    predict_proba walks every tree as scikit-learn's predictor does (a NaN goes to the side stored in
    missing_go_to_left, otherwise x <= threshold goes left), adds the leaf values to the baseline in
    tree order and applies the logistic function, so the scores match model.predict_proba."""

    def __init__(self, model):
        nodes = [pp[0].nodes for pp in model._predictors]
        if model.n_trees_per_iteration_ != 1 or any(n["is_categorical"].any() for n in nodes):
            raise ValueError("TreeEnsemble supports binary models with numeric features only")
        sizes = np.array([len(n) for n in nodes], dtype=np.int64)
        self.offset = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        allnodes = np.concatenate(nodes)
        self.feature = allnodes["feature_idx"].astype(np.int64)
        self.threshold = allnodes["num_threshold"].astype(np.float64)
        self.missing_left = allnodes["missing_go_to_left"].astype(bool)
        self.left = allnodes["left"].astype(np.int64)
        self.right = allnodes["right"].astype(np.int64)
        self.is_leaf = allnodes["is_leaf"].astype(bool)
        self.value = allnodes["value"].astype(np.float64)
        self.baseline = float(np.asarray(model._baseline_prediction).ravel()[0])
        self.n_features_in_ = int(model.n_features_in_)
        self.n_iter_ = len(nodes)

    def raw(self, X) -> np.ndarray:
        """Sum of the leaf values (+ baseline) of every tree, added in tree order.  Each tree follows only the rows
        that have not reached a leaf yet (10/27: about 2x faster than walking every row to the deepest leaf, same
        values bit for bit)."""
        X = np.asarray(X, dtype=np.float64)
        n, nf = X.shape
        flat = X.ravel()
        base = np.arange(n, dtype=np.int64) * nf
        out = np.full(n, self.baseline)
        if getattr(self, "_gleft", None) is None:   # child ids as global node ids
            tree = np.repeat(np.arange(len(self.offset)), np.diff(np.r_[self.offset, len(self.feature)]))
            self._gleft = self.offset[tree] + self.left
            self._gright = self.offset[tree] + self.right
        res = np.empty(n)
        for o in self.offset:
            idx = np.arange(n)
            node = np.full(n, o, dtype=np.int64)
            while True:
                leaf = self.is_leaf[node]
                if leaf.any():
                    res[idx[leaf]] = self.value[node[leaf]]
                    keep = ~leaf
                    idx, node = idx[keep], node[keep]
                    if not len(idx):
                        break
                x = flat[base[idx] + self.feature[node]]
                go_left = np.where(np.isnan(x), self.missing_left[node], x <= self.threshold[node])
                node = np.where(go_left, self._gleft[node], self._gright[node])
            out += res
        return out

    def predict_proba(self, X) -> np.ndarray:
        r = self.raw(X)
        try:
            from scipy.special import expit
            p = expit(r)
        except Exception:  # pragma: no cover
            p = 1.0 / (1.0 + np.exp(-r))
        return np.column_stack([1.0 - p, p])


class AverageEnsemble:
    """Mean of the match probabilities of several rankers trained on the same 59 features
    (each a TreeEnsemble).  Every pair is scored by each member on its own, so rows stay independent."""

    def __init__(self, members):
        self.members = list(members)
        self.n_features_in_ = self.members[0].n_features_in_
        self.n_iter_ = sum(m.n_iter_ for m in self.members)

    def predict_proba(self, X) -> np.ndarray:
        return np.mean([m.predict_proba(X) for m in self.members], axis=0)


class RawAverage:
    """Mean of several rankers' raw scores (TreeEnsemble logits) for the same pairs; predict_proba applies the
    logistic function.  Each pair is scored on its own features."""

    def __init__(self, members):
        self.members = list(members)
        self.n_features_in_ = self.members[0].n_features_in_
        self.n_iter_ = sum(m.n_iter_ for m in self.members)

    def raw(self, X) -> np.ndarray:
        return np.mean([m.raw(X) for m in self.members], axis=0)

    def predict_proba(self, X) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-self.raw(X)))
        return np.column_stack([1.0 - p, p])


class ScriptRouter:
    """Scores Hangul records (feature 0 'hangul' = 1, the same for all candidates of a record) with one model and
    the other records with another.  Each pair is scored from its own features only."""

    def __init__(self, hangul_model, latin_model):
        self.hangul_model, self.latin_model = hangul_model, latin_model
        self.n_iter_ = (hangul_model.n_iter_, latin_model.n_iter_)

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X)
        h = X[:, 0] > 0
        out = np.zeros((len(X), 2))
        if h.any():
            out[h] = self.hangul_model.predict_proba(X[h])
        if (~h).any():
            out[~h] = self.latin_model.predict_proba(X[~h])
        return out


class LgbRaw:
    """A LightGBM model kept as its text dump (model_to_string), so model.pkl holds a str only; raw(X) loads the
    Booster on first use (lightgbm is in the evaluation image) and returns its raw scores."""

    def __init__(self, model_str: str):
        self.model_str = model_str
        self.n_features_in_ = len(FEATURES)
        self.n_iter_ = model_str.count("\nTree=")

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_booster"}

    def raw(self, X) -> np.ndarray:
        if getattr(self, "_booster", None) is None:
            import lightgbm as lgb
            self._booster = lgb.Booster(model_str=self.model_str)
        return self._booster.predict(np.asarray(X, dtype=np.float64), raw_score=True)


def _zscore(r: np.ndarray) -> np.ndarray:
    if len(r) < 2:
        return r
    sd = r.std()
    return (r - r.mean()) / (sd if sd > 1e-9 else 1.0)


class ListRouter:
    """Scores each record's candidate list (RANK_LGB): Hangul records by hangul_model unless none of the record's allowed
    candidates reaches HANGUL_SWITCH (w_hit_q >= 1: holds every word of the record's name), then by rank_model alone;
    Latin records by the sum of latin_model's and rank_model's scores, each standardised over the record's list.  Uses
    a record's own features and allowed mask only."""

    def __init__(self, hangul_model, latin_model, rank_model, switch=HANGUL_SWITCH):
        self.hangul_model, self.latin_model, self.rank_model = hangul_model, latin_model, rank_model
        self.switch_feature, self.switch_at = FEATURES.index(switch[0]), float(switch[1])
        self.n_iter_ = (hangul_model.n_iter_, latin_model.n_iter_, rank_model.n_iter_)

    def score_lists(self, feats, keep) -> list:
        X = np.vstack(feats)
        h = X[:, 0] > 0
        own = np.zeros(len(X))
        if h.any():
            own[h] = self.hangul_model.raw(X[h])
        if (~h).any():
            own[~h] = self.latin_model.raw(X[~h])
        rank = self.rank_model.raw(X)
        hrank = getattr(self, "hangul_rank_model", None)
        if hrank is not None and h.any():   # 10/31: a lambdarank trained on Hangul records only
            rank = rank.copy()
            rank[h] = hrank.raw(X[h])
        out, off = [], 0
        for f, k in zip(feats, keep):
            a, r = own[off: off + len(f)], rank[off: off + len(f)]
            off += len(f)
            if len(f) and f[0, 0] > 0:
                if HANGUL_MODE == "switch":
                    hit = np.nan_to_num(f[k, self.switch_feature], nan=-1.0) >= self.switch_at
                    out.append(a if hit.any() else r)
                else:
                    out.append(_combine(a, r, HANGUL_MODE))
            else:
                out.append(_combine(a, r, LATIN_MODE))
        return out


# 10/31: how ListRouter combines a record's two scores (the pointwise / listwise model's and lambdarank's), each
# standardised over the record's candidate list.  Hangul: "switch" (10/28 rule: pointwise unless no allowed candidate
# holds every word of the name, then lambdarank), "blend" (sum of the two z-scores), "rank" (lambdarank only), "own"
# (pointwise only), "w<k>" (z pointwise + k/10 * z lambdarank).  Latin: "blend" (10/28), "own" (listwise only), "rank",
# "w<k>".  Uses the record's own scores only.
HANGUL_MODE = "switch"
LATIN_MODE = "blend"


def _combine(a: np.ndarray, r: np.ndarray, mode: str) -> np.ndarray:
    if mode == "rank":
        return r
    if mode == "own":
        return a
    if mode.startswith("w"):
        return _zscore(a) + int(mode[1:]) / 10.0 * _zscore(r)
    return _zscore(a) + _zscore(r)


def _softmax_by_record(F, starts, rec):
    m = np.maximum.reduceat(F, starts)
    e = np.exp(F - m[rec])
    return e / np.add.reduceat(e, starts)[rec]


class ListwiseBooster:
    """Gradient-boosted trees for the listwise loss  L = -sum_q w_q log softmax(F_q)[true candidate].

    Each tree is grown by scikit-learn's own histogram tree grower (as in HistGradientBoosting: max_leaf_nodes,
    min_samples_leaf, l2_regularization, monotonic constraints) on the Newton gradients g = w (p - y) and
    hessians h = w p (1 - p), p = softmax of the record's candidate scores (LambdaMART-style boosting, Burges
    2010).  Records whose answer is not among the candidates are left out; validation_fraction of the records
    (at random) stop the boosting when their loss has not improved for n_iter_no_change trees.  Exposes the
    attributes TreeEnsemble reads, so the trees are saved as numpy arrays.  Training only (uses private
    scikit-learn modules, imported in fit); prediction goes through TreeEnsemble."""

    def __init__(self, max_iter=400, learning_rate=0.06, max_leaf_nodes=48, min_samples_leaf=40,
                 l2_regularization=1.0, monotonic_cst=None, max_bins=255, n_iter_no_change=40,
                 validation_fraction=0.1, random_state=0):
        self.max_iter, self.learning_rate = max_iter, learning_rate
        self.max_leaf_nodes, self.min_samples_leaf = max_leaf_nodes, min_samples_leaf
        self.l2_regularization, self.monotonic_cst, self.max_bins = l2_regularization, monotonic_cst, max_bins
        self.n_iter_no_change, self.validation_fraction = n_iter_no_change, validation_fraction
        self.random_state = random_state

    def fit(self, X, y, lens, record_weight):
        from sklearn.ensemble._hist_gradient_boosting._gradient_boosting import _update_raw_predictions
        from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
        from sklearn.ensemble._hist_gradient_boosting.common import G_H_DTYPE
        from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
        from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        lens = np.asarray(lens, dtype=np.int64)
        qw = np.asarray(record_weight, dtype=np.float64)
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
        rec = np.repeat(np.arange(len(lens)), lens)
        has_pos = np.add.reduceat(y, starts) > 0
        rng = np.random.default_rng(self.random_state)
        val = (rng.random(len(lens)) < self.validation_fraction) & has_pos
        trn = has_pos & ~val
        nt = _openmp_effective_n_threads()
        self.n_features_in_ = X.shape[1]
        bm = _BinMapper(n_bins=self.max_bins + 1, random_state=self.random_state, n_threads=nt)
        Xb = bm.fit_transform(X[trn[rec]])
        Xv = bm.transform(X[val[rec]]) if val.any() else None

        def part(mask):
            ln = lens[mask]
            st = np.concatenate([[0], np.cumsum(ln)[:-1]]).astype(np.int64)
            rc = np.repeat(np.arange(len(ln)), ln)
            return y[mask[rec]], st, rc, qw[mask][rc]

        yt, st, rt, wt = part(trn)
        has_missing = (Xb == bm.missing_values_bin_idx_).any(axis=0).astype(np.uint8)
        cst = None if self.monotonic_cst is None else np.asarray(self.monotonic_cst, dtype=np.int8)
        F = np.zeros(len(yt))
        if Xv is not None:
            yv, sv, rv, wv = part(val)
            Fv = np.zeros(len(yv))
        preds, best, best_it = [], np.inf, 0
        grng = np.random.default_rng(self.random_state)
        for it in range(self.max_iter):
            p = _softmax_by_record(F, st, rt)
            g = (wt * (p - yt)).astype(G_H_DTYPE)
            h = (wt * np.maximum(p * (1.0 - p), 1e-16)).astype(G_H_DTYPE)
            grower = TreeGrower(X_binned=Xb, gradients=g, hessians=h, max_leaf_nodes=self.max_leaf_nodes,
                                min_samples_leaf=self.min_samples_leaf, n_bins=self.max_bins + 1,
                                n_bins_non_missing=bm.n_bins_non_missing_, has_missing_values=has_missing,
                                monotonic_cst=cst, l2_regularization=self.l2_regularization, rng=grng,
                                shrinkage=self.learning_rate, n_threads=nt)
            grower.grow()
            pr = grower.make_predictor(binning_thresholds=bm.bin_thresholds_)
            preds.append([pr])
            _update_raw_predictions(F, grower, nt)
            if Xv is None:
                continue
            Fv += pr.predict_binned(Xv, bm.missing_values_bin_idx_, nt)
            pv = _softmax_by_record(Fv, sv, rv)
            loss = float(-(wv * yv * np.log(np.maximum(pv, 1e-300))).sum() / wv[yv > 0].sum())
            if loss < best - 1e-7:
                best, best_it = loss, it
            elif it - best_it >= self.n_iter_no_change:
                break
        keep = best_it + 1 if Xv is not None else len(preds)
        self._predictors = preds[:keep]
        self.n_iter_ = keep
        self.n_trees_per_iteration_ = 1
        self._baseline_prediction = np.zeros((1, 1))
        return self


def _fit_latin_listwise(X, y, w, lens, seed: int = 0):
    """Listwise ranker on the Latin records' pairs (rows of a record are contiguous in X)."""
    lens = np.asarray(lens, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
    latin = X[starts, 0] <= 0
    rows = np.repeat(latin, lens)
    Xl = np.array(X[rows], dtype=np.float64)
    Xl[:, np.isnan(Xl).all(axis=0)] = 0.0  # Hangul-only features: constant for Latin records, never split on
    cst = [1 if f in MONO_UP else -1 if f in MONO_DOWN else 0 for f in FEATURES] if MONOTONIC else None
    b = ListwiseBooster(monotonic_cst=cst, **{**LIST_PARAMS, "random_state": seed}).fit(
        Xl, np.asarray(y)[rows], lens[latin], np.asarray(w)[starts][latin])
    return TreeEnsemble(b)


def _fit_rank(X, y, w, lens):
    """LightGBM lambdarank (RANK_PARAMS, RANK_TREES) on every record's pairs; records whose answer is not among the
    candidates carry no ranking signal and are left out."""
    import lightgbm as lgb
    lens = np.asarray(lens, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
    keep = np.add.reduceat(np.asarray(y), starts) > 0
    rows = np.repeat(keep, lens)
    params = dict(RANK_PARAMS, monotone_constraints=[1 if f in MONO_UP else -1 if f in MONO_DOWN else 0
                                                     for f in FEATURES])
    ds = lgb.Dataset(np.asarray(X)[rows], label=np.asarray(y)[rows], weight=np.asarray(w)[rows], group=lens[keep])
    return LgbRaw(lgb.train(params, ds, num_boost_round=RANK_TREES).model_to_string())


def _for_numpy1(model):
    """Drop fit-only numpy state so a pickle written under numpy 2.x loads under numpy 1.x
    (the baseline pickle suggests the server runs numpy 1.x).  Predictions are unchanged."""
    model.__dict__.pop("_feature_subsample_rng", None)
    if isinstance(getattr(model, "_random_seed", None), np.integer):
        model._random_seed = int(model._random_seed)
    if isinstance(getattr(model._bin_mapper, "random_state", None), np.integer):
        model._bin_mapper.random_state = int(model._bin_mapper.random_state)
    return model


def _fit_router(train: pd.DataFrame, base: pd.DataFrame, cat: Catalog, pri: Priors, cst):
    """fit()'s rankers on the training pairs retrieved with the current matcher settings: one pointwise model per seed
    (Hangul, raw scores averaged), the first seed's listwise model (Latin) and lambdarank model, in a ListRouter."""
    import augment
    hangul_models, latin_models, rank_models = [], [], []
    bf, bc = _train_features(base, cat, pri)
    seed0 = augment.AUG_SEED
    for s in SEEDS:
        augment.AUG_SEED = seed0 + 1000 * s
        try:
            extra = augment.augment(train, cat) if AUGMENT else train.iloc[:0]
        finally:
            augment.AUG_SEED = seed0
        ef, ec = _train_features(base, cat, pri, extra, base_rows=False) if len(extra) else ([], [])
        rows = pd.concat([base, extra], ignore_index=True)
        cands = bc + ec
        true = rows["target_vessel_id"].map(cat.id_to_index).to_numpy()
        X = np.vstack(bf + ef)
        y = np.concatenate([(c == t).astype(int) for c, t in zip(cands, true)])
        lens = [len(c) for c in cands]
        w = pair_weights(rows, lens, X)
        model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=48,
                                               min_samples_leaf=40, l2_regularization=1.0, random_state=s,
                                               early_stopping=EARLY_STOPPING, monotonic_cst=cst)
        model.fit(X, y, sample_weight=w)
        hangul_models.append(TreeEnsemble(model) if EXPORT_TREES else _for_numpy1(model))
        if LATIN_RANKER == "listwise" and (LATIN_AVERAGE or not latin_models):
            latin_models.append(_fit_latin_listwise(X, y, w, lens, seed=s))
        if RANK_LGB and not rank_models:
            rank_models.append(_fit_rank(X, y, w, lens))
        del X, y, w, model
    one = len(SEEDS) == 1
    ranker = hangul_models[0] if one else RawAverage(hangul_models)
    if LATIN_RANKER == "listwise":
        latin = latin_models[0] if len(latin_models) == 1 else RawAverage(latin_models)
        ranker = ListRouter(ranker, latin, rank_models[0]) if RANK_LGB else ScriptRouter(ranker, latin)
    return ranker


def fit(train: pd.DataFrame, cat: Catalog, callsign: bool = False):
    """Train the ranker(s), and with RENAME_PASS the second-pass rankers (exact-tonnage block on, renamed copies).
    callsign=True also stores the train call_sign -> vessel table (only needed for predict(..., use_callsign=True); off
    while that rule is unclear)."""
    import augment
    train = train.reset_index(drop=True)
    base = train
    if LLM_HANGUL:
        base = pd.concat([train, llm_records(train, Path(__file__).resolve().parent / LLM_HANGUL)], ignore_index=True)
    pri = Priors(base, cat)
    saved = matcher.V2_CHANNEL, matcher.NOTON_TYPE_BAND
    if not RETRIEVAL_1024_IN_TRAIN:
        matcher.V2_CHANNEL = matcher.NOTON_TYPE_BAND = False
    cst = [1 if f in MONO_UP else -1 if f in MONO_DOWN else 0 for f in FEATURES] if MONOTONIC else None
    rename = None
    try:
        ranker = _fit_router(train, base, cat, pri, cst)
        if RENAME_PASS is not None:
            saved2 = matcher.EXACT_CHANNEL, augment.AUG_RENAME
            matcher.EXACT_CHANNEL, augment.AUG_RENAME = True, RENAME_AUG
            try:
                rename = _fit_router(train, base, cat, pri, cst)
            finally:
                matcher.EXACT_CHANNEL, augment.AUG_RENAME = saved2
    finally:
        matcher.V2_CHANNEL, matcher.NOTON_TYPE_BAND = saved
    bundle = {"model": ranker, "priors": pri,
              "train_vessels": sorted(set(train["target_vessel_id"].astype(str)))}
    if rename is not None:
        bundle["rename_model"] = rename
    if callsign:
        bundle["callsign"] = dict(zip(train["call_sign"].astype(str), train["target_vessel_id"].astype(str)))
    return bundle


def score_candidates(bundle, df: pd.DataFrame, cat: Catalog, chunk: int = 500):
    """Candidate catalog indices and their scores for every row of df: match probabilities, or for a ListRouter the
    list scores given each record's allowed candidates (allowed(), tonnage gaps from its features).  With RENAME_PASS
    and a "rename_model" in the bundle, a record whose allowed candidates all match its name weakly gets the candidates
    and scores of the second pass (_second_pass()) when the second pass's best allowed candidate is a vessel its first
    list did not have (one only the exact-tonnage block found); otherwise it keeps the first pass.

    Each row is scored on its own (chunks only batch the matrix work)."""
    pri, model = bundle["priors"], bundle["model"]
    rename = bundle.get("rename_model") if RENAME_PASS is not None else None
    cands_all, scores_all = [], []
    for s in range(0, len(df), chunk):
        part = df.iloc[s: s + chunk]
        q = QueryBatch(part, cat, pri)
        feats, cands = features(q, cat, pri, retrieve(q, cat))
        if hasattr(model, "score_lists"):
            gaps = [(f[:, GAP_MIN], f[:, CGT_MISSING] > 0.5) for f in feats]
            keep = allowed(bundle, cands, cat, gaps=gaps)
            cands, scores = list(cands), model.score_lists(feats, keep)
            if rename is not None:
                j = FEATURES.index(RENAME_PASS[0])
                weak = [k for k, (f, ok) in enumerate(zip(feats, keep))
                        if not (np.nan_to_num(f[ok, j], nan=-1.0) >= RENAME_PASS[1]).any()]
                if weak:
                    for k, c, sc, ok in zip(weak, *_second_pass(bundle, rename, part.iloc[weak], cat)):
                        best = int(np.asarray(c)[ok][int(np.argmax(np.asarray(sc)[ok]))])
                        if best not in set(np.asarray(cands[k]).tolist()):   # found only by the exact-tonnage block
                            cands[k], scores[k] = c, sc
            cands_all += cands
            scores_all += scores
            continue
        flat = model.predict_proba(np.vstack(feats))[:, 1]
        off = 0
        for c in cands:
            cands_all.append(c)
            scores_all.append(flat[off: off + len(c)])
            off += len(c)
    return cands_all, scores_all


def _second_pass(bundle, model, df: pd.DataFrame, cat: Catalog):
    """RENAME_PASS: the records' candidates retrieved with the exact-tonnage block (matcher.EXACT_CHANNEL), their list
    scores by the second-pass model and their allowed() masks."""
    pri = bundle["priors"]
    saved = matcher.EXACT_CHANNEL
    matcher.EXACT_CHANNEL = True
    try:
        q = QueryBatch(df, cat, pri)
        feats, cands = features(q, cat, pri, retrieve(q, cat))
    finally:
        matcher.EXACT_CHANNEL = saved
    gaps = [(f[:, GAP_MIN], f[:, CGT_MISSING] > 0.5) for f in feats]
    keep = allowed(bundle, cands, cat, gaps=gaps)
    return list(cands), model.score_lists(feats, keep), keep


def tonnage_gaps(df: pd.DataFrame, cat: Catalog, cands) -> list:
    """Per record: the candidates' tonnage gap (gap_min of the features, float32 as there) and whether the catalog
    has no tonnage (c_gt_missing)."""
    out = []
    for g, ig, c in zip(df["gross_tonnage"], df["international_gross_tonnage"], cands):
        c = np.asarray(c)
        g, ig, cg = matcher.positive(g), matcher.positive(ig), cat.gt[c]
        gap = np.array([min(matcher.rel_gap(g, x), matcher.rel_gap(ig, x)) for x in cg], dtype=np.float32)
        out.append((gap, ~(cg > 0)))
    return out


def allowed(bundle, cands, cat: Catalog, exclude=None, gaps=None) -> list:
    """Per record, the candidates that may be its answer (bool mask): with EXCLUDE_TRAIN_VESSELS not a train answer
    vessel; with gaps (tonnage_gaps()) and TON_GATE, none further off than TON_GATE when one of those is within it
    (candidates without a catalog tonnage stay); every candidate when that leaves none.  Per record only."""
    exclude = EXCLUDE_TRAIN_VESSELS if exclude is None else exclude
    seen = np.zeros(len(cat.ids), dtype=bool)
    if exclude and bundle.get("train_vessels"):
        seen[[cat.id_to_index[v] for v in bundle["train_vessels"] if v in cat.id_to_index]] = True
    out = []
    for k, c in enumerate(cands):
        ok = ~seen[np.asarray(c, dtype=np.int64)]
        if gaps is not None and TON_GATE is not None:
            gap, unknown = gaps[k]
            near = gap <= TON_GATE
            if (near & ok).any():
                ok &= near | unknown
        out.append(ok if ok.any() else np.ones(len(ok), dtype=bool))
    return out


def pick(bundle, cands, scores, cat: Catalog, exclude=None, gaps=None) -> list:
    """Catalog index of each record's answer: its best-scored allowed candidate (allowed()).  Per record only."""
    return [int(np.asarray(c)[k][int(np.argmax(np.asarray(s)[k]))])
            for c, s, k in zip(cands, scores, allowed(bundle, cands, cat, exclude, gaps))]


def predict(bundle, test: pd.DataFrame, cat: Catalog, use_callsign: bool = False) -> list:
    if use_callsign and "callsign" not in bundle:
        raise ValueError("model.pkl has no call_sign table; train it with fit(..., callsign=True)")
    cands, scores = score_candidates(bundle, test, cat)
    gaps = tonnage_gaps(test, cat, cands) if TON_GATE is not None else None
    out = [cat.ids[i] for i in pick(bundle, cands, scores, cat, gaps=gaps)]
    if use_callsign and "call_sign" in test.columns:
        known = bundle["callsign"]
        out = [known.get(str(cs), p) if str(cs) in known and known[str(cs)] in cat.id_to_index else p
               for cs, p in zip(test["call_sign"], out)]
    return out
