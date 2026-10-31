# author: opus
"""Shared CV split + metric. Both agents must use this so numbers are comparable."""
from pathlib import Path
import numpy as np
import pandas as pd

# The project root is the first ancestor holding data/open (also works from pinned copies).
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data" / "open").is_dir())
DATA = ROOT / "data" / "open"


def load():
    train = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig")
    catalog = pd.read_csv(DATA / "vessel_catalog.csv", encoding="utf-8-sig")
    return train, catalog


def is_hangul(s) -> bool:
    return isinstance(s, str) and any("가" <= c <= "힣" for c in s)


def folds(train: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Fold id per row, grouped by vessel (target_vessel_id) so a vessel never spans folds.
    Stratified-ish: hangul vessels are spread evenly across folds."""
    rng = np.random.default_rng(seed)
    g = train.groupby("target_vessel_id")["vessel_name"].agg(lambda s: any(map(is_hangul, s)))
    fold_of = {}
    for flag in (True, False):
        ids = np.array(sorted(g.index[g.values == flag]))
        rng.shuffle(ids)
        for i, v in enumerate(ids):
            fold_of[v] = i % n_splits
    return train["target_vessel_id"].map(fold_of).to_numpy()


def score(names, truth, pred) -> dict:
    h = np.array([is_hangul(n) for n in names])
    c = np.asarray(truth) == np.asarray(pred)
    hk, nh = float(c[h].mean()), float(c[~h].mean())
    return {"weighted": 0.7 * hk + 0.3 * nh, "hangul": hk, "non_hangul": nh}


def folds_rows(train: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Row-level random folds: the same vessel can sit in train and validation,
    like a test set that revisits known ships."""
    return np.random.default_rng(seed).permutation(len(train)) % n_splits
