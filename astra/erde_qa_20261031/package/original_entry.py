"""DACON 항만 선박 매칭 v42 - inference entry point (v38 + typo-robust fuzzy candidate retrieval for hangul names, pipe23; scores are raw model outputs, z-scored per row and averaged; v38 = per-script ranker sets) (v33 + borrowed-name features: whether the record name equals /
is close to a name already carried by a train.csv vessel; train vessels stay in the candidate set and are only
skipped at the final pick, as in v31/v33) + train-presence features: which candidates are train vessels, whether
the name fits a train vessel better than any eligible one, and each candidate's gaps to the best eligible one).

Each test row is processed independently:
  * english names: fuzzy / partial name + tonnage + type/flag retrieval
  * hangul names : learned hangul->english transliteration + Korean romanisation (global + word-skipping
                   local score, EM-fitted on train.csv) + token-level standard-orthography spellings +
                   tonnage retrieval
then LightGBM lambdarank model(s) pick one catalog vessel_id per row (scores of several rankers, if present,
are averaged per row after standardising them over that row's candidates).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "model"))

import lightgbm as lgb  # noqa: E402
import pipe23 as pipe4  # noqa: E402
import translit as T  # noqa: E402


def find(rel: str) -> Path:
    for root in (Path.cwd(), BASE):
        for d in ("data", "open"):
            p = root / d / rel
            if p.exists():
                return p
    raise FileNotFoundError(rel)


def load_rankers():
    """model/ranker_<nn>[_H|_E|_HE].txt, used in file-name order. A name ending in _H / _E / _HE is used only for
    hangul / non-hangul / both kinds of rows (rows with such rankers ignore the others); no suffix = every row."""
    out = []
    for path in sorted((BASE / "model").glob("ranker*.txt")):
        fj = path.with_name(path.name + ".features.json")
        if not fj.exists():
            fj = BASE / "model" / "features.json"
        fnames = json.load(open(fj))
        suf = path.stem.split("_")[-1]
        scope = {"H": ("H",), "E": ("E",), "HE": ("H", "E")}.get(suf, ("all",))
        out.append((lgb.Booster(model_file=str(path)), [pipe4.FEATURES7.index(f) for f in fnames], scope))
    return out


def row_scores(rankers, X, hang: bool):
    own = [r for r in rankers if ("H" if hang else "E") in r[2]]
    use = own or [r for r in rankers if "all" in r[2]]
    if len(use) == 1:
        m, cols, _ = use[0]
        return m.predict(X[:, cols], num_threads=1, raw_score=True)
    scores = np.zeros(X.shape[0])
    for m, cols, _ in use:
        s = m.predict(X[:, cols], num_threads=1, raw_score=True)
        sd = s.std()
        scores += (s - s.mean()) / (sd if sd > 0 else 1.0)
    return scores


def main():
    t0 = time.time()
    test = pd.read_csv(find("test.csv"), encoding="utf-8-sig")
    catalog = pd.read_csv(find("vessel_catalog.csv"), encoding="utf-8-sig")
    train = pd.read_csv(BASE / "model" / "train.csv", encoding="utf-8-sig")

    index = pipe4.CatalogIndex(catalog)
    stats = pipe4.TrainStats(train, catalog)
    td = T.Transducer(catalog["vessel_name"].tolist())
    lm = T.LetterLM(td.letters)
    lmv = np.array([lm.logprob(s) for s in td.letters])
    pipe4.fit_transducer(td, train, index)
    train_ids = set(train["target_vessel_id"])
    matcher = pipe4.Matcher23(index, stats, td, lmv, word_dict=pipe4.build_word_dict(train, catalog),
                              excluded=None, train_names=train["vessel_name"].tolist(),
                              train_mask=np.isin(index.ids, list(train_ids)))
    rankers = load_rankers()
    print(f"ready in {time.time() - t0:.1f}s, {len(rankers)} ranker(s)", flush=True)

    # Features are built row by row (each row only sees its own values); the rankers then score a block of rows
    # in one call per ranker, which is the same arithmetic per row as scoring each row alone, only faster.
    ih = pipe4.FEATURES7.index("is_hangul")
    rows = test.to_dict("records")
    preds = []
    CH = 256
    for c0 in range(0, len(rows), CH):
        block = [matcher.features(row) for row in rows[c0:c0 + CH]]
        hang = [bool(X[0, ih] > 0.5) for _, X in block]
        sizes = [len(c) for c, _ in block]
        cuts = np.concatenate([[0], np.cumsum(sizes)])
        Xb = np.vstack([X for _, X in block])
        use = {}
        for sc in ("H", "E"):
            own = [i for i, r in enumerate(rankers) if sc in r[2]]
            use[sc] = own or [i for i, r in enumerate(rankers) if "all" in r[2]]
        hrow = np.repeat(np.array(hang, bool), sizes)
        raw = {}
        for i, (m, cols, scope) in enumerate(rankers):
            need = (hrow & (i in use["H"])) | (~hrow & (i in use["E"]))
            if need.any():
                r = np.full(len(Xb), np.nan)
                r[need] = m.predict(Xb[need][:, cols], num_threads=1, raw_score=True)
                raw[i] = r
        for j, (cands, X) in enumerate(block):
            ids_use = use["H" if hang[j] else "E"]
            a, b = cuts[j], cuts[j + 1]
            if len(ids_use) == 1:
                scores = raw[ids_use[0]][a:b]
            else:
                scores = np.zeros(b - a)
                for i in ids_use:
                    sc = raw[i][a:b]
                    sd = sc.std()
                    scores += (sc - sc.mean()) / (sd if sd > 0 else 1.0)
            order = np.argsort(-scores, kind="stable")
            pick = order[0]
            for q in order:
                if index.ids[cands[q]] not in train_ids:
                    pick = q
                    break
            preds.append(index.ids[cands[pick]])
        print(f"{min(c0 + CH, len(rows))}/{len(rows)} {time.time() - t0:.1f}s", flush=True)

    out_dir = Path.cwd() / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": test["sample_id"], "target_vessel_id": preds}).to_csv(
        out_dir / "submission.csv", index=False, encoding="utf-8")
    print(f"saved {len(preds)} rows in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
