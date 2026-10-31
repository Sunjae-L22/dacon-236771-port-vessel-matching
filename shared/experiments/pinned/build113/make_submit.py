# author: opus
"""Train on all of train.csv and write submit/ + submit.zip in DACON's required layout."""
import os
import pickle
import shutil
import sys
import zipfile
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cv import load  # noqa: E402
import opus_lib as L  # noqa: E402
from opus_cv import PARAMS  # noqa: E402

from cv import ROOT  # noqa: E402
OUT = ROOT / os.environ.get("OUT_DIR", "submit")  # separate dirs let builds run in parallel

SCRIPT = '''"""DACON inference entry point (opus/argon ranker)."""
from pathlib import Path
import os
import pickle
import sys

if os.environ.get("PYTHONHASHSEED") is None:  # same predictions in every process
    os.execve(sys.executable, [sys.executable] + sys.argv, dict(os.environ, PYTHONHASHSEED="0"))

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "model"))
import opus_lib as L


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
    models = [lgb.Booster(model_file=str(p)) for p in sorted((HERE / "model").glob("ranker*.txt"))]
    preds = []
    for row in test.to_dict("records"):
        cs = ix.cs_map.get(row.get("call_sign"))
        if cs is not None:
            preds.append(ix.ids[cs])
            continue
        idx, F = L.query_features(ix, row, L.query_candidates(ix, row))
        score = sum(m.predict(F) for m in models)
        preds.append(ix.ids[idx[min(range(len(idx)), key=lambda j: (-score[j], idx[j]))]])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": test["sample_id"], "target_vessel_id": preds}).to_csv(out, index=False)
    print(f"Saved {len(preds)} predictions to {out}")


if __name__ == "__main__":
    run()
'''


def main(rounds=600):
    # Read the library sources now: training takes minutes and the files may be edited meanwhile.
    sources = {name: (Path(__file__).parent / name).read_text(encoding="utf-8")
               for name in ("opus_lib.py", "ko_translit_argon.py")}
    train, catalog = load()
    ix = L.Index(catalog, train)
    # Every scored build so far used Argon's universal dictionary; 42 lost it by omission, so require it.
    assert os.environ.get("KO_DICT") == "1", "set KO_DICT=1 (builds 08-39 used it)"
    # The KoMatcher dictionary and the ranker's training rows can take different LLM spellings.
    here = Path(__file__).parent
    llm = L.llm_korean_rows(train, here / os.environ.get("LLM_TRAIN_CSV", "llm_korean_scored.csv"))
    llm_dict = L.llm_korean_rows(train, here / os.environ.get("LLM_DICT_CSV", "llm_korean_scored.csv"))
    ix.ko = L.KA.KoMatcher(catalog, pd.concat([train, llm_dict], ignore_index=True))
    print(f"LLM Korean rows: ranker {len(llm)}, dictionary {len(llm_dict)}", flush=True)
    import numpy as np
    from kocat import KC_FEATURES, KoCat, kc_features
    kc = KoCat(catalog, here / os.environ["KOCAT_TABLE"])
    # SEEDS=n trains n rankers on differently-seeded augmentation; the script sums their scores.
    params = dict(PARAMS)
    if os.environ.get("MONO") == "1":  # better name/tonnage agreement may never lower a score
        from opus_tune import DOWN, META_UP, NAME_UP, mono
        params.update(monotone_constraints=mono(NAME_UP + META_UP, DOWN), monotone_constraints_method="advanced")
    models = []
    for seed in range(int(os.environ.get("SEEDS", "1"))):
        aug = seed * 1000 + int(os.environ.get("SEED_OFFSET", "0"))  # another draw of the same recipe
        df = pd.concat([L.augment_korean(train, aug), L.augment_corrupt(train, aug),
                        L.augment_rename(train, aug), llm], ignore_index=True)
        X, y, g, cands = L.build(ix, df, True)
        # 113: LLM-Hangul catalog similarity (kocat.py); NaN on LLM-written Korean rows (see train_kocat.py)
        extra = [np.full((len(c), 2), np.nan, np.float32) if str(sid).startswith("LLM_") else kc_features(kc, nm, c)
                 for sid, nm, c in zip(df["sample_id"], df["vessel_name"], cands)]
        X = np.hstack([X, np.concatenate(extra)])
        models.append(lgb.train(params, lgb.Dataset(X, y, group=g, feature_name=L.FEATURES + KC_FEATURES), rounds))
        print(f"trained seed {seed}", flush=True)
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "model").mkdir(parents=True)
    for seed, model in enumerate(models):
        model.save_model(str(OUT / "model" / f"ranker_{seed}.txt"))
    stats = {"cs_map": {k: ix.ids[v] for k, v in ix.cs_map.items()}, "flag_p": ix.flag_p, "type_p": ix.type_p,
             "ko_pairs": pd.concat([train.loc[train["vessel_name"].map(L.is_hangul)], llm_dict])[
                 ["vessel_name", "target_vessel_id"]].to_dict("records"),
             "alias_rows": train[["sample_id", "vessel_name", "target_vessel_id"]].to_dict("records")}
    with open(OUT / "model" / "stats.pkl", "wb") as f:
        pickle.dump(stats, f)
    for name, text in sources.items():
        (OUT / "model" / name).write_text(text, encoding="utf-8")
    # The server has no KO_DICT env var: bake the training-time setting into the shipped code.
    lib = OUT / "model" / "opus_lib.py"
    default = os.environ.get("KO_DICT", "0")
    lib.write_text(lib.read_text(encoding="utf-8").replace(
        'os.environ.get("KO_DICT", "0")', f'os.environ.get("KO_DICT", "{default}")'), encoding="utf-8")
    (OUT / "script.py").write_text(SCRIPT, encoding="utf-8")
    (OUT / "requirements.txt").write_text("# uses preinstalled lightgbm, rapidfuzz, numpy, pandas\n", encoding="utf-8")
    (ROOT / "submissions").mkdir(exist_ok=True)
    done = sorted((ROOT / "submissions").glob("[0-9][0-9]_submit.zip"))
    num = int(os.environ.get("SUBMIT_NUM", 0)) or (int(done[-1].name[:2]) + 1 if done else 1)
    zp = ROOT / "submissions" / f"{num:02d}_submit.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                z.write(p, p.relative_to(OUT))
    print("wrote", zp)


if __name__ == "__main__":
    main()
