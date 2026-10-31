# author: opus
"""113: 110 + Korean rankers retrained with the LLM-Hangul catalog features (pinned/build113 = build70 recipe +
kc_sim / kc_margin from shared/src/kocat.py), with the top-K KoCat matches added as Korean candidates.
GATE: Korean rows where 110's six-ranker mean gap is >= GATE keep 110's answer (Public precision 371/373 there);
"none" applies the new rankers to every Korean row. English path, dictionary, stats are 93's; 110's rankers stay
as model/ranker_*.txt, the new ones are model/kc113_*.txt.

usage: python3 build_113_kocat_rank.py NUM TABLE.csv OUT_A OUT_B GATE [K=50]
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(num, table, out_a, out_b, gate, k=50):
    new = [Path(out_a) / "model" / f"ranker_{i}.txt" for i in range(3)] + \
          [Path(out_b) / "model" / f"ranker_{i}.txt" for i in range(3)]
    assert all(p.exists() for p in new)
    gate_val = "None" if gate == "none" else str(float(gate))
    edits = [
        (b'"""DACON inference entry point (110_submit: ',
         b'"""DACON inference entry point (%d_submit: 110 + Korean rankers retrained with LLM-Hangul catalog features, '
         b'KoCat top-%d, gate %s; base ' % (num, k, gate_val.encode())),
        (b"    fx = FuzzyExt(ix, catalog)\n",
         b"    fx = FuzzyExt(ix, catalog)\n    from kocat import KoCat, kc_features\n"
         b"    kc = KoCat(catalog, HERE / \"model\" / \"catalog_ko.csv\")\n"
         b"    kc_models = [lgb.Booster(model_file=str(p)) for p in sorted((HERE / \"model\").glob(\"kc113_*.txt\"))]\n"
         b"    assert len(kc_models) == 6\n    GATE = %s\n" % gate_val.encode()),
        (b"            score = sum(w * sum(m.predict(feats[fam]) for m in models[fam]) for fam, w in KO.items())\n",
         b"            score = sum(w * sum(m.predict(feats[fam]) for m in models[fam]) for fam, w in KO.items())\n"
         b"            use_kc = True\n"
         b"            if GATE is not None:\n"
         b"                o110 = sorted(range(len(idx)), key=lambda j: (-float(score[j]), idx[j]))\n"
         b"                a110 = [j for j in o110 if idx[j] not in train_targets] or o110\n"
         b"                g110 = float(score[a110[0]] - score[a110[1]]) / len(models[\"ranker\"]) if len(a110) > 1 else 99.0\n"
         b"                use_kc = g110 < GATE\n"
         b"            if use_kc:\n"
         b"                cand = fx.add(L.query_candidates(ix, row), row)\n"
         b"                for rk, (ci, _) in enumerate(kc.search(q, %d)):\n"
         b"                    e = cand.setdefault(int(ci), [999, 999, 999])\n"
         b"                    e[0] = min(e[0], rk)\n"
         b"                idx, F62 = L.query_features(ix, row, cand)\n"
         b"                F62 = np.hstack([F62, kc_features(kc, q, idx)])\n"
         b"                score = sum(m.predict(F62) for m in kc_models)\n" % k),
    ]
    out = ROOT / "submissions" / f"{num}_submit.zip"
    assert not out.exists(), out
    z = zipfile.ZipFile(ROOT / "submissions/110_submit.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as wz:
        for info in z.infolist():
            data = z.read(info)
            if info.filename == "script.py":
                for a, b in edits:
                    assert data.count(a) == 1, a
                    data = data.replace(a, b)
            wz.writestr(info, data)
        for i, p in enumerate(new):
            wz.write(p, f"model/kc113_{i}.txt")
        wz.write(ROOT / "shared/src/kocat.py", "model/kocat.py")
        wz.write(table, "model/catalog_ko.csv")
    print(num, round(out.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], *(int(a) for a in sys.argv[6:]))
