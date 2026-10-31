# author: terra
"""Aggregate completed dictionary-only diagnostics and check their boundaries."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    rows = []
    seen = set()
    for k in (0, 1, 2):
        path = args.root / f"fold{k}_stress"
        summary = json.loads((path / "summary.json").read_text())
        assert "holdout_metrics" in summary, f"fold {k} not complete"
        r = pd.read_csv(path / "holdout_name_ranks.csv")
        assert not r.duplicated(["sample_id", "scenario", "variant"]).any()
        ids = set(r.sample_id)
        assert not seen & ids
        seen |= ids
        selected = pd.read_csv(path / "fold_rescored_augmentation.csv")
        assert not set(selected.vessel_id) & set(r.vessel_id)
        assert set(r.variant) == {"real_only", "v1", "v12"}
        assert len(r) == len(ids) * 6 * 3
        r["fold"] = k
        r["hit"] = r["rank"] <= 1
        for m in summary["holdout_metrics"].values():
            assert 0 <= m["top1"] <= m["top5"] <= m["top40"] <= m["pool_recall"] <= 1
        rows.append(r)
    data = pd.concat(rows, ignore_index=True)
    metrics = []
    for (scenario, variant), d in data.groupby(["scenario", "variant"]):
        macro = d.groupby("vessel_id").hit.mean()
        changed = d[d.changed_query]
        metrics.append({"scenario": scenario, "variant": variant, "rows": len(d),
                        "vessels": len(macro), "row_top1": d.hit.mean(),
                        "ship_macro_top1": macro.mean(), "top5": (d["rank"] <= 5).mean(),
                        "top40": (d["rank"] <= 40).mean(), "changed_rows": len(changed),
                        "changed_row_top1": changed.hit.mean() if len(changed) else None})
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(args.root / "aggregate_metrics.csv", index=False)
    paired = []
    for scenario, d in data.groupby("scenario"):
        ships = d.groupby(["vessel_id", "variant"]).hit.mean().unstack()
        delta = (ships.v12 - ships.v1).to_numpy()
        rng = np.random.default_rng(42)
        boot = delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(axis=1)
        lo, hi = np.quantile(boot, [.025, .975])
        paired.append({"scenario": scenario, "independent_vessels": len(delta),
                       "v12_minus_v1_ship_macro": delta.mean(), "bootstrap_low": lo,
                       "bootstrap_high": hi,
                       "vessels_better": int((delta > 0).sum()), "vessels_worse": int((delta < 0).sum())})
    paired = pd.DataFrame(paired)
    paired.to_csv(args.root / "paired_vessel_differences.csv", index=False)
    # Few-shot coverage assertion independent of the evaluator's assertion.
    project = Path(__file__).resolve().parents[2]
    fs = set(pd.read_csv(project / "shared/llm_aug/llm_fewshot.csv").vessel_id)
    fs |= set(pd.read_csv(project / "shared/llm_aug/llm_fewshot_sets.csv").vessel_id)
    assert not fs & set(data.vessel_id)
    qa = {"status": "PASS", "folds": [0, 1, 2], "original_rows": len(seen),
          "independent_vessels": data.vessel_id.nunique(), "scenarios": 6,
          "dictionary_variants": 3, "total_row_variant_evaluations": len(data),
          "boundary": "dictionary ranking only; not final ranker CV, GPU runtime, or Public performance",
          "checks": ["disjoint held-out vessels vs augmentation", "no held-out prompt examples",
                     "nonoverlapping held-out rows across folds", "complete paired evaluation",
                     "monotonic rank metrics"],
          "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in Path(__file__).parent.glob("*audit*terra.py")}}
    (args.root / "qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False))
    print(metrics.pivot(index="scenario", columns="variant", values="row_top1").round(4).to_string())
    print(paired.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
