# author: sol
"""Choose w/K using tune only, then report untouched eval and the +2pp/orig1.000 gate."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_K = (10, 20)
DEFAULT_W = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def validate(frame, expected_candidates=None):
    required = {"query_sample_id", "orig_sample_id", "candidate_vessel_id", "target_vessel_id", "rank", "ranker_score",
                "llm_logprob", "llm_model", "role", "variant"}
    assert required.issubset(frame.columns), f"missing {required - set(frame.columns)}"
    assert not frame[["query_sample_id", "candidate_vessel_id"]].duplicated().any()
    assert set(frame["role"]) == {"tune", "eval"}
    assert frame["llm_model"].nunique() == 1, "evaluate one model at a time"
    assert np.isfinite(frame[["rank", "ranker_score", "llm_logprob"]].to_numpy()).all()
    assert frame.groupby("query_sample_id")["rank"].apply(lambda s: sorted(s.astype(int)) == list(range(1, len(s) + 1))).all()
    assert frame.groupby("query_sample_id").size().eq(20).all(), "incomplete top-20 scores: finish scoring before evaluation"
    for col in ["role", "variant", "target_vessel_id"]:
        assert (frame.groupby("query_sample_id")[col].nunique() == 1).all()
    assert "orig" in set(frame["variant"])
    assert frame.groupby("role")["variant"].apply(lambda s: s.nunique() > 1).all()
    expected_variants = set(frame["variant"])
    for role, part in frame.groupby("role"):
        assert set(part["variant"]) == expected_variants, f"missing variant in {role}"
        originals = set(part.loc[part["variant"] == "orig", "orig_sample_id"])
        for variant, group in part.groupby("variant"):
            assert set(group["orig_sample_id"]) == originals, f"incomplete {role}/{variant} query coverage"
    if expected_candidates is not None:
        key = ["query_sample_id", "candidate_vessel_id"]
        assert not expected_candidates[key].duplicated().any()
        actual_keys = set(map(tuple, frame[key].to_numpy()))
        expected_keys = set(map(tuple, expected_candidates[key].to_numpy()))
        assert actual_keys == expected_keys, "scored CSV does not cover the original candidate CSV exactly"


def predictions(frame, k, w):
    """Restrict to top K first, standardize within K (population std), retain baseline rank on ties."""
    pool = frame[frame["rank"] <= k].copy()
    grouped = pool.groupby("query_sample_id")["llm_logprob"]
    mu = grouped.transform("mean")
    sd = grouped.transform(lambda s: s.std(ddof=0))
    z = (pool["llm_logprob"] - mu) / sd.clip(lower=1e-8)
    pool["combined_score"] = pool["ranker_score"] + w * z
    top = pool.sort_values(["query_sample_id", "combined_score", "rank"], ascending=[True, False, True], kind="stable").groupby("query_sample_id", sort=False).head(1).copy()
    top["correct"] = top["candidate_vessel_id"].eq(top["target_vessel_id"])
    return top


def metrics(top):
    by_variant = top.groupby("variant")["correct"].agg(["mean", "size"])
    corrupted = by_variant.drop(index="orig")
    return float(corrupted["mean"].mean()), float(by_variant.loc["orig", "mean"]), by_variant


def evaluate(frame, ks=DEFAULT_K, weights=DEFAULT_W, output_path=None, expected_candidates=None):
    validate(frame, expected_candidates)
    tune = frame[frame["role"] == "tune"]
    eval_frame = frame[frame["role"] == "eval"]
    tune_baseline = metrics(predictions(tune, max(ks), 0.0))
    choices = []
    for k in sorted(set(ks)):
        for w in sorted(set(weights)):
            macro, original, _ = metrics(predictions(tune, k, w))
            if original == 1.0:
                choices.append((macro, -w, -k, k, w))
    assert choices, "no tune setting preserves original acc@1=1.000; check baseline"
    _, _, _, k, w = max(choices)
    tune_reranked = metrics(predictions(tune, k, w))
    # No evaluation-role result is used above this line.
    eval_baseline = metrics(predictions(eval_frame, max(ks), 0.0))
    eval_reranked = metrics(predictions(eval_frame, k, w))
    rows = []
    for role, baseline, reranked in [("tune", tune_baseline, tune_reranked), ("eval", eval_baseline, eval_reranked)]:
        for variant, item in baseline[2].iterrows():
            after = reranked[2].loc[variant]
            rows.append(dict(role=role, variant=variant, queries=int(item["size"]), baseline_acc1=float(item["mean"]),
                             rerank_acc1=float(after["mean"]), gain_pp=100 * float(after["mean"] - item["mean"])))
        rows.append(dict(role=role, variant="CORRUPTED_MACRO", queries=int(baseline[2].drop(index="orig")["size"].sum()),
                         baseline_acc1=baseline[0], rerank_acc1=reranked[0], gain_pp=100 * (reranked[0] - baseline[0])))
    table = pd.DataFrame(rows)
    gain = eval_reranked[0] - eval_baseline[0]
    gate = gain >= 0.02 - 1e-12 and eval_reranked[1] == 1.0 and eval_baseline[1] == 1.0
    model_id = frame["llm_model"].iloc[0]
    action = "ACCURACY_GATE_PASS: prepare submission and verify full GPU script/time" if gate else (
        "TRY_7B_ONCE" if "3B" in model_id else "STOP_TRACK_B")
    print(f"Model {model_id}; tune-selected K={k}, w={w:g} (top-K then within-K z-score; ties use rank)")
    print(table.to_string(index=False, float_format=lambda n: f"{n:.4f}"))
    print(f"Eval corrupted macro gain={100 * gain:+.3f}pp; orig={eval_reranked[1]:.4f}; {action}")
    print("Validation uses the existing single-ranker raw score. Before a 36-based submission, align its 12-model score scale (mean) and train-target candidate policy, then recheck validation.")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.assign(selected_K=k, selected_w=w, llm_model=model_id, decision=action).to_csv(output_path, index=False)
        print(f"Saved {output_path}")
    return {"K": k, "w": w, "accuracy_gate_pass": gate, "decision": action, "table": table}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scored_csv", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--candidates-csv", type=Path,
                        default=Path(__file__).resolve().parents[1] / "experiments" / "llm_rerank" / "candidates_top20.csv")
    parser.add_argument("--k", default=",".join(map(str, DEFAULT_K)))
    parser.add_argument("--w", default=",".join(map(str, DEFAULT_W)))
    args = parser.parse_args()
    frame = pd.read_csv(args.scored_csv, encoding="utf-8-sig", dtype={"query_sample_id": str, "candidate_vessel_id": str, "target_vessel_id": str})
    expected = pd.read_csv(args.candidates_csv, encoding="utf-8-sig", dtype={"query_sample_id": str, "candidate_vessel_id": str})
    evaluate(frame, tuple(map(int, args.k.split(","))), tuple(map(float, args.w.split(","))), args.out, expected)


if __name__ == "__main__":
    main()
