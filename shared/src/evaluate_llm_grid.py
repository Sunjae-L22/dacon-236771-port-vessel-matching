# author: sol
"""Keep every K/w result; select on tune before examining any eval result."""
import argparse
from pathlib import Path

import pandas as pd

from evaluate_llm_rerank import DEFAULT_K, DEFAULT_W, metrics, predictions, validate


def summarize(top):
    damaged, original, variants = metrics(top)
    orig = top[top.variant == "orig"]
    return dict(original_acc=original, original_correct=int(orig.correct.sum()),
                original_rows=len(orig), corrupted_macro_acc=damaged,
                mixed_score=0.5 * original + 0.5 * damaged,
                **{f"{v}_acc": float(variants.loc[v, "mean"])
                   for v in ["trunc60%", "drop_last_word", "drop_first_word", "rename"]})


def evaluate_grid(frame, grid_path, output_path, expected_candidates=None,
                  ks=DEFAULT_K, weights=DEFAULT_W, selection="mixed"):
    validate(frame, expected_candidates)
    assert selection in ("mixed", "strict")
    tune = frame[frame.role == "tune"]
    baseline_tune = summarize(predictions(tune, max(ks), 0))
    assert baseline_tune["original_acc"] == 1.0
    rows = []
    for k in sorted(set(ks)):
        for w in sorted(set(weights)):
            item = summarize(predictions(tune, k, w))
            loss = baseline_tune["original_correct"] - item["original_correct"]
            rows.append(dict(role="tune", K=k, w=w, **item, original_loss_rows=loss,
                             eligible_strict=loss <= 0, eligible_one_row=loss <= 1))
    eligible = [r for r in rows if r["eligible_one_row" if selection == "mixed" else "eligible_strict"]]
    field = "mixed_score" if selection == "mixed" else "corrupted_macro_acc"
    chosen = max(eligible, key=lambda r: (r[field], -r["w"], -r["K"]))
    mixed_best = max([r for r in rows if r["eligible_one_row"]],
                     key=lambda r: (r["mixed_score"], -r["w"], -r["K"]))
    k, w = chosen["K"], chosen["w"]
    # The chosen setting is frozen here. The entire eval grid is reporting only.
    eval_frame = frame[frame.role == "eval"]
    baseline_eval = summarize(predictions(eval_frame, max(ks), 0))
    for kk in sorted(set(ks)):
        for ww in sorted(set(weights)):
            item = summarize(predictions(eval_frame, kk, ww))
            loss = baseline_eval["original_correct"] - item["original_correct"]
            rows.append(dict(role="eval", K=kk, w=ww, **item, original_loss_rows=loss,
                             eligible_strict=loss <= 0, eligible_one_row=loss <= 1))
    grid = pd.DataFrame(rows)
    grid["selected_from_tune"] = grid.K.eq(k) & grid.w.eq(w)
    grid["mixed_best_on_tune"] = grid.K.eq(mixed_best["K"]) & grid.w.eq(mixed_best["w"])
    for role, baseline in [("tune", baseline_tune), ("eval", baseline_eval)]:
        mask = grid.role.eq(role)
        for col in ["original_acc", "corrupted_macro_acc", "mixed_score",
                    "trunc60%_acc", "drop_last_word_acc", "drop_first_word_acc", "rename_acc"]:
            grid.loc[mask, col + "_gain_pp"] = 100 * (grid.loc[mask, col] - baseline[col])
    grid_path, output_path = Path(grid_path), Path(output_path)
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(grid_path, index=False)

    selected = []
    for role, part in [("tune", tune), ("eval", eval_frame)]:
        before = predictions(part, max(ks), 0).groupby("variant").correct.agg(["mean", "size"])
        after = predictions(part, k, w).groupby("variant").correct.agg(["mean", "size"])
        for variant, item in before.iterrows():
            selected.append(dict(role=role, variant=variant, queries=int(item["size"]),
                                 baseline_acc1=item["mean"], rerank_acc1=after.loc[variant, "mean"],
                                 gain_pp=100 * (after.loc[variant, "mean"] - item["mean"]),
                                 selected_K=k, selected_w=w, selection=selection))
    table = pd.DataFrame(selected)
    table.to_csv(output_path, index=False)
    ev = grid[(grid.role == "eval") & grid.selected_from_tune].iloc[0]
    print(f"Tune selection={selection}: K={k}, w={w:g}; mixed=.5*orig+.5*damaged; tune original loss <=1 row for mixed", flush=True)
    print(grid[["role", "K", "w", "original_acc", "original_loss_rows", "corrupted_macro_acc",
                "mixed_score", "mixed_score_gain_pp", "eligible_strict", "eligible_one_row",
                "selected_from_tune"]].to_string(index=False), flush=True)
    print(f"판정: eval 가중평균 {ev.mixed_score_gain_pp:+.3f}%p; "
          f"trunc60% {ev['trunc60%_acc_gain_pp']:+.3f}%p; "
          f"drop_last {ev.drop_last_word_acc_gain_pp:+.3f}%p; "
          f"원본 손실 {int(ev.original_loss_rows)}/{int(ev.original_rows)}행. "
          "eval 전 구간은 보고용이며 선택에 사용하지 않았음.", flush=True)
    print(f"Saved grid: {grid_path}\nSaved selected variants: {output_path}", flush=True)
    return dict(K=k, w=w, grid=grid, table=table, eval_selected=ev)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scored_csv", type=Path)
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--selection", choices=["mixed", "strict"], default="mixed")
    p.add_argument("--candidates", type=Path,
                   default=Path(__file__).resolve().parents[1] / "experiments/llm_rerank/candidates_top20.csv")
    a = p.parse_args()
    dtype = {c: str for c in ["query_sample_id", "orig_sample_id", "candidate_vessel_id", "target_vessel_id"]}
    evaluate_grid(pd.read_csv(a.scored_csv, dtype=dtype), a.grid, a.out,
                  pd.read_csv(a.candidates, dtype=dtype), selection=a.selection)


if __name__ == "__main__":
    main()
