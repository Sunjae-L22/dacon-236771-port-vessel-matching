# author: sol
"""Analyze saved eval top-20 retrieval/ranking errors without any model inference."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "shared/experiments/llm_rerank"
frame = pd.read_csv(OUT / "candidates_top20.csv")
catalog = pd.read_csv(ROOT / "data/open/vessel_catalog.csv", usecols=["vessel_id", "vessel_name"])
names = catalog.set_index("vessel_id").vessel_name.to_dict()
part = frame[(frame.role == "eval") & frame.variant.isin(["trunc60%", "drop_last_word", "drop_first_word"])]
rows = []
for qid, group in part.groupby("query_sample_id", sort=False):
    first = group.sort_values("rank").iloc[0]
    target = group[group.candidate_vessel_id == first.target_vessel_id]
    rank = int(target.iloc[0]["rank"]) if len(target) else None
    bucket = ("outside_top20" if rank is None else "1" if rank == 1 else "2" if rank == 2 else
              "3" if rank == 3 else "4-5" if rank <= 5 else "6-10" if rank <= 10 else "11-20")
    rows.append(dict(query_sample_id=qid, variant=first.variant, vessel_name=first.vessel_name,
                     original_vessel_name=first.original_vessel_name, target_vessel_id=first.target_vessel_id,
                     target_name=names[first.target_vessel_id], target_rank=rank, rank_bucket=bucket,
                     error_type="outside_top20" if rank is None else "correct" if rank == 1 else "ranked_below_top1",
                     predicted_id=first.candidate_vessel_id, predicted_name=first.candidate_vessel_name))
queries = pd.DataFrame(rows)
errors = queries[queries.error_type != "correct"]
counts = pd.crosstab(queries.variant, queries.error_type).reindex(
    columns=["correct", "ranked_below_top1", "outside_top20"], fill_value=0)
counts["queries"] = counts.sum(axis=1)
counts["acc1"] = counts.correct / counts.queries
counts["recall20"] = 1 - counts.outside_top20 / counts.queries
distribution = pd.crosstab(queries.variant, queries.rank_bucket).reindex(
    columns=["1", "2", "3", "4-5", "6-10", "11-20", "outside_top20"], fill_value=0)
counts.to_csv(OUT / "weakness_summary.csv")
distribution.to_csv(OUT / "weakness_target_ranks.csv")
errors.to_csv(OUT / "weakness_errors.csv", index=False)
print(counts.to_string())
print(distribution.to_string())
print("outside_top20는 저장된 top20 밖이며, 전체 검색 후보 밖이라는 뜻은 아님.")
