# Forbidden competitors in relative features — static audit

Conclusion: **yes, forbidden train vessels can affect the relative feature context in both production112 and v20 before final exclusion.** This is distinct from member softmax normalization. Its presence is not proof of an error or of a beneficial correction: the frozen rankers were trained on the original full-candidate feature definitions.

## Production112

In frozen `astra/arena_20261101/baseline/model/en/model/opus_lib.py:409`, `idx=list(cand.keys())` contains the entire retrieved candidate set. No exclusion mask enters `query_features`.

- `name_best = chosen_similarity - max(chosen_similarity)` and `name_margin = maximum - second` use every entry in `idx` (`492–493`). In this frozen version the Korean chosen similarity is `ck_acr_ratio`; the later mutable shared source has different overrides and must not be substituted silently.
- `ko_comb_best` and `ko_comb_margin` use the Korean combined score over every candidate (`499–500`).
- `kc_features` similarly subtracts the best OTHER candidate's KoCat similarity. The implementation receives full `idx`; forbidden competitors are included.
- `rank_name` and `rank_gt` come from retrieval channel ranks. These are not simply rankings within the final candidate group; filtering and compressing them would be an additional intervention.
- `gt_exact_n`, name frequency and other catalog-wide features are broader catalog statistics, not the same relative-feature mechanism.

The production112 script constructs all these features, predicts model scores, and only then excludes train targets when calculating the 110 gate/choosing the final answer. A forbidden candidate can therefore alter allowed candidates' feature vectors even when it cannot be the final answer.

## v20

The frozen `model/ko/model/src/features.py` likewise builds from the full `signals['cand']` before script-level exclusion. Direct group-relative features include:

- `key_rank_in_group`, `key_lev_gap_to_best`;
- `llm_ratio_rank_in_group`, `llm_ratio_gap_to_best` when beams exist;
- `sim_word_gap_to_best`, `sim_compact_gap_to_best`;
- `dl_rank_in_group`, `group_size`.

`rank_word`, `rank_compact`, `rank_key`, `rank_key_word`, `rank_llm`, `rank_gt`, `rank_tok` originate in retrieval signals; `tok_pool` is a retrieval pool count. Their meanings must not be conflated with ranks recomputed over the final allowed subset. `name_dup_count` is a catalog-wide count and should not change in a narrow candidate-context ablation.

Stable ordinal ranks in this source break equal-score ties by the existing candidate order. Recalculation must preserve that original order; switching to dense ranks would be another change.

## Training mismatch and previous experiments

Frozen training caches contain full-candidate relative features. Recomputing them over allowed candidates only at inference changes their scale, ordinal distribution, margins and group sizes. The rankers are trees: even a common offset subtracted from all candidates can move different candidates across thresholds and alter rank. A query-constant margin can also change which interaction branches the trees use. This is not the harmless common-logit constant that cancels from softmax.

An inference-only ablation could diagnose sensitivity, but it should not be described as restoring the model's original feature semantics. A matched training experiment would rebuild features after removing other fit vessels while retaining the positive target. It is a separate training/data-policy change, outside the frozen current A/B experiments.

The nearest prior is `external/ksh/tools/probe_exclusion_training.py`. It keeps the positive and removes other training-vessel rows **by slicing precomputed `xb[keep]`**, without recomputing relative features. Its logged results were small gains (LightGBM about .0024, GPU CatBoost about .0006) and it was deferred. This is not the proposed context-only recomputation.

`external/ksh/docs/context_density_20261029.md` instead estimates query/candidate metadata density ratios. It does not change candidate-relative name features. Earlier reverse-ranking work explicitly preserved full-pool raw78 features because rebuilding them on top20 would change their meaning (`portfolio_log.md:1083`). No identical allowed-relative recomputation experiment was found in the inspected sources.

## Existing production112 trace exposure counts

`FORBIDDEN_CONTEXT_STATS.json` and `FORBIDDEN_CONTEXT_ROWS.json` summarize only the already captured score trace. No model was run or changed.

| Trace group | Ranked rows | Any forbidden candidate | Unmasked score winner forbidden | Mean forbidden candidates |
|---|---:|---:|---:|---:|
| All ranked rows | 384 | 384 | 11 | 69.875 |
| Original110 gap <4 | 198 | 198 | 10 | 68.263 |
| Protected gap >=4 | 186 | 186 | 1 | 71.591 |

The exclusion set contains 5,851 train vessels. These counts measure exposure and final-score competition, **not which relative feature maxima are forbidden, not feature-caused errors, and not a bound on possible rescues**. The trace does not contain the feature matrices needed for that causal count. The 400-row benchmark is table-seen synthetic and does not estimate Public frequency.

## Safe boundary for any later diagnostic

Keep original candidate IDs/order and raw pair features unchanged; calculate alternate relative columns from an allowed mask into a copied matrix; compare the exact columns and predicted scores. Do not change retrieval ranks, catalog counts, 110 routing gate or current protocols by implication. Report this as a third, separate sensitivity diagnosis, not fold it into pre-member-softmax masking. No such intervention was executed in this audit.
