# Next narrow hypothesis: short final-token prefix evidence

This is a proposed next experiment only. No new model or submission was built for it.

The 84/88/91 Korean path still uses build70's `ko_translit_argon.py`. A short final Korean token can provide useful prefix evidence that full-token edit similarity undervalues. build81 contains two relevant changes, but also unrelated phonetic, vowel, and combined-score changes. Do not transplant build81 wholesale.

## Exact relevant 81 changes

1. `build81/ko_translit_argon.py:889` adds this alternative to `ko_tail_sim` when both query and candidate have at least two tokens:

```python
0.88 * max(
    _lcp_ratio(qt_last, ct_last),
    _lcp_ratio(qr_last, ce_last),
    _lcp_ratio(qr_last, ce_last_r),
) if 1 <= len(qr_last) <= 4 else 0.0
```

The corresponding original block is `build84_astra/ko_translit_argon.py:868–879`; it only considers full similarity, then the vowel/consonant penalty. 81 also changes the vowel class from `aeiou` to `aeiouwy`. Treat that as a separate alteration; preserve the old class in a narrow first ablation.

2. `build81/ko_translit_argon.py:938` adds this fallback after the existing `len(qt_tr) >= 3` prefix branch:

```python
elif 1 <= len(qt_rom) <= 3 and len(ct) >= 2 and (
    ct.startswith(qt_tr) or ce1.startswith(qt_rom) or ce2.startswith(qt_rom)
):
    sc = max(sc, 0.86)
```

The original `build84_astra/ko_translit_argon.py:923` has only the >=3 branch. Guard empty translated strings explicitly if implementing a standalone feature; `startswith('')` must never create evidence.

## Preserve candidate recall and isolate ranking

The recommended first experiment appends two or three features to the fixed candidate pool rather than changing existing KoMatcher scores/retrieval:

- Last-token short-prefix agreement, derived from `qinfo['tok_projections'][-1]` and `ix.ko.cat_tok_e2k[candidate][-1]`, zero unless both names have >=2 tokens and the query roman token has length1–4.
- Average short-token prefix coverage over query tokens, using the short-prefix condition above without replacing existing `ko_token_cov`.
- Optionally the difference between recomputed short-prefix tail similarity and original `ko_tail_sim` (preserving the old vowel class and original similarity terms).

No hand-written vessel mappings or external data are needed. All learned token translations come from the same allowed fold dictionary.

## Reuse coldship caches safely

`astra/coldship91/foldN/queries.json` preserves the exact augmented query rows; `inner_folds.json` identifies which inner partition owns each target. Each `features/NNNNNN.npz` stores original 71-feature `X`, flattened catalog `idx`, `groups`, `query_index`, `target`, `inner_fold`, and labels.

Recreate the three inner training indexes exactly as coldship91: metadata from the other two vessel partitions, LLM dictionary pairs filtered to those partitions, KO_DICT=0. For each saved query, use its saved `inner_fold`, read its saved candidate IDs, calculate only the additional features, and append them to the original rankless70 columns. Load each compressed `X`/idx/groups array once per chunk, not once per query. This avoids expensive full candidate retrieval and ensures no label leakage or candidate-pool change.

For validation, construct only the outer-fit dictionary and evaluate the added features over `neural87_foldN/eval_features_enhanced.npz` candidate IDs. The underlying validation query reconstruction must exactly match the existing9 stress variants plus all native rows. A comparison should use unchanged600-tree parameters/seed84 and report orig/native, each stress variant, and vessel-level uncertainty. Do not tune the constants or gates against already inspected fold4 outcomes and then call them untouched confirmation.

If instead replacing original tail/token scores, recompute all dependent fields: `ko_combined_score` contains `0.11*tail_sim + 0.05*token_cov`; `opus_lib.py:494–506` creates `ko_comb_best`, `ko_comb_margin`, and `ko_comb_rev` from that score. Keeping old derived fields after mutating primitive scores would create another train/serve inconsistency. The additive version avoids this dependency risk.
