# Preregistered2×2 ablation: primary rejected

The predeclared primary was **no direct KC-score difference + hardest wrong candidate only**. It fails the clean-harm0 requirement in fit4→confirmation2: two harmed clean rows from one vessel. It was not full-fitted and was not run on production400 or the90-row15-vessel final confirmation. The other three cells are diagnostic controls, not fallback candidates selected by their scores.

All models use the same250-tree LightGBM parameters, fixed0.95 gate, reversed-pair augmentation and equal total weight per vessel. All original OOF scopes, top20 candidates, target-absent training exclusions and evaluation denominators remain unchanged. There was no threshold search.

| Cell | Direction | Clean rescue/harm | Noise rescue/harm | Total changed |
|---|---|---:|---:|---:|
| Direct score + all negatives (existing) |2→4|0/0|2/0|2|
| Direct score + all negatives (existing) |4→2|0/0|7/0|9|
| Direct score + hardest1 |2→4|0/0|9/5|19|
| Direct score + hardest1 |4→2|0/0|20/7|47|
| No direct score + all negatives |2→4|0/0|19/3|29|
| No direct score + all negatives |4→2|8/0|32/10|69|
| **No direct score + hardest1: primary** |2→4|0/0|21/14|44|
| **No direct score + hardest1: primary** |4→2|8/2|29/20|82|

Removing the explicit KC-score difference increases departures from112's ranking even with all negatives. Hardest-only training also increases changes, including harms. Combining both factors gives the largest change rate and breaks the primary clean-harm constraint. Thus the original verifier's conservative behavior is affected by both direct score information and negative composition. This does not identify a safe independent new signal, and row counts are not independent vessel counts.

The primary's noisy rescue/harm vessels are10/7 in fold4 and6/10 in fold2. Its eight clean rescued rows are a single vessel; its two clean harmed rows are another single vessel. Reused development folds and nonnested base-model interactions remain the same as in the original screening.

The whole ablation took7.06seconds. `ABLATION_PROTOCOL.json` in the parent directory was saved before execution and names the primary in advance. `RESULTS.json` contains all cells; per-cell model, training receipts and row-level results remain available for audit. No best-cell promotion, production experiment, final-confirmation consumption, ZIP build or submission followed this primary failure.
