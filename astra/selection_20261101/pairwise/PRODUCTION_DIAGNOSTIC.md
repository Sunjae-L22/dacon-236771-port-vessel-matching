# Actual C_vote400: small net gain, different mechanism from OOF validation

The fixed combined-OOF verifier changed 13/400 rows. Actual C_vote baseline accuracy on this diagnostic was 351/400; the overlay produced355/400. Noise rescue7/harm3 (net+4), clean rescue0/harm0 (one wrong-to-wrong change). This is a reused, table-seen and style-biased synthetic diagnostic with oracle-clean metadata, not a Public score estimate.

**All13 changes replaced a non112 C_vote answer with112's existing top1.** None of the146 eligible rows where C_vote already selected112 top1 changed. Consequently the positive OOF result—improving selection over112's top1—did not recur in this production diagnostic. Its observed benefit instead comes entirely from reverting some majority/v20 selections to112. All three harms undo correct C_vote decisions in favor of wrong112 answers.

This difference matters: the OOF screening evaluated112 top1 as incumbent; the requested production application permits any C_vote answer inside112 top20 as incumbent. That application was fixed before this diagnostic, but is a distinct use of the model. The model has no features describing why v20 selected its incumbent; it sees only112 candidate features and scores. The0.95 gate is not calibrated correctness: one harmed case receives a score above0.999.

| Metric | Result |
|---|---:|
| Original110-hard rows |198|
| Eligible C_vote answers inside112 top20 |179|
| Outside-pool protected rows |19|
| Eligible incumbents different from112 top1 |33|
| Noise rescue / harm, distinct rescued/harmed vessels |7 /3, 5 /3|
| Changed rows restoring112 top1 |13/13|
| Changed rows replacing112 top1 |0|

All production features were regenerated with the untouched original112 package using pandas2.2.3 default `read_csv`. Over198 hard rows, raw input fingerprints, all candidate IDs/order, allowed top20/top1, original110 gate and six-KC-model score sums exactly reproduced the original traces (all differences0). The pair score feature uses six-model sum divided by6. Actual v20 predictions came from `astra/selection_20261101/v20_hard_pred.csv`; HSW/112 from the original400-row component execution. The original C_vote majority rule was applied before the verifier.

The verifier retained original sample order, preserved protected rows, and produced identical per-row records after reversing the input order. Predictions were saved in `production_predictions_before_truth.jsonl` before loading the truth file. Results and complete per-row comparisons are in `production_CVOTE_RESULTS.json` and `production_CVOTE_ROWS.jsonl`.

The separately prepared90-row15-vessel train-token-composition confirmation diagnostic has47 hard rows and also reproduces the original production112 traces exactly. It uses the same frozen model/threshold and awaits actual v20 outputs. No parameter or gate changes were made after observing these results. No submission ZIP was built or submitted.
