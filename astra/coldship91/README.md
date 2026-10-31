# Cold-ship training pilot — completed, no adoption

`coldship91` is an experiment name. It is **not** the model in `submissions/91_submit.zip`; that ZIP is the separately verified deployment repair.

Only train.csv, vessel_catalog.csv, and permitted train-derived LLM pairs were used. No hidden test data or GPU was used. All pilot processes finished.

## Recipe

The 88 ranker learns candidate features from a dictionary and metadata statistics that include the same training ship. This can make training examples easier than genuinely unseen ships. This pilot keeps the exact augmented queries,70 rankless features,600 trees,seed84 and candidate-selection code, but divides the outer training ships into3 inner partitions(seed91). Every training query obtains its translation dictionary and flag/type/unit statistics from the other2 partitions. Candidate pools remain unfiltered. The target can legitimately be absent; no positive candidate was injected.

Outer validation uses exactly the saved88 candidate/feature caches, including all native Korean rows. Its dictionary/statistics fit only the outer training vessels. The fold0 and fold4 augmented-query/code hashes match their original88 training caches: `c26b68dae72c025c`, `28f7e4e3cbc113a4`.

## Results

| Validation | 88 | Cold-ship | Delta |
|---|---:|---:|---:|
| fold0,77 vessels×9 conditions |82.2511%|83.5498%|+1.2987pp|
| fold4,76 vessels×9 conditions |79.0936%|79.6784%|+0.5848pp|
| pooled,153 vessels×9 conditions |80.6826%|81.6267%|+0.9441pp|

Pooled vessel-bootstrap95% interval: **−0.4357 to +2.3239pp**. Native-row accuracy stayed99.7917%; orig stayed99.3464%. Across synthetic conditions,37 mistakes were corrected and24 correct predictions regressed. Truncation plus tonnage-noise accuracy was unchanged in the pooled sample (fold0−2,fold4+2 cases).

Fold0 passed the predefined pilot gate(+0.5pp with native/orig no loss), which triggered the same-recipe fold4 check. Before reading fold4 results, `progression_gate.json` fixed a stricter progression requirement: pooled gain≥1pp, eachfold nonnegative, vesselCI lower>0, native/orig no loss, pooled trunc_noise no loss. It **failed gain≥1pp and CI lower>0**. No additionalfolds, full production training, or new submission was started.

These two folds are not proof of Public improvement. Fold0 selected the method; fold4 confirmed this recipe but was already used in earlier experiments. Bootstrap intervals are descriptive after pilot selection.

## Fixed50:50 score blend

One no-search blend was fixed before the fold4 model/result existed: `.5*score88 + .5*scorecoldship`, on identical candidates. It lost on both folds: fold0−0.2886pp, fold4−0.1462pp; pooled80.4648%(−0.2179pp). It is rejected. No otherweight was tried.

## Training-candidate diagnostic

| Fold | Queries | Missing positive with88 features | Missing positive with cold features |
|---|---:|---:|---:|
|0|8,952|278|341|
|4|9,036|299|355|

LambdaRank receives no useful ordering signal from an all-zero group. Injecting the known positive only during training could be a future separate test; it was **not** used here. A leave-one-vessel-out statistics scheme could also reduce the dictionary-size change introduced by3-way partitioning, but no such test was run.

## Artifacts

- `two_fold_report.json`: complete metrics, intervals, gate checks and limits.
- `fold0/` and `fold4/`: models, original fixed plans, predictions, paired predictions, manifests and featurechunks with catalog candidate IDs for reuse.
- `blend50_plan.json`: fixed no-search blending rule.
- `progression_gate.json`: criteria recorded before inspecting fold4 results.
- `../audit90/short_prefix_next.md`: separately proposed short-prefix feature test; not executed.

The feature cache includes original71-feature arrays and a separate70-feature trainingmatrix, so future narrow feature additions can reuse candidate IDs without rerunning full retrieval. Keep these files isolated from existing submissions and old pinned code.
