# Astra audit, 2026-10-29

This audit uses only train.csv, vessel_catalog.csv and previously saved train-derived validation artifacts. It does not read hidden test data.

## Confirmed deployment defect beyond pyarrow

88 trained with `KO_DICT=0`, but imported the 79 `opus_lib.py` whose baked default was 1 before setting the environment variable to 0. That environment variable is read only at import. `KoMatcher` subsequently copies `_UNIVERSAL_KO_SUBWORDS` at construction without reading the environment. Therefore 88's Korean matcher actually used the universal dictionary ON during serving.

A full training-versus-ZIP source comparison found `astra_lib.py` and `ko_translit_argon.py` identical. The only `opus_lib.py` difference was the import-time default 0 → 1. See `train_serving_source_diff.json`.

90's builder forces the legacy import ON, constructs the original English matcher, then temporarily clears the universal dictionary while constructing the new Korean matcher, restoring the global afterward. The matcher copies the globals only during initialization; later methods use instance dictionaries. This preserves the existing English instance. The independent packaging agent verified exact candidate indices and all 71 feature values on 90 queries/30,602 candidates against the real pinned training code, including NaNs.

89 is a pure pyarrow portability control and retains the semantic mismatch. 90 is the corrected training-contract candidate. Neither has a measured Public score at audit time.

## What the existing validation establishes

The 5-fold improvement from 80.48% to 82.61% is a paired comparison of newly trained Korean-only single-seed models, not an evaluation against the 79 production ensemble. It also uses dictionary OFF, 2,500 sampled LLM rows and a different augmentation recipe from 79. All 5 folds have now informed decisions; they must not be described as permanently untouched holdouts for subsequent experiments.

The 9 conditions are artificial stress tests. 98/382 Korean vessels have one-word original names, so several delete/reorder conditions repeat the original name. The synthetic `rename` condition replaces the name with a random real training Korean name, whereas the training rename recipe mixes prefix replacement, mixed names, and complete replacement. Equal weighting of these conditions does not estimate the unknown Public corruption mix.

## Remaining bottleneck: ranking more than retrieval

| Condition | Enhanced candidate recall | Enhanced accuracy |
|---|---:|---:|
| drop last word | 99.48% | 79.06% |
| truncate | 99.48% | 80.89% |
| truncate + tonnage noise | 97.64% | 60.47% |
| rename | 89.27% | 44.76% |

The old weakness table contains 241 failures: 203 inside top 20 and only 38 outside top 20. In 125 failures tonnage is tied, 67 have the wrong candidate closer, and 49 have the target closer. Widening retrieval alone is unlikely to close the gap.

The oracle that chooses the correct answer among the baseline, enhanced, and rankless rankers reaches 86.94%, compared with 82.61% for rankless. This shows complementary mistakes, not a usable model or forecast. A learned confidence blend must use cross-fitted predictions; arbitrary voting can easily regress.

## Metadata limits and actionable focus

Among the 382 Korean validation vessels, 188 have at least one other eligible catalog vessel with exactly the same catalog GT, flag, and type (up to 13 indistinguishable ships). This is an equivalence-class diagnostic even with oracle catalog labels, not a numerical upper bound on Public accuracy. The query contains neither DWT nor size, so those catalog-only values cannot provide a direct match measurement. Pure name-free rename is intrinsically ambiguous for these ships.

In 5-fold Korean validation, 40/382 ships have missing query-country labels. After treating missing values consistently as the production code does, the correct country mapping is unseen for 7/382 ships and the type mapping for only 1/382. These include genuine mismatches between recorded and catalog flags. Sparse rare-type anecdotes therefore do not explain most Korean failures in this sample. Missing-country uncertainty and metadata changes make hard filtering unjustified.

## Strong new experiment: cold-ship training features

The ranker currently learns on the same ships used to build its translation dictionary and metadata probabilities. A direct diagnostic on 382 native Korean vessels compared a full-fit matcher with the actual outer-fold matcher that never saw each target:

- Translation changes for 188/382 vessels.
- Mean true-target `ko_trans_sim`: 0.98818 in-fit versus 0.91130 out-of-fit.
- Perfect translation similarity: 352/382 in-fit versus 189/382 out-of-fit.
- Mean `ko_combined_score`: 0.97258 versus 0.93188.

This is training-feature optimism, not outer-validation leakage. The existing outer validation is vessel-disjoint. It shows why the ranker may learn thresholds inappropriate for unseen ships.

`astra/coldship91.py` tests a 3-way inner vessel split for the ranker's training examples: each query uses a matcher and metadata statistics fitted only to the other two partitions. Candidate pools remain unfiltered and all training examples/70 features/600 trees/seed 84 are held equal to rankless88. Its exact augmented-query/code hash matches the existing fold0 cache (`c26b68dae72c025c`). The pilot uses outer fold0 and requires at least +0.5 percentage point stress accuracy with no native-row or orig loss before proceeding to a fixed fold4 method confirmation (fold4 was used by prior experiments and is not a pristine unused holdout). Its outputs are in `astra/coldship91/`, separate from this audit.

Translated qcmp/token features are being tested separately by the lead agent. They should be additive, keeping raw transliteration features, and trained under the same dictionary contract.

## Cold-ship pilot result

Outer fold0 completed in 1,200.8 seconds: stress accuracy 82.2511% → 83.5498% (+1.2987 percentage points), native rows 99.6% unchanged, orig 98.7013% unchanged. Vessel bootstrap 95% interval for the delta is [-0.433, +3.175] percentage points. Rename and word deletion improved, while truncation plus tonnage noise regressed. It passes the prespecified point-estimate gate and proceeds to fold4 method confirmation; this is not enough evidence for a submission model yet. Training groups missing their positive increased from 278/8,952 to 341/8,952.
