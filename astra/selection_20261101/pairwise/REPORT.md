# Fixed hard-pair verifier: OOF passed; production400 gain is a112 rollback effect

**Latest decision:** the preregistered2×2 follow-up primary (no KC-score difference + hardest negative only) failed clean-harm0: fit4→2 harmed two clean rows from one vessel. No primary fullfit or production/final-confirmation evaluation followed. Other cells were not promoted by best-score selection. See `ablation/REPORT.md`. The90-row15-vessel confirmation remains unconsumed by a pair-verifier prediction experiment.

The preregistered 0.95 threshold passed both directional screens. Following that screen, the same recipe was fitted to the combined two OOF folds (124 vessels, 5.42 seconds; `FULLFIT.json`). Actual C_vote400 comparison gives noise+7/−3, clean0/0, but every one of its13 changes merely restores112 top1 where C_vote chose another answer. None improves on112 top1. See `PRODUCTION_DIAGNOSTIC.md` for this material distinction and exact production reproduction QA. The separate90-row confirmation awaits v20. These results do not establish Public improvement or0.98. No submission or production ZIP was created.

| Fit → confirmation | Noise rescue / harm | Noise changed | Rescued vessels | Clean changed / harm | All primary rows |
|---|---:|---:|---:|---:|---:|
| 2 → 4 | +2 / −0 | 2 | 2 | 0 / 0 | 2,030 |
| 4 → 2 | +7 / −0 | 9 | 4 | 0 / 0 | 2,150 |

The reverse direction includes two wrong-to-wrong changes. Several rescue rows repeat a vessel/name with different source records: +7 is four rescued vessels, not seven independent ships. Both confirmatory folds were used in earlier experiments; neither is a fresh sealed holdout.

## Frozen recipe

Primary Korean rows with original110 gap < 4 only. Fixed existing top20 allowed OOF112 candidates, no target injection or catalog-exclusion changes. Training pairs compare the truth against every other existing top20 candidate, with reversed labels/orientations added. Inputs are A raw62, B raw62, A−B raw62, and single-KC-model score difference, 187 numeric features without ID/name inputs. Equal total weight per vessel, rescaled to mean weight one. LightGBM binary: 250 trees, leaves15, depth4, minimum leaf30, learning rate0.03, seed112, two threads. Both directions use the same recipe without search.

At inference, each challenger is compared against the frozen112 incumbent using `0.5 * (pAB + 1 - pBA)`. The highest-scoring challenger replaces it only at ≥0.95; ties preserve original candidate rank. This value is an uncalibrated model gate, not a 95% correctness guarantee.

Training fold2 used 626 supported groups / 23,788 oriented pairs / 62 vessels. Training fold4 used 516 / 19,608 / 62. Unsupported groups were excluded only from training and retained in evaluation denominators: fold2 31 and fold4 8 hard rows. Each confirmatory vessel set is disjoint from its pair-verifier fit set.

## Reproduction and QA

- Regenerated raw query metadata and candidates against the original stored packet: top100 IDs, indices, order and top1 had zero mismatches in all 657 fold2 and 524 fold4 hard rows. Maximum KC score difference was exactly 0.0 in each fold. Feature names/order match the frozen62-feature model. No heldout target was in that fold's base-fit exclusion set.
- Shared KoCat CSV is byte-identical to the snapshot CSV; hashes are in `INPUT_HASHES.json`.
- Protected rows changed zero times. English and original110 gap ≥ 4 remain untouched by the verifier.
- Pair-score reversal/shuffle over all 12,483 fold2 and 9,956 fold4 challenger pairs produced exactly zero difference; 100 individually scored pairs per direction also matched the corresponding batch scores exactly. This checks pair-verifier row independence; it is not a full production 3,000/600-row timing test.
- Total preparation/training/evaluation time: 269.76 seconds on CPU; no GPU used.

## Remaining limits and next decision

There are 430 / 406 clean rows, but only 55 rows across 13 vessels in fold2 and 14 rows across 5 vessels in fold4 are eligible for a change. Clean harm0 therefore has limited support. The data comprise native heldout training names and fixed synthetic edits/tonnage removal, not the unknown Public distribution.

OOF112 here uses one KC model, while production112 uses six. Original110 gate models also differ: six in fold2, one in fold4. Each base packet excludes its own evaluation vessel fold, but pair-verifier fitting is not strict nested CV: the base used to generate fit-fold features was trained using the opposite confirmation fold. The direct pair-verifier vessel split is correct; this remaining base-model interaction must not be called a fully independent nested test.

The top20 selection oracle contains 99 noise errors in fold2 and 71 in fold4; this verifier fixes only 7 and 2 respectively. Thus it does not establish the scale of gain needed for 0.98. Before any production promotion, compare the changed cases with actual C_vote to determine whether its other voters already repair them, address the single-model versus six-model feature scale, and perform actual deployment QA. No threshold reduction, alternative model search, or submission was performed. Combined OOF fitting keeps the same recipe and does not turn this screening into independent validation.

Machine-readable results: `RESULTS.json`, `fit2_confirm4_rows.jsonl`, `fit4_confirm2_rows.jsonl`, `QA.json`, and per-fold PREPARE/TRAIN reports. `PROTOCOL.json` was written before execution.
