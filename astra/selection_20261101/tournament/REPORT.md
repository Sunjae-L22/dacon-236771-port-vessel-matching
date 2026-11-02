# Fixed all-pairs Borda experiment

Decision: the predefined OOF and 400-row development screens pass, but **there is no evidence that Borda resolves the anchor problem better than the original pair policy**. Its actual C_vote development gains are entirely returns to 112 top1. No model was trained, no threshold was tuned, no submission was made, and the 15-vessel confirmation set remains unread.

## Frozen selection policy

Use the existing `no_score_all19` classifier on 186 columns: candidate A's 62 features, candidate B's 62 features, and their difference. No ranker score difference is supplied. Evaluate all ordered pairs among the original 112 top20, form `sym(A,B)=.5*(p(A,B)+1-p(B,A))`, and average each candidate's symmetric win probability against every OTHER candidate. Pick the largest mean; ties preserve original candidate order. Change the incumbent only if this winner beats it with symmetric probability at least .95.

The Borda ranking itself does not use incumbent identity. The .95 replacement gate intentionally does. English, original110 gap>=4, and an actual C_vote incumbent outside 112 top20 are preserved.

`PROTOCOL.json` was saved before OOF evaluation; `PRODUCTION_PROTOCOL.json` was saved before the subsequent authorized 400-row application. Both include source/model hashes. The production model hash matches `pairwise/FOLLOWUP_LOCK.json` and has exactly 186 features and 250 trees.

## OOF result

| Fit model → evaluation fold | Clean rescue/harm | Noise rescue/harm | Noise net |
|---|---:|---:|---:|
| 2 → 4 | 0 / 0 | 19 / 3 | +16 |
| 4 → 2 | 8 / 0 | 31 / 10 | +21 |

Both pass the predefined clean-harm-zero/noise-positive screen. A total 448,780 ordered pairs were scored. Protected changes were zero; antisymmetry error was at most 1.11e−16. Candidate permutation checks passed.

However, fold2's eight clean rescues belong to one vessel. On its noisy subset, five vessels have positive net change and six have negative net change. Repeated variants are not independent evidence. These are cross-fold pair heads on single-model OOF112 features, not strict nested CV or actual C_vote.

Compared with the **existing no_score_all19 incumbent-versus-challenger policy**, Borda changes four fold4 predictions with zero additional rescues/harms, and five fold2 predictions with zero additional rescues and one additional harm. Thus the first screen does not establish an incremental benefit from Borda.

## Actual C_vote 400-row development result

All prediction rows were saved before the truth file was read; row-reversal predictions matched exactly. The current actual C_vote predictions were reconstructed using the fixed 112/HSW/v20 routing rule.

| Group | Rows changed | Rescue | Harm | Net |
|---|---:|---:|---:|---:|
| Clean | 1 | 0 | 0 | 0 |
| Noise | 12 | 5 | 3 | +2 |
| All | 13 | 5 | 3 | +2 |
| Changed answer returns to 112 top1 | 11 | 5 | 3 | +2 |
| Changed answer is another candidate | 2 | 0 | 0 | 0 |

C_vote correctness changes from 351/400 to 353/400. All five rescues and all three harms are 112-top1 returns. Both other selected answers remain wrong. This is a **table-seen synthetic development diagnostic**, not Public, independent confirmation, or an estimate of a .98 score.

The final predictions on all 400 rows are **identical to the existing no_score_all19 one-to-one policy**: zero changed answers, zero added rescues, zero added harms. `PRODUCTION_POLICY_COMPARISON.json` records this direct comparison. This diagnostic therefore provides no evidence that the more expensive Borda aggregation fixes anchor dependence.

The classifier was trained on truth-versus-negative pairs. Borda additionally evaluates negative-versus-negative pairs, which are outside the supervised pair construction. Symmetrization enforces complementary probabilities but does not establish calibration, transitivity, or correctness on those extra comparisons.

Files: `RESULTS.json` and direction-specific row files contain OOF evidence; `POLICY_COMPARISON.json` compares the prior pair policy; `production_CVOTE_RESULTS.json` and `production_CVOTE_ROWS.jsonl` contain the development diagnostic; `production_predictions_before_truth.jsonl` records the label-free frozen predictions. No ZIP was built.
