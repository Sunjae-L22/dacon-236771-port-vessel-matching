# Independent ranking-policy audit — pre-member exclusion

Decision: the proposal is a genuinely different, testable **ensemble conditioning policy**, with no observed prior implementation in the inspected project files. It is not a proven bug fix, and accuracy improvement does not follow from the algebra.

## Verified original implementation

`astra/arena_20261101/baseline/model/ko/model/src/ensemble.py` normalizes each member over the entire candidate group and then adds its weighted probabilities. `model/ensemble.json` has three CatBoost members weighted 3 each and two LightGBM members weighted 1 each. `script.py:220` masks fixed train-vessel IDs only after that sum, renormalizes, applies KO_W=4, and then lets `encoder_views.py` choose its top20 and determine whether to invoke neural scorers.

The earlier `external/ksh/tools/probe_exclude_train_vessels.py` first calls `blend(...)` and masks the final scores afterward. v14–v18 submission wrappers likewise mask after the ensemble. The published local exclusion gain therefore does not validate member-before-softmax exclusion. No matching earlier experiment was found in the inspected source, tools and docs.

The failed low-dimensional calibration experiment in `external/ksh/docs/calibration_20261030.md` changes base/Qwen/BGE log-odds coefficients. It is a related warning about confidence versus accuracy, but not this five-member mask-order ablation. That report explicitly found improved conditional NLL with worse top1 accuracy.

## Algebra and its limitation

Let `p_m(i)` be member m's full-universe softmax, `A` the allowed candidates, `Z_m = sum_{i in A} p_m(i)`, and `q_m(i)=p_m(i)/Z_m` on A.

Current policy:

`q_old(i) = sum_m [w_m Z_m / sum_l(w_l Z_l)] q_m(i)`.

Proposed policy:

`q_new(i) = sum_m [w_m / sum_l w_l] q_m(i)`.

Thus the current policy implicitly downweights members that assign much probability to forbidden vessels. The new policy restores the configured member weights after conditioning each member. With equal `Z_m`, identical member conditionals, or no exclusions, the policies coincide.

Crucially, the old expression is also the ordinary Bayesian conditioning of a mixture on the event that the answer is allowed. A member highly confident in a forbidden train vessel may have unreliable rankings among remaining vessels. Restoring its full weight can amplify that unreliable residual distribution. Conversely, if forbidden confidence is mainly memorization/irrelevant candidate competition, it can suppress useful residual rankings. Which interpretation holds needs held-out outcomes.

`mask_math.py` independently verifies the identity, 200 randomized candidate-permutation/forbidden-decoy tests, and a concrete rank flip. In the toy example the old allowed probabilities are `[.2692,.7308]`, while the new ones are `[.5749,.4251]`. The first member's effective old weight is only .000381 despite configured weight .5. Either candidate could be the true answer: the example demonstrates mechanism, not improvement.

## Runtime requirements for a faithful ablation

- Keep the exact candidate IDs, order, and feature matrices. Only move exclusion into each member's normalization. Removing candidates before feature construction changes candidate-relative features and is a different experiment.
- Compute each member's softmax directly over allowed raw logits, subtracting the allowed maximum. Dividing already-underflowed full-universe probabilities by an almost-zero allowed mass is numerically unsafe.
- Preserve original behavior if all candidates are excluded. The normal case must retain at least one allowed candidate. A single allowed candidate should be a point mass.
- Check hook cardinalities explicitly: number of blocks, current candidate lists, and current names must agree. `zip` alone silently truncates. No previous-row candidate mask may be reused.
- Rerun the unchanged KO prior and neural policy. Pre-masking can change both the top20 set and the confidence gate. Old neural scores are reusable only for identical query/candidate inputs; a changed top20 or newly activated gate requires actual scoring. A base-only rank flip is not the final v20 result.
- Apply the original C_vote router afterward. A v20 change that does not change the routed vote has no C_vote benefit. Keep English and original110 gap>=4 unchanged.

## What the trace should report

For every row: member raw logits and IDs, fixed exclusion mask, `Z_m`, effective old weights, old/new pre-prior winner, post-prior winner, top20, neural activation, final v20 winner, and final C_vote winner. Record target-in-candidate status before separating retrieval from ranking mistakes. Count C_vote rescue/harm, including whether original112 repairs are harmed.

The 198-hard-row capture comes from the protocol's **400-row table-seen synthetic diagnostic**. The real deployed C_vote executable makes the runtime comparison faithful, but does not make these synthetic queries representative of Public. Its gains are not an unbiased .98 estimate. A correct math test, changed predictions, or improved probability loss alone does not establish ranking accuracy.

## Additional structure idea

The most useful additional check is a **forbidden-decoy invariance diagnostic**, not another deployable heuristic: hold member raw scores on allowed candidates fixed while changing only forbidden logits. The proposed aggregation is invariant; current aggregation can flip winners. Stratify observed flips by whether the affected member's allowed conditional ranking is right or wrong on held-out data. This directly discriminates useful conditional evidence from unreliable residual noise without top2 replacement or name-first rules.

No second deployment method is recommended from the audited evidence. Generic temperature/weight calibration and token priors were already tested and rejected elsewhere; simply renaming them would repeat old work. This audit creates only a mathematical test and report, with no Opus edits, GPU use, or submissions.
