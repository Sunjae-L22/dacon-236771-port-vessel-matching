# Post-ablation-selected follow-up: direct KC-score removal only

This is explicitly an exploratory follow-up selected after observing the ablation. The original preregistered primary, no-score+hardest1, remains rejected. The selected follow-up removes only the last direct KC-score difference feature from the original all-negative verifier (187→186); all other parameters and the0.95 gate remain fixed. Its reused-fold results are not independent confirmation.

The combined OOF2/4 model is fitted and locked in `FOLLOWUP_LOCK.json`. A diagnostic over 3,762 pairs changes the removed score input to an extreme unrelated value and produces exactly zero prediction difference, verifying that this field is absent from inference. The original 62 similarity/rank features still remain; removing the direct model-score column does not create new independent evidence.

## Existing400 rows: development only

Actual C_vote351/400 becomes353/400. Noise rescue5/harm3, clean rescue0/harm0. Thirteen rows change, including one clean wrong-to-wrong change. Scope protection and row reversal checks pass with zero mismatches.

| Change mechanism | Changed | Rescue | Harm | Net |
|---|---:|---:|---:|---:|
| Restore112's existing top1 |11|5|3|+2|
| Select another112 candidate |2|0|0|0|

The two non-restoration changes are `ASTRAHG_002_swap_nogt` (original112 rank3→rank2) and `ASTRAHG_049_double_nogt` (rank1→rank3); both remain wrong. Therefore the follow-up still has no demonstrated useful production rescue beyond returning to112's existing answer. Complete incumbent/challenger ranks are in `production_no_score_all19_CVOTE_RESULTS.json` and its corresponding row file.

## Final confirmation completed once with the locked model

On the90-row15-vessel train-token-composition diagnostic, actual C_vote68/90 becomes70/90. The30 clean rows remain30/30; the60 noisy rows improve38→40, with rescue3/harm1. The model/hash and0.95 threshold remained fixed. All useful gains again come from restoring112 top1:6 restoration changes yield+3/−1, while3 other changes (ranks15→13,20→3,9→2) are all wrong-to-wrong.

The originally frozen score-inclusive verifier was evaluated only as the preregistered contrast:68→69, clean harm0, noise+3/−2. Its one non-restoration change (rank15→2) rescues one case. This contrast was not used to select a different model. No further adjustment followed the final results.

Thus the selected follow-up satisfies the narrow clean-harm0/positive-noise-net diagnostic condition, but the confirmed effect remains a small112-restoration effect. It does not demonstrate large new candidate-selection gains or expected Public0.98. The final dataset is synthetic, and results from15 vessels remain limited. Original112 feature/candidate/score/gate reproduction and reverse-row checks all have zero differences.

No additional tuning, submission ZIP or Public submission has been performed.
