# Independent forward-PMI audit

Static decision: **the proposed candidate-dependent correction is mathematically meaningful and the inspected likelihood indexing is correct.** No scoring or GPU inference was launched by this audit. Runtime numerical QA and measured ranking benefit remain separate requirements.

## Mathematical interpretation

For observed Korean name K, candidate English name E, fixed demonstration prompt D, and assistant-turn terminator EOS, the implemented score is

`[log P(E,EOS | K,D) - log P(E,EOS | N/A,D)] / (tokens(E)+1)`.

The denominator of the likelihood ratio depends on the candidate, so subtracting it can change candidate order. This is unlike subtracting `log P(K)` from reverse `log P(K|E)` for fixed K: that subtraction is common to every candidate and cannot change their ordering.

The construction is related to [Holtzman et al., EMNLP 2021](https://aclanthology.org/2021.emnlp-main.564/), which scores options relative to a task-conditioned prior. Here literal `N/A` is a **reference prompt proxy**, not a demonstrated marginal distribution of ship names. The most precise description is a token-normalized, N/A-reference log-likelihood ratio. The paper's multiple-choice gains do not establish transliteration or vessel-matching gains.

Subtracting a weak prior can over-reward rare or implausible strings. N/A also changes the instruction context: a transliteration model may prefer copying `N/A` or declining an empty input rather than sampling from a ship-name domain. The denominator is therefore not guaranteed to remove only irrelevant surface-frequency bias. Token normalization is a fixed additional ranking heuristic; the resulting score is not a normalized posterior. Adding it at lambda1 to a tree ranker's raw score is an ablation, not a calibrated Bayesian combination.

## Source audit

- Original `forward_name_likelihood.py` hardcodes batch32 in its constructor. Passing batch8 again through kwargs would fail. New `scorer.py` correctly initializes `ReverseNameScorer` directly with batch8/pad512.
- New `reverse_local.py` is byte-identical to the inspected original `reverse_name_likelihood.py`.
- The input is prompt plus target tokens, **without an input EOS**. The final T+1 next-token logits are correctly aligned with target tokens plus assistant-turn EOS. The final target-token position predicts EOS. `logits_to_keep=33` covers the maximum 32-token target plus its terminator.
- Padding is left-aligned to fixed512, mask positions use cumulative valid-token counts, and the last partial batch repeats its last pair to eight rows. The original implementation rejects too-long targets/prompts; it does not truncate silently.
- The new forward subclass derives the assistant-turn terminator from the actual chat template and asserts 128009. This is safer than assuming a generic end-of-text token.
- `run.py` computes `(cond_sum+cond_EOS-prior_sum-prior_EOS)/(T+1)` exactly and asserts identical conditional/prior target token counts. No second mean normalization or sign reversal was found.
- Its independent CE calculation shifts full input labels by one and aligns the final33 positions with model outputs. The target-tail assertion is correct. Padding perturbation preserves mask/position IDs and only changes masked token IDs.

The inherited scorer `protocol()` would still describe a reverse prompt if called directly. The new separate `PROTOCOL.json` accurately describes the actual forward implementation; inherited metadata should not be treated as its protocol without an override. This has been reported to the implementation agent.

## Independent tokenizer-only check

`tokenizer_check.py` loads only the actual payload tokenizer, with Torch/TensorFlow disabled. Across five conditioning strings and 204 sampled/artificial candidate targets (1,020 combinations), full assistant-chat tokenization exactly matches separate prefix plus target tokenization. The assistant-turn terminator is 128009, which also equals this tokenizer's EOS ID.

Prefix lengths are 392–439 tokens, and maximum prompt+target length is450 in that check. No sample exceeds512. The implementation agent's separate final-pair preflight reports maximum406 and maximum target length10 for the selected experiment. The broader check is not a substitute for that final-input validation.

Results are stored in `TOKENIZER_CHECK.json`. This audit did not import Torch, load model weights, allocate GPU tensors, or generate candidates.

## Selection and numerical-independence limits

The prepared sample has96 queries from48 distinct vessels, 24 vessels per fold. It uses genuine train clean names, one fixed hash-chosen noise variant per vessel, and filters few-shot matches. Selection does not inspect prediction correctness or target presence. Unsupported targets remain in the denominator. The28 hard rows all contain Hangul; there are only7 active clean rows and21 active noisy rows. Thus clean-harm-zero on the96-row sample is weaker evidence than testing48 actively modified clean queries.

Fixed batch and padding shapes reduce numerical variability but do not prove MPS BF16 row/slot/companion invariance. Global pair sorting changes microbatch companions when a different query subset is evaluated. Actual reversed-order, repeated-singleton, and padding-perturbation checks are necessary before reusing caches.

The declared .001 numerical tolerance is not an exact prediction-invariance proof. Near-tied `KC+PMI` totals can change winner under smaller perturbations. If nonzero differences occur, report them as numerical stability measurements and check final winner stability; do not label them an exact row-independence pass. These cautions were sent directly to the implementation agent.

The prototype is a reused-development-fold single-KC-model screen. Original C_vote ranking, fresh confirmation, L4 performance, and Public improvement are not established. No lambda, prior string, target length rule, or gate should be changed after results under this fixed experiment.

## Cutoff

The run watchdog is06:35 KST or45 minutes. The original draft checked before loading and before each score batch, but not between multiple QA calls; adding checks before QA model calls was recommended. A single in-flight model call can still finish after a wall-clock cutoff unless execution is externally interrupted. No new candidate work after07:00 is authorized.
