# Independent static/CPU QA — contrastive pair judge

Status: no execution-blocking defect found in the reviewed selection/preparation/inference source. This is permission to interpret the fixed diagnostic once run, not evidence of accuracy or readiness for submission. Independent reviewer did not load torch, model weights, or run GPU inference.

## Decision implementation

The actual `decision.py::decide(row, cache)` passed all759 possible directed-or-tied graphs on2/3/4candidates;17,664 candidate-pool permutations;759 comparison reversal plus AB/BA swaps;16 valid/invalid response combinations;759 protected-row cases; identical-name abstention; and successive unrelated-row/unchanged-input checks. Results: `GRAPH_PROPERTIES.json`. This checks the concrete algorithm, not a substitute reimplementation. The current pool can reach5; the exhaustive tests cover2–4, while the statically reviewed loop and strict criterion have no size-specific branch.

AB and BA responses map letters back to candidate identities before an edge is created. Two A responses or two B responses yield disagreement, not a win. Invalid or tied responses produce no edge. A candidate must win exactly pool_size−1 comparisons, so cycles/incomplete graphs abstain. At most one strict winner exists; code asserts this. Each call creates fresh win counts and output state. Protected rows return the incumbent before reading cache entries. Missing cached scores raise KeyError and stop evaluation rather than reuse prior state; this is fail-closed diagnostic behavior, not a packaged runtime fallback.

## Tokenization and inputs

Independent tokenizer-only check covered all494 primary ordered prompts and240 supplemental prompts, including both assistant A and B completions (1,468 boundary checks). A=[32], B=[33]; assistant-prefix and separate target-token encodings matched exactly; assistant turn-end ID128009. Primary prompt length108–115, supplemental107–124; both use fixed256 with at least132 tokens of headroom. No truncation or split-token issue was found. `TOKENIZER_CHECK.json` confirms no torch/model/GPU load.

Pure CPU input audit independently reconstructed every ordered cache key and pair count, incumbent membership, original110gap/Korean gate, row ID uniqueness, and catalog name mapping. Primary90 has47hard rows,494 ordered prompts. Supplemental174 has40hard rows,240 ordered prompts. Neither contains a target/truth field in prepared model rows. Every pool pair has both orientations. There are no different candidate IDs with identical English names in either prepared pool. Results: `INPUT_CHECKS.json`.

Cache keys preserve `(query, nameA, nameB)` order. They intentionally share identical semantic prompts across rows; no row metadata/labels enter the prompt. Different voter IDs or initial rankings are not shown to the model.

## Inference source review

`run.py` uses eval mode, BF16 MPS, fixed batch8/leftpad256, explicit position IDs from the attention mask, use_cache=False, and logits_to_keep=1 at the last nonpad prompt position. Last batches repeat the final prompt to8. No labels are read until a predictions-before-truth artifact is saved. Original row order is retained; row reversal is checked again in evaluation.

Full-vocabulary top2 logits must have a strict maximum, and the top token must be32 or33. This correctly rejects a letter tied with a nonletter as well as A/B ties. It validates the **next-token decision**; it does not generate a complete response or prove the model would stop after that letter. That is consistent with a one-step scoring judge but should not be described as complete generated-format validation.

The neural QA compares reversed fixed batches, individual prompts repeated to8, and masked-pad perturbation. It requires greedy token and valid-letter choice equality, while recording logit differences. These sampled checks cannot establish all-row numerical invariance or CUDA/L4 equivalence. Model loading, each inference call, and the scoring loop check the06:45 cutoff; a daemon timer also exits the process at that time. No submission package is produced.

## Research and prior experiments

Qin et al. NAACL2024 §3.1 describes two orders with a tie when they disagree. Their Allpair method aggregates wins/ties; this branch's strict Condorcet replacement and full-vocabulary letter check are additional policies. Its reported document-ranking gains do not establish vessel-name matching gains. Source: https://aclanthology.org/2024.findings-naacl.97.pdf

`external/ksh/tools/probe_target97.py` already tested Kanana10-option letter scoring with metadata. `external/ksh/docs/portfolio_log.md`1088–1089 records Qwen3-4B20-option joint scoring with two orders:7rescues/11harms across60ships. No identical two-candidate consistency plus strict Condorcet implementation was found in the searched sources. This branch is a distinct diagnostic policy, not the first candidate joint comparison or proof that previous failure has been overcome.

## Evaluation limits and supplemental provenance

Primary90 is already-seen synthetic development data with actual C_vote predictions; any benefit remains exploratory. It must not be called independent validation. Previous PMI remains rejected regardless of this branch's outcome.

Supplemental174 consists of all87 remaining eligible vessels after removing the48 used in the PMI screen:44fold2+43fold4. The original eligibility, fewshot exclusion, source-choice hashes, and noise-choice hashes were preserved. Every genuine clean name/sample/target was checked against official train; same-source noise target was also checked. Ground truth stays in a separate file. There was no correctness/target-in-pool selection. All4 rows whose truth is outside top3 remain in the denominator.

Supplemental candidate pool is OOF112top3; no v20/HSW voters are available. Its incumbent is OOF112, not C_vote, and model-row `cvote` is only an adapter field explicitly labeled as such. Hard40 consists of genuineclean7/noise33. The other134 protected rows do not provide evidence of selector safety. Reused folds, small genuine-hard-clean coverage, and different baseline prohibit converting a local gain into a Public0.98 forecast. Record per-fold clean/noise metrics after scoring, without changing the inference policy.

Artifacts in `../validation/`: PROTOCOL.json, MODEL_ROWS.json, ROWS_WITHOUT_TRUTH.json, TRUTH.json, PROMPT_KEYS.json, MANIFEST.json, prepare.py, preflight.py. Source hashes use raw read_bytes SHA256; legacy normalized-text hashes are separately labeled. The original train file was not changed.
