# Frozen forward domain-PMI: no promotion

The96-row screen completed in1,038.32seconds (17.31minutes). It did not satisfy the preregistered requirement of positive noisy-row net gain in **both** folds. No production follow-up, candidate ZIP or submission was executed. The MPS process finished successfully and released its model.

| Fold | Native clean rescue/harm | Noisy rescue/harm | Overall correct, baseline→PMI | PMI screen |
|---|---:|---:|---:|---|
|2,24 vessels|2/0|1/0|40→43 of48|Pass|
|4,24 vessels|0/0|0/0|47→47 of48|No positive noise gain|

The clean names are exact human-written official training names, verified against source sample ID, name and target. They are not copied from catalog alias tables. One fixed synthetic noisy variant was selected per vessel. Selection used vessel/source hashes, not prediction correctness or candidate coverage. Each fold contributes24 native-clean and24 noisy rows; the original110 gate permits changes in only7 clean and21 noisy rows overall.

## What the control comparison actually shows

Uncorrected conditional likelihood also reaches43/48 and47/48. Equal counts do **not** mean identical predictions: PMI differs from that control in four fold2 rows and one fold4 row.

- Both methods rescue the clean names `루손` and `베마그레이스`, selecting original112 candidate ranks2 and9.
- PMI alone rescues noisy `루손손` (rank2).
- Uncorrected conditional likelihood alone rescues `에스스그레이스`; PMI retains the wrong baseline answer there.
- The other differences remain wrong-to-wrong.

The candidate-prior correction therefore adds one distinct rescue while losing a rescue available to the uncorrected control: **additional net gain0** on this screen. It is not evidence of a large improvement from PMI or expected Public0.98. The control remains diagnostic; it was not substituted for the frozen primary after the result.

## Important limitation of the screen

The only fold4 baseline error has its truth outside the fixed top20. Consequently no candidate selector could achieve positive net gain in that fold on the selected sample. This is a statistical-power/design limitation, not evidence that the model harms fold4. This attainable-error audit should have been made explicit before the expensive inference. The sample and criterion were not changed after this was observed.

Fold2 contains four reachable baseline errors; PMI fixes three. Across all96 rows, five truths are outside the fixed pool and remain in the denominator. The folds have been reused in earlier development, and the original ranker still uses generated catalog aliases even though the clean query strings are genuine human names. These limitations prevent treating the screen as independent CV confirmation.

## Implementation and numerical checks

The only model inputs are a Korean query (or fixed `N/A`) and a normalized full English candidate name. No candidate ID, label or metadata enters the model. Existing Kanana weights are reused without training. The fixed score is:

`OOF112 KC score + (logP(fullEnglish+turnend | query) - logP(fullEnglish+turnend | N/A)) / (English token count+1)`

Lambda1, original top20 and original110 hard gate remain fixed. The runtime is BF16/MPS, batch8, left padding512, transformer4.57.6, torch2.11.0, and assistant-turn-end128009. Max actual sequence length is406; max target length10. Original source files and weights remain unchanged.

On the fixed numerical QA sample, independent shifted cross entropy, reversed batch order, singleton repeated to the same batch shape and masked-padding token perturbation all give maximum difference exactly0.0. This is sampled numerical QA, not a proof of every possible production ordering or CUDA/MPS equivalence. Protected rows changed0 times.

All1,070 unique conditional/prior pairs and scores are retained in `MODEL_PAIRS.json` and `PAIR_SCORES.npy`. Selection, source provenance, results and row comparisons are in `MANIFEST.json`, `SELECTED_ROWS.json`, `RESULTS.json` and `RESULTS_ROWS.jsonl`.

## Follow-up status and research distinction

Before outcomes, a production policy was frozen to overlay only a genuinely changed112 top1 on C_vote; a mere return to the existing112 answer would not count as a new signal. Its predefined90-row scope contains42 eligible rows and1,422 uncached pairs. A time-only amendment extended its internal stop to06:45 while preserving the user's07:00 freeze. **It was not executed because the original screen condition was not met**, regardless of that additional time allowance.

This neural candidate-prior subtraction differs from the existing failed forward conditional-only probe, but broad PMI/LM-prior correction already exists in Erde's graphone pipeline. The mechanism draws on [domain-conditional PMI, EMNLP2021](https://aclanthology.org/2021.emnlp-main.564/) and [content-free contextual calibration, ICML2021](https://proceedings.mlr.press/v139/zhao21c.html); neither paper establishes vessel-matching improvement. `RESEARCH.md` records the source comparison.
