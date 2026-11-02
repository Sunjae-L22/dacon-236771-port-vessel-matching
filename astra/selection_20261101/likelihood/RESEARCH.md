# Why this is a different, bounded test

Holtzman et al. describe domain-conditional PMI: subtract each answer's task-domain prior likelihood rather than rank options solely by conditional string probability. Their evidence concerns multiple-choice language tasks, not vessel names. [EMNLP 2021 primary paper](https://aclanthology.org/2021.emnlp-main.564/)

Zhao et al. identify answer biases associated with prompts and common answers, and estimate such bias with content-free inputs such as `N/A`. Their contextual calibration experiments also do not establish benefit for Korean vessel transliteration. This experiment borrows the content-free input, fixes the domain prompt, and uses a length-normalized log-probability difference; it is a narrow application hypothesis, not a claimed replication of all their calibration choices. [ICML 2021 primary paper](https://proceedings.mlr.press/v139/zhao21c.html)

The existing `external/ksh/docs/forward_name_probe_20261029.md` reports a failed Kanana forward-likelihood experiment: 48 selected vessels, zero rescues and two harms. Its forward conditional score mixes full English names and spans; it does not subtract the English candidate's `N/A` prior. The original forward/reverse teacher-forcing implementation is reused read-only, with a local copy and an explicit batch8 constructor to avoid the original forward class's hardcoded batch32.

The existing reverse-likelihood discussion correctly warns that subtracting `log P(Korean query)` from `log P(Korean query|English candidate)` is constant across candidates. Here the direction is forward and the subtracted `log P(English candidate|N/A)` differs across candidates. These are mathematically different adjustments.

Erde's existing graphone pipelines already subtract a candidate LM prior from learned transliteration scores. Thus prior correction is not novel in this workspace at the broad method-family level. A source search found no identical Kanana forward-domain `N/A` subtraction. This experiment tests that different frozen neural score; it does not present PMI itself as newly invented or as proven high-gain evidence.

The selected clean queries come directly from official human-written training rows and are verified against sample ID, name and target. They are not copied from generated alias tables. The OOF ranker still uses synthetic catalog aliases, the folds have been reused, and noise variants remain synthetic; those limitations survive this better clean-query provenance.
