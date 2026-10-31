# Matched-recipe incumbent reference (outer fold2)

This retrains the recipe of79's Korean route on a vessel-disjoint training fold. It never evaluates the existing full-data79 model on heldout train rows, and it is not a submission.

Production79 uses six Korean rankers trained on mixed English/Korean rows, the full allowed v1 LLM set, and augmentation seeds0,1000,2000,5000,6000,7000. All600-tree rankers use63 leaves and learning rate0.05. The original recipe **omits the LightGBM seed argument**. Its serialized default seed is0, with subordinate seeds data1/feature2/bagging3/drop4/objective5/extra6. Explicitly setting `seed=0` changes those subordinate seeds; the current code omits it to preserve the original behavior.

The first comparison trains augmentation seeds0 and5000. Other members are not automatically run without reviewing the pair. Identical real/nogt/LLM feature blocks are cached once; seed-dependent noisy/corrupt/rename blocks retain original ordering. Chunk files include features, labels, candidate IDs, group boundaries, targets and source-query indices for reuse.

The source `src/opus_lib.py` and `src/ko_translit_argon.py` are copied byte-for-byte from the immutable79 ZIP. Fold2 has9,772 fit rows/4,681 fit vessels,6,359 fold-rescored accepted LLM rows, and24,391 queries per augmentation seed. All metadata and learned dictionary pairs use only fit vessels.

## Deliberate differences from the production model

- Training and statistics exclude the heldout vessels.
- The175 existing universal loanword entries are filtered to fit-only support: retain an entry only when its Korean string and corresponding English string both occur in an allowed fit Korean→catalog-target pair. Native and permitted train-only LLM witnesses are recorded.157 entries are retained,18 removed. This avoids assuming that every existing manual entry is independent of heldout labels; it also means this reference has a narrower dictionary than actual79.
- LLM scores are recomputed using fit-only statistics; the globally supplied score column is not used for selection.
- Training uses2 threads for the resource limit instead of the original8; all other recipe parameters are preserved.

The175-entry list is described as the original general loanword list in `argon/007_20261027-1522_reply-opus021-v4lite-oof-and-type-d-features.md`; its origin differs from the48 later train-error-derived additions. The older comment in `opus_lib.py` predates that cleanup. We still keep the support filter and disclose the difference rather than call this a byte-identical production model.

## Evaluation

- Original9 stress conditions plus every native Korean heldout row.
- New `astra/strategy97/benchmark_fold2.csv`, SHA256 `ea4de4d33a95a131307dc602e040815f00418a692e7a6c1a669c2c0d32c4176e`, with uniquequery_id and separatecohorts.
- Model query input is restricted to the original8 allowed input columns. Target labels, corruption family, cohort and effective-change indicators are used only in evaluation.
- Report everycondition/cohort separately. Synthetic condition proportions are not estimates of hidden-test prevalence.
- English results from this mixed-trained Korean-route model are auxiliary diagnostics; they do not validate79's separate65 English route.
- The91 comparison uses its already trained outer-fold2 counterpart (`rankless87_fold2`), not the existing all-train91 submission model. Compare only Korean-source conditions; full-data65 English weights are not used in heldout validation.

No existing data, pinned code, or submission ZIP is modified.
