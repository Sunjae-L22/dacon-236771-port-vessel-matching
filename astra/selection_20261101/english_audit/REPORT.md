# English selection audit — 2026-11-01 06:01 KST

## Decision

No new English candidate is justified by the accessible evidence. There is a real historical failure family — changed ship/owner names competing with a name-similar wrong vessel — but the project already implemented exact-GT candidate coverage and rename augmentation for that family. This audit did not find an independent, large, currently unaddressed ranking mechanism. No model was trained or evaluated with train exclusion removed, no GPU was used, and no candidate ZIP or submission was made.

The reported current C_vote English score is 3,506/3,568; the 62 Public wrong-row identities and labels are unavailable. Consequently these historical errors cannot be identified as C_vote errors. One English rescue contributes 0.3/3,568 = 0.0000840807175. Even correcting all 62, with zero harm, adds at most 0.00521300448, taking an approximately 0.96189 baseline only to approximately 0.96710. English alone cannot reach 0.98. This is a ceiling from the reported counts, not a forecast of achievable gain.

## Existing real-data OOF evidence

Read `shared/experiments/cache/oof_v5.csv` and its five `v5_fold*.npz` caches. These contain 12,222 rows, including 11,031 English rows from 5,471 vessels. All 5,851 targets across both languages occur in exactly one fold. The owning `shared/src/cv.py` defines vessel-grouped folds, and `shared/src/opus_cv.py` fits an index on the complementary folds. No training or inference was rerun. The old v5 snapshot is not present, so this is a source-consistent historical-cache audit, not complete reconstruction of its original training provenance.

English v5 has 9 wrong rows. For every cached validation group, candidate IDs and `yva` were cross-checked against the catalog target index; there were no discrepancies. Seven wrong rows have the true candidate present; two do not. Candidate list order is insertion order, not the model score ranking. **No true-candidate top20 or current112 top20 claim is supported.** Saved per-candidate v5 model scores are not present in these caches.

| Query | True name | Wrong prediction | True candidate present | Pattern |
|---|---|---|---|---|
| BOW SUCCESS | Fairchem Success | Bao Success | Yes | Owner-prefix change; correct GT exact |
| CENTRAL PARK | Caifu Chao | Central | Yes | Full rename; correct GT exact |
| CX KYTHIRA | Clipper Kythira | Kythira I | Yes | Prefix change; correct GT exact |
| FAIRCHEM BRONCO | City Island | Fairchem Rooibos | Yes | Full rename; correct GT exact |
| GINGA SUNBIRD | Fairchem Sunbird | Sunbird | Yes | Owner-prefix change; correct GT exact |
| DIIJLAH | Dijilah | Dijlah | Yes | Close-name collision; query GT missing |
| ROSASNA | Rossana | Rosanna | Yes | Close-name collision; correct GT about 4.4% below query |
| HAFNIA WREN | BW Wren | Hafnia Raven | No | Owner-prefix change; retrieval miss |
| HAFNIA MERLIN | BW Merlin | Hafnia Ametrine | No | Owner-prefix change; retrieval miss |

Thus 5/7 historical selection failures reflect excessive name reliance despite exact true GT, not demonstrated metadata overtrust. The two near-name cases differ in available evidence, and do not establish a large learnable common failure mode. Historical v3/v4 already had only 7 English errors each, while v5 had 9; this audit does not select a model based on those numbers.

## Why the tempting broad directions are repetitions

`opus/010_20261022-1800_rename-discovery.md` already identifies the real rename family and the historical shortage of such pairs (~12 ships). It explicitly introduced v9 exact-GT candidate coverage and v10 realistic owner-prefix/full-name augmentation. Current `shared/src/opus_lib.py::augment_rename` still implements owner-prefix changes (45%), new combinations (40%), and another vessel's name (15%). These examples support that existing rationale; they do not establish a new untried remedy.

Current `shared/src/opus_lib.py` also records that English token features were disabled after submissions 16/17 fell from approximately .930 to .912 compared with 18 without them. Merely adding brand/last-token features is not a distinct justified experiment.

The existing independent English reconstruction probe at `astra/research_20261101/newsignal/english_eval/REPORT.md` held out 1,097 real input forms/1,096 vessels by vessel hash and compared full-catalog normalized-name similarity with the English noise-channel approach. Both scored 1,095/1,097 with zero prediction changes. This is not a112/C_vote ranker evaluation. It nevertheless supplies no evidence that ordinary transcription restoration is an untapped large source of complementary English gain.

## Evidence needed before a different English selector is justified

A frozen current-English ranker, trained without a vessel-heldout evaluation set, would need per-candidate scores and current-mask-respecting candidate lists on real heldout English queries, including naturally occurring renames and name collisions. This audit did not locate such a current112 OOF trace. Running the fully trained112 model on known train targets after lifting its exclusion mask would not supply that evidence and was not done.

If future traces show a substantial residual family, a learned conditional reliability model for name versus stable metadata would be a coherent broad mechanism. It is **not proposed as a new candidate here**: rename augmentation already teaches a similar tradeoff, natural training examples are sparse, and neither current residual prevalence nor independent incremental gain is established. Current priority should remain the ongoing frozen Korean likelihood experiment rather than spend deadline time or submissions on this English hypothesis.

Reproduction: run `audit.py` with the absolute Python3.11.9 interpreter; it writes only this folder's `RESULTS.json`. The JSON records all nine examples and explicitly separates candidate presence from score rank. No Public labels were read.
