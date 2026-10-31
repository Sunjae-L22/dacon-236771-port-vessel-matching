# Transliteration literature and overlap audit — 2026-11-01

Scope: read-only inspection of existing models and primary literature. No training, GPU, Public submission, or submission ZIP. Only this research directory is written. The parent experiment owns any channel implementation and validation.

## Decision

Do **not** retrain a generic transliterator or add another learned edit-distance score: the C_vote HanSunwook voter already contains stochastic EM transliteration, context-dependent graphones, n-best back-transliteration and typo tolerance. A distinct, CPU-feasible hypothesis remains: separate **whole-syllable observation corruption** from transliteration, score the exact probability of delete/duplicate/adjacent-swap paths from each catalog Korean alias, and marginalize aliases with normalized priors. This is a hypothesis, not a predicted leaderboard gain.

Even the syllable operations themselves are not new here: Astra's earlier query restorer already used all three. The new claim must be restricted to **full-name generative likelihood, learned event proportions/path probabilities, and normalized catalog-alias evidence**, rather than a dictionary-restoration candidate union or generic similarity.

## Existing implementation, checked directly

Paths below are relative to the project root.

| Component | Actual existing mechanism | Consequence |
|---|---|---|
| `astra/arena_20261101/baseline/model/hs/model/matcher.py:662` | Stochastic many-to-many transliteration, eight forward-backward EM iterations; NFD jamo versus Latin units `(1,0),(0,1),(1,1),(1,2),(1,3)` | A fresh generic noisy-channel/EM model repeats an existing voter. |
| Same file around 875 | Joint graphone bigram with interpolated absolute discounting, explicitly citing Li et al. and Bisani–Ney | “Use joint sequence models” alone is not a new experiment. |
| Same file `1190–1320` | Beam 48 per token, 10 unit outputs, optional one inserted Latin letter between jamo, 30 name spellings, 200 n-gram neighbors, top 10 extra candidates | Back-transliteration candidate expansion is already implemented. |
| Same file `_word_orders` | Written order plus one adjacent **word** swap, at most four orders | Not a model of adjacent Hangul **syllable** swaps inside a word. |
| Same file `2059`, `_tngt_block` | Viterbi can skip candidate Latin letters at fixed log cost −8; log-odds against independent letter/jamo backgrounds | Fixed Latin-side deletion tolerance is already present; neither a new generic likelihood ratio nor a deletion feature is sufficient novelty. |
| Same file near 65 | OSA distance, including character transpositions | Plain character edit-distance novelty is absent. |
| `shared/src/ko_translit_argon.py:659` | Train-pair exact stems, token pairs and DP-segmented subwords | A train-pair dictionary/segmenter is also already present in 112's feature family. |
| `astra/query_restore_20261031/restorer.py` | Reverse map of one syllable deletion/duplication/adjacent swap for dictionary words of length at least four; segmentation DP allows two edits across words; max three views; abstains on exact coverage or ambiguity | Closest prior Astra experiment. Full-name probability and catalog-wide alias scoring must demonstrate value beyond this restoration approach. |
| `astra/sequence90/pilot.py` and `report.json` | Learned weighted character/digraph edit costs from actual pairs; previous fold4 +2/−1, uncertainty included zero, not adopted | Another small edit-feature gain would not justify claiming a route to .98. |

## Primary literature and what it actually supports

- [Brill and Moore, ACL 2000](https://aclanthology.org/P00-1037/): string-to-string error channels generalize uniform single-character spelling errors. This supports separating observation corruption from a pronunciation/transliteration channel; it does not establish improvement on ship matching.
- [Li, Zhang and Su, ACL 2004](https://aclanthology.org/P04-1021/): joint source–target sequence models learn transliteration correspondences without a mandatory phonemic intermediate. This family is already in HSW.
- [Ristad and Yianilos, Learning String-Edit Distance](https://arxiv.org/abs/cmp-lg/9610005): learned stochastic edits provide a principled alternative to fixed edit cost. Also already represented by HSW's stochastic edit machinery.
- [Mohri, CL 1997](https://aclanthology.org/J97-2003/), and [Mohri, Pereira and Riley, 2002](https://research.google/pubs/weighted-finite-state-transducers-in-speech-recognition-3/): weighted finite-state composition combines separate lexical, pronunciation and observation models. The useful transfer is composition of an alias/transliteration relation and a corruption relation, not adoption of a new package under deadline.
- [Wang, Wu and Tsai, NEWS 2015](https://aclanthology.org/W15-3913/): English–Korean transliteration compared grapheme segmentation and Hangul-block/romanized representations with M2M alignment and DirecTL+. It supports testing the right unit of corruption. External pronunciation dictionaries and web data used in some conditions are not proposed here.
- [Sajjad et al., CL 2017](https://aclanthology.org/J17-2003/): transliteration mining can separate transliteration and background models. HSW already has a log-odds background score, so this alone adds no new signal.
- [Jung et al., EACL 2026](https://aclanthology.org/2026.eacl-long.365/): recent work finds romanization useful for multilingual NER/POS/NLI through vocabulary sharing. Those are different tasks; it provides no evidence for .98 vessel matching or for reusing the existing v20 encoder more aggressively.

## Concrete composed-channel algorithm

For catalog vessel `v`, clean Korean alias `a`, observation `q`, define

`P(q | v) = sum_a P(a | v) sum_{k=0..2} P(k) sum_paths(a -> q, k) P(path | a,k)`.

Each event is a whole-syllable deletion, duplication, or adjacent swap. Normalize event location choices and event-type probabilities; an event that makes no change must be consistently excluded or explicitly modeled. Repeated symbols can produce the same output through multiple distinct paths: retain their probability mass. Use one frozen generative definition across fitting and inference.

The parent's separate English audit reports observed corruption proportions, with an explicit caveat that English event proportions are not proof of Korean proportions. Transfer only operation-type priors as a hypothesis; English character-conditioned probabilities cannot be transferred directly to Korean syllables.

Implementation under the deadline:

1. Native-code broad prefilter on aliases, then exact one/two-event enumeration for the surviving aliases. Unrestricted Damerau–Levenshtein at distance at most two is compatible with two sequential adjacent-swap events. A more conservative Indel-distance threshold four is safe for these operations.
2. Explicitly enumerate first-event intermediates and second-event paths. A usual monotone OSA dynamic program allows only disjoint edit transitions and can miss overlapping swaps or repeated duplication. Calling that exact composition would be incorrect.
3. Cache by normalized query and alias, not row position. Compute output with stable IDs; do not use previous-row state.
4. Normalize alias priors. A vessel with many aliases must not automatically acquire more total prior mass. Record both best explanatory alias/path and marginalized evidence for diagnosis.
5. Exact-match prior, one/two-event prior and thresholds must be frozen before evaluating the holdout. Query clean roots must be actual train Korean names on held-out vessels, not generated from the same aliases being searched. Reject known prompting examples where possible.

## Evidence required before any candidate

- Incremental retrieval and ranking gain against actual C_vote predictions, not only generic similarity.
- Clean real-name damage zero on the stated local set; report counts and uncertainty, not a universal safety claim.
- Separate native-real queries from injected corruptions; do not extrapolate a synthetic average into Public accuracy.
- Show whether recovered target was outside the old candidate set and whether all-three-agree mistakes can be addressed. Restricting only to the old 112-versus-HSW disagreement region preserves the earlier gain ceiling.
- Actual L4 runtime and package validation remain mandatory. No literature result supplies them.

This audit has no standalone measured C_vote improvement. The parent owns the real-name channel probe. The literature narrows the implementation and rejects redundant approaches; it does not certify a leaderboard target.
