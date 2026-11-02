# Independent audit — in progress

Primary research checked: Qin et al., NAACL2024 Findings, https://aclanthology.org/2024.findings-naacl.97.pdf, §3.1–3.2. Comparing both input orders and declaring ties when they disagree follows the paper. Strict all-opponent Condorcet replacement and requiring full-vocabulary A/B argmax are additional conservative policies; the paper primarily ranks aggregate pairwise wins. Document-ranking evidence does not establish vessel-transliteration accuracy.

Prior-work check: `external/ksh/tools/probe_target97.py` already tested Kanana10-option letter scoring with metadata; `external/ksh/docs/portfolio_log.md` lines1088–1089 records Qwen3-4B20-option joint scoring using two input orders (7rescues/11harms in60ships). This branch is not the first joint candidate comparison. No previous implementation of two-candidate swapped-order consistency plus unique strict Condorcet selection was found in the searched project sources.

Preparation source review: primary `prepare.py` forms C_vote from original110 gap and all three original voters; candidate pool contains allowed112top3 and the three incumbent voters. True labels are not read. The single-stage decision protocol preserves English/easy rows. Cache keys serialize ordered triples `(query,nameA,nameB)`; reversing candidates creates a different key unless the candidate English names are identical. Identical names are observationally indistinguishable and must yield no consistent edge, not an arbitrary ID win.

Supplemental CPU preparation is in ../validation/. All87remaining eligible vessels after the48PMI-screen ships are retained;174rows, hard40(clean7/noise33), ordered240prompts. All targets outside top3 remain in the denominator. True labels are separate. The validation pool uses OOF112top3 and cannot be reported as actualC_vote. Exact primary prompt tokenizer preflight max124tokens, pad256, headroom132, A/B IDs32/33 and every assistant boundary verified. No torch/model/GPU was loaded by the independent tokenizer probe.

Completed: see REPORT.md, GRAPH_PROPERTIES.json, INPUT_CHECKS.json and TOKENIZER_CHECK.json.
