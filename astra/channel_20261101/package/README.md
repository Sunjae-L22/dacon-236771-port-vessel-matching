# ASTRA CHANNEL DIAGNOSTIC

This is an offline diagnostic kit, not a submission model. No Public submission is made and no model ZIP is created. Run the accompanying ASTRA_CHANNEL_DIAGNOSTIC.ipynb before 07:00 KST on 2026-11-01, after F1 finishes.

The original C_vote ZIP SHA256 must be 601c71ab3bc87ee5128305261c1708c432eb75dca53258e4d3841cce813a97d3. The runner verifies the source ZIP, kit files and original CPU component identities. It invokes the original three voter entry points unchanged, checks all voters succeeded, then applies one frozen CPU observation-error channel. Results and each voter prediction are cached. Runtime stops at 07:00 KST or 1800 seconds for the complete diagnostic, whichever occurs first.

Evaluation uses 64 unseen catalog entities with Korean queries composed from official train token correspondences, not copied from KoCat/Kanana. It is still synthetic, shares train vocabulary with baseline, and cannot estimate Public uplift. The fixed validation set has 384 Korean rows and 16 English controls. Truth is separate and is never read by inference.

V3 requires original110 gap<4, channel-relative mass share>=.99 and no identity/one-delete/two-delete path from any winning-vessel alias. This is not a calibrated 99% correctness guarantee. The deletion-only veto was designed on separate development ships. The package contains the predeclared validation dataset and frozen code.

Local actual112 validation:36 rescue/0 harm, clean128 unchanged. Actual full C_vote and L4 execution remain pending. Production local_check3000/600 remains pending. CPU-only repeated/shuffled600 check is not the full production check. Read the accompanying REVIEW.md for limitations and evidence.

Return ASTRA_CHANNEL_RESULTS.zip, including failure results if any. Keep the original submission ZIP and F1 files untouched.
