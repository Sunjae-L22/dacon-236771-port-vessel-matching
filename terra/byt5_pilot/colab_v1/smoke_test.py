# author: terra
"""Tiny random CPU model checks. Never downloads or claims pretrained/GPU accuracy."""
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch
from transformers import ByT5Tokenizer, T5Config, T5ForConditionalGeneration, Adafactor

from runtime_common import make_batch, per_example_nll, sequence_scores, sha, write_json
from evaluate_byt5 import report


def main():
    torch.set_num_threads(1); torch.manual_seed(42)
    tok = ByT5Tokenizer()
    cfg = T5Config(vocab_size=384, d_model=32, d_ff=64, num_layers=1, num_decoder_layers=1,
                   num_heads=4, d_kv=8, pad_token_id=0, eos_token_id=1, decoder_start_token_id=0,
                   dropout_rate=0.)
    model = T5ForConditionalGeneration(cfg)
    english, korean = ["Sea Wolf", "SH Freesia"], ["씨 울프", "에스에이치 프리지아"]
    batch = make_batch(tok, english, korean, "cpu")
    assert (batch["labels"] == -100).any(), "Padding must be masked"
    losses = per_example_nll(model, batch)
    assert torch.isfinite(losses).all() and losses.shape == (2,)
    optimizer = Adafactor(model.parameters(), lr=.001, relative_step=False, scale_parameter=False)
    losses.mean().backward(); optimizer.step(); optimizer.zero_grad()
    a = sequence_scores(model, tok, english, korean, "cpu", 2)
    b = sequence_scores(model, tok, english[::-1], korean[::-1], "cpu", 1)[::-1]
    assert np.allclose(a, b, atol=1e-5), (a, b)
    try:
        make_batch(tok, ["X"*200], ["한글"], "cpu")
        raise AssertionError("Overflow was silently truncated")
    except AssertionError as exc:
        assert "Source exceeds" in str(exc)
    with tempfile.TemporaryDirectory() as temp:
        out = Path(temp)
        rows = []
        for role in ("tune", "eval"):
            for variant in ("orig", "typo1"):
                for rank in (1, 2):
                    rows.append({"query_sample_id": f"{role}_{variant}", "role": role,
                        "variant": variant, "target_vessel_id": role+"_true", "candidate_vessel_id": f"c{rank}",
                        "rank": rank, "ranker_score": float(3-rank), "byt5_score": float(rank), "is_target": rank==2})
        data = pd.DataFrame(rows)
        data.to_csv(out/"scored_candidates.csv", index=False)
        def receipt():
            write_json(out/"scoring_manifest.json", {"scored_candidates_sha256":sha(out/"scored_candidates.csv"),
                                                    "batch_probe_pass": True})
        receipt(); report(out)
        first = json.loads((out/"pilot_report.json").read_text())["selected_alpha"]
        # Changing evaluation outcomes must not change tune-selected alpha.
        data.loc[data.role.eq("eval"), "is_target"] = ~data.loc[data.role.eq("eval"), "is_target"]
        data.to_csv(out/"scored_candidates.csv", index=False)
        receipt(); report(out)
        second = json.loads((out/"pilot_report.json").read_text())["selected_alpha"]
        assert first == second
    print(json.dumps({"status":"PASS_TINY_RANDOM_CPU_ONLY", "torch":torch.__version__,
                      "checks":["masked per-sequence NLL", "optimizer backward/step", "batch/order score consistency",
                                "no silent truncation", "fusion alpha independent of eval outcomes"]}))


if __name__ == "__main__": main()
