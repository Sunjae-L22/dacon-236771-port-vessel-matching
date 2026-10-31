# author: sol
"""Teacher-forced candidate likelihoods; reusable by the self-contained Colab notebook.

No generation, model fitting, target labels, chat template, EOS, or external vessel data.
Each query uses its own fixed rank-ordered batches, so test order/subsets do not alter padding.
"""
from pathlib import Path
import gc
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROMPT_VERSION = "raw_ko_en_v1"
KEY = ["query_sample_id", "candidate_vessel_id"]


def encode_pair(tokenizer, ko, en, direction="forward"):
    assert direction in ("forward", "reverse")
    assert str(en).strip() and str(ko).strip(), "empty vessel name"
    if direction == "reverse":
        prompt = f"영문 선박명: {en}\n한국어 선박명:"
        continuation = " " + str(ko)
    else:
        prompt = f"한국어 선박명: {ko}\n영문 선박명:"
        continuation = " " + str(en)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + continuation, add_special_tokens=False)["input_ids"]
    # Fail visibly if a tokenizer merges across the prompt/name boundary.
    assert full_ids[:len(prompt_ids)] == prompt_ids, "prompt boundary token merged"
    assert len(full_ids) > len(prompt_ids) > 0
    return full_ids, len(prompt_ids)


def continuation_logprob(logits, input_ids, continuation_mask, reduction="mean"):
    """Logit t predicts token t+1; count only continuation, never prompt or padding."""
    shifted_mask = continuation_mask[:, 1:]
    targets = input_ids[:, 1:][shifted_mask]
    selected_logits = logits[:, :-1, :][shifted_mask].float()
    logp = -F.cross_entropy(selected_logits, targets, reduction="none")
    counts = shifted_mask.sum(-1)
    assert bool((counts > 0).all()), "no candidate tokens"
    # Separate per-row reductions avoid CUDA atomic scatter_add nondeterminism.
    assert reduction in ("mean", "sum")
    return torch.stack([values.mean() if reduction == "mean" else values.sum()
                        for values in torch.split(logp, counts.tolist())])


def mean_continuation_logprob(logits, input_ids, continuation_mask):
    return continuation_logprob(logits, input_ids, continuation_mask, "mean")


@torch.inference_mode()
def score_batch(model, tokenizer, records, direction="forward", reduction="mean"):
    encoded = [encode_pair(tokenizer, r["vessel_name"], r["candidate_vessel_name"], direction) for r in records]
    max_len = max(len(ids) for ids, _ in encoded)
    device = next(model.parameters()).device
    pad = tokenizer.pad_token_id
    ids = torch.full((len(encoded), max_len), pad, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, (tokens, start) in enumerate(encoded):
        ids[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
        attention[i, :len(tokens)] = 1
        mask[i, start:len(tokens)] = True
    positions = attention.cumsum(-1) - 1
    positions.masked_fill_(attention == 0, 0)
    output = model(input_ids=ids, attention_mask=attention, position_ids=positions,
                   use_cache=False, return_dict=True)
    scores = continuation_logprob(output.logits, ids, mask, reduction)
    return scores.cpu().numpy().astype(float)


def validate_candidates(frame):
    required = KEY + ["rank", "vessel_name", "candidate_vessel_name", "ranker_score", "role", "variant"]
    assert set(required).issubset(frame.columns), f"missing: {set(required) - set(frame.columns)}"
    assert not frame[KEY].duplicated().any(), "duplicate query/candidate"
    assert set(frame["role"]).issubset({"tune", "eval"})
    assert frame[required].notna().all().all(), "missing candidate values"
    for _, group in frame.groupby("query_sample_id", sort=False):
        ranks = sorted(group["rank"].astype(int))
        assert ranks == list(range(1, len(ranks) + 1)), "non-contiguous candidate ranks"
        assert group["vessel_name"].nunique() == 1


def score_candidates(frame, model, tokenizer, model_id, output_path, batch_size=20,
                     checkpoint_queries=100, load_seconds=0.0, existing_ranker_seconds=500.0,
                     deployment_pairs=30000, direction="forward", reduction="mean"):
    """Resume complete queries from one CSV; write progress to the same file every 100 queries."""
    validate_candidates(frame)
    prompt_version = PROMPT_VERSION if direction == "forward" and reduction == "mean" else f"raw_{direction}_{reduction}_v1"
    frame = frame.sort_values(["query_sample_id", "rank"], kind="stable").reset_index(drop=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = None
    if output_path.exists():
        existing = pd.read_csv(output_path, encoding="utf-8-sig", dtype={c: str for c in KEY})
        assert not existing[KEY].duplicated().any(), "duplicate rows in checkpoint"
        assert set(existing["llm_model"]) == {model_id}, "use a different output file for another model"
        assert set(existing["llm_prompt_version"]) == {prompt_version}, "prompt changed: start a new output"
        assert set(existing["llm_batch_size"].astype(int)) == {batch_size}, "keep batch size on resume"
        assert np.isfinite(existing["llm_logprob"]).all()
        checked = existing.merge(frame, on=KEY, how="left", suffixes=("_saved", ""), validate="one_to_one", indicator=True)
        assert (checked["_merge"] == "both").all(), "checkpoint contains unknown candidates"
        for col in ["vessel_name", "candidate_vessel_name", "rank", "ranker_score"]:
            left, right = checked[col + "_saved"], checked[col]
            equal = np.isclose(left, right, rtol=0, atol=1e-12) if col == "ranker_score" else left.eq(right)
            assert bool(np.asarray(equal).all()), f"checkpoint input changed: {col}"
    complete = set()
    if existing is not None:
        expected = frame.groupby("query_sample_id").size()
        done = existing.groupby("query_sample_id").size()
        complete = set(done.index[done.eq(expected.reindex(done.index))])
        existing = existing[existing["query_sample_id"].isin(complete)].copy()
    pieces = [] if existing is None or existing.empty else [existing]
    todo = [(qid, group) for qid, group in frame.groupby("query_sample_id", sort=False) if qid not in complete]
    print(f"Resume {len(complete)} complete queries; remaining {len(todo)} queries / {sum(len(g) for _, g in todo)} pairs", flush=True)

    def checkpoint():
        saved = pd.concat(pieces, ignore_index=True).sort_values(["query_sample_id", "rank"], kind="stable")
        # Atomic replacement is a temporary write, not an extra retained snapshot.
        tmp = output_path.with_name(output_path.name + ".tmp")
        saved.to_csv(tmp, index=False, encoding="utf-8-sig")
        tmp.replace(output_path)
        return saved

    torch.cuda.synchronize()
    start = time.perf_counter()
    infer_seconds = 0.0
    fresh_pairs = 0
    for number, (_, group) in enumerate(todo, 1):
        records = group.to_dict("records")
        result = []
        torch.cuda.synchronize()
        tick = time.perf_counter()
        for offset in range(0, len(records), batch_size):
            result.extend(score_batch(model, tokenizer, records[offset:offset + batch_size], direction, reduction))
        torch.cuda.synchronize()
        infer_seconds += time.perf_counter() - tick
        scored = group.copy()
        scored["llm_logprob"] = result
        scored["llm_model"] = model_id
        scored["llm_prompt_version"] = prompt_version
        scored["llm_direction"] = direction
        scored["llm_reduction"] = reduction
        scored["llm_batch_size"] = batch_size
        assert np.isfinite(scored["llm_logprob"]).all()
        pieces.append(scored)
        fresh_pairs += len(group)
        if number % checkpoint_queries == 0 or number == len(todo):
            checkpoint()
            rate = fresh_pairs / max(infer_seconds, 1e-9)
            print(f"{number}/{len(todo)} queries; {fresh_pairs} new pairs; inference {infer_seconds:.1f}s; {rate:.2f} pairs/s", flush=True)
    saved = checkpoint() if pieces else frame
    assert len(saved) == len(frame) and not saved[KEY].duplicated().any(), "incomplete scoring"
    elapsed = time.perf_counter() - start
    if fresh_pairs:
        rate = fresh_pairs / infer_seconds
        # Use end-to-end scoring time, including CSV/Drive work, for the conservative estimate.
        end_to_end_rate = fresh_pairs / elapsed
        projected = load_seconds + existing_ranker_seconds + deployment_pairs / end_to_end_rate
        print(f"New pairs={fresh_pairs}; GPU inference seconds={infer_seconds:.2f}; wall seconds={elapsed:.2f}")
        print(f"Inference pairs/s={rate:.3f}; wall pairs/s={end_to_end_rate:.3f}")
        print(f"30,000-pair estimate: model load {load_seconds:.1f}s + ranker {existing_ranker_seconds:.1f}s + scoring {deployment_pairs / end_to_end_rate:.1f}s = {projected:.1f}s / 1800s")
        print(f"CUDA peak allocated GiB={torch.cuda.max_memory_allocated() / 2**30:.3f}; reserved GiB={torch.cuda.max_memory_reserved() / 2**30:.3f}")
        saved["llm_session_pairs_per_second"] = rate
        saved["llm_session_wall_pairs_per_second"] = end_to_end_rate
        saved["llm_session_load_seconds"] = load_seconds
        saved["llm_session_inference_seconds"] = infer_seconds
        saved["llm_session_scoring_wall_seconds"] = elapsed
        saved["llm_projected_total_seconds_30000_pairs"] = projected
        tmp = output_path.with_name(output_path.name + ".tmp")
        saved.to_csv(tmp, index=False, encoding="utf-8-sig")
        tmp.replace(output_path)
    print(f"Saved {len(saved)} pairs: {output_path}", flush=True)
    return saved
