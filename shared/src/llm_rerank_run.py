# author: sol
"""Run the unchanged Track B scorer/evaluator in a fresh, isolated Colab process."""
import argparse
import gc
import os
from pathlib import Path
import time
from functools import partial

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

from score_llm_candidates import encode_pair, score_batch, score_candidates, validate_candidates
from evaluate_llm_rerank import evaluate
from evaluate_llm_grid import evaluate_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--direction", choices=["forward", "reverse"], default="forward")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU가 연결되지 않았습니다. 런타임 유형을 L4 GPU로 바꾼 뒤 모두 실행하세요.")
    print(f"GPU={torch.cuda.get_device_name(0)}; torch={torch.__version__}; transformers={transformers.__version__}", flush=True)
    if "L4" not in torch.cuda.get_device_name(0):
        print("현재 GPU는 L4가 아니므로 서버 시간 실측으로 간주하지 않습니다.", flush=True)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # The math attention backend keeps the repeated fixed-query check deterministic.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    frame = pd.read_csv(args.work / "candidates_top20.csv", encoding="utf-8-sig",
                        dtype={c: str for c in ["query_sample_id", "orig_sample_id", "candidate_vessel_id", "target_vessel_id"]})
    validate_candidates(frame)
    assert len(frame) == 78540 and frame.query_sample_id.nunique() == 3927
    reduction = "sum" if args.direction == "reverse" else "mean"
    tag = args.model.split("/")[-1] + ("_reverse" if args.direction == "reverse" else "")
    scores_path = args.work / f"scores_{tag}.csv"
    eval_path = args.work / f"evaluation_{tag}.csv"
    batch_size = args.batch_size
    if scores_path.exists():
        previous = pd.read_csv(scores_path, nrows=1)
        assert previous.llm_model.iloc[0] == args.model
        batch_size = int(previous.llm_batch_size.iloc[0])

    tick = time.perf_counter()
    local_model = Path(snapshot_download(args.model, allow_patterns=["*.safetensors", "*.json", "*.txt", "LICENSE"]))
    print(f"Model download/cache seconds={time.perf_counter()-tick:.1f}", flush=True)
    tick = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(local_model, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        local_model, local_files_only=True, torch_dtype=torch.float16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa", trust_remote_code=False,
    ).eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - tick
    torch.cuda.reset_peak_memory_stats()
    print(f"Offline model load seconds={load_seconds:.2f}", flush=True)

    # Check all real prompt/name boundaries before spending GPU time on scoring.
    for _, group in frame.groupby("query_sample_id", sort=False):
        suffixes = set()
        for ko, en in group[["vessel_name", "candidate_vessel_name"]].itertuples(index=False, name=None):
            tokens, start = encode_pair(tokenizer, ko, en, args.direction)
            if args.direction == "reverse":
                suffixes.add(tuple(tokens[start:]))
        if args.direction == "reverse":
            assert len(suffixes) == 1, "candidate contexts changed the observed Korean tokens"
    print("PASS: 실제 후보 전체의 토큰 경계 검사", flush=True)
    batch_score = partial(score_batch, model, tokenizer, direction=args.direction, reduction=reduction)
    lengths = frame.vessel_name.str.len() + frame.candidate_vessel_name.str.len()
    probe_id = frame.loc[lengths.idxmax(), "query_sample_id"]
    probe = frame[frame.query_sample_id == probe_id].sort_values("rank").to_dict("records")
    while True:
        try:
            batch_score(probe[:batch_size])
            break
        except torch.cuda.OutOfMemoryError:
            if scores_path.exists() or batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            gc.collect()
            torch.cuda.empty_cache()
            print(f"사전 점검 메모리에 맞춰 batch_size={batch_size}", flush=True)

    def fixed(records):
        return np.concatenate([batch_score(records[i:i+batch_size])
                               for i in range(0, len(records), batch_size)])
    first = fixed(probe)
    other_id = next(q for q in frame.query_sample_id.unique() if q != probe_id)
    other = frame[frame.query_sample_id == other_id].sort_values("rank").to_dict("records")
    fixed(other)
    np.testing.assert_array_equal(first, fixed(probe))
    singles = np.array([batch_score([r])[0] for r in probe[:3]])
    together = batch_score(probe[:3])
    lengths = [len(ids)-start for ids, start in [encode_pair(tokenizer, r["vessel_name"],
               r["candidate_vessel_name"], args.direction) for r in probe[:3]]]
    differences = np.abs(singles-together) / (np.array(lengths) if reduction == "sum" else 1)
    assert np.max(differences) < 0.02
    print(f"PASS: 실제 Qwen GPU 점수·고정 배치 반복 일치; batch_size={batch_size}", flush=True)

    scored = score_candidates(frame, model, tokenizer, args.model, scores_path,
                              batch_size=batch_size, checkpoint_queries=25,
                              load_seconds=load_seconds, existing_ranker_seconds=500.0,
                              deployment_pairs=30000, direction=args.direction, reduction=reduction)
    if args.direction == "reverse":
        evaluate_grid(scored, args.work / f"grid_{tag}.csv", eval_path,
                      expected_candidates=frame, selection="mixed")
        print("역방향 7B 1회 실행 완료. 추가 모델/프롬프트 재시도는 하지 않습니다.", flush=True)
        return
    result = evaluate(scored, output_path=eval_path, expected_candidates=frame)
    print("완료:", scores_path, eval_path, sep="\n", flush=True)
    if not result["accuracy_gate_pass"]:
        print("3B 기준 미달: MODEL_ID만 7B로 바꿔 한 번 재실행하세요." if "3B" in args.model
              else "7B도 기준 미달: Track B를 종료합니다.", flush=True)


if __name__ == "__main__":
    main()
