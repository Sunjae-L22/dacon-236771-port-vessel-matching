# author: terra
"""Natural candidate export, ByT5 scoring, tune-only fusion, and holdout report."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

from runtime_common import sequence_scores, sha, verify_package, write_json


def baseline_candidates(root, output, allow_train=False):
    root, output = Path(root), Path(output)
    os.environ["KO_DICT"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    code = root / "shared/experiments/pinned/build31"
    sys.path.insert(0, str(code))
    import lightgbm as lgb
    import opus_lib as L
    from opus_cv import PARAMS
    verify_package(root)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / "config.json").read_text())
    train = pd.read_csv(root / "data/open/train.csv")
    catalog = pd.read_csv(root / "data/open/vessel_catalog.csv")
    split = pd.read_csv(root / "terra/byt5_pilot/split_v1/rows.csv")
    roles = split.set_index("sample_id").role
    assert train.sample_id.map(roles).notna().all()
    tr = train[train.sample_id.map(roles).eq("train")].copy()
    ix = L.Index(catalog, tr)
    llm_csv = root / "shared/experiments/byt5_baseline/llm_scored_trainrole.csv"
    gen = pd.read_csv(llm_csv)
    assert set(gen.vessel_id) <= set(tr.target_vessel_id)
    llm = L.llm_korean_rows(tr, llm_csv)
    ix.ko = L.KA.KoMatcher(catalog, pd.concat([tr, llm], ignore_index=True))
    model_path = output / "ranker.txt"
    receipt_path = output / "baseline_manifest.json"
    signature = {"split": sha(root / "terra/byt5_pilot/split_v1/rows.csv"),
                 "opus_lib": sha(code / "opus_lib.py"), "matcher": sha(code / "ko_translit_argon.py"),
                 "llm": sha(llm_csv), "train": sha(root / "data/open/train.csv"),
                 "recipe": "build31 KO_DICT=1 seed0 rounds600 role=train only"}
    if model_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt["signature"] == signature
        assert receipt["model_sha256"] == sha(model_path)
    else:
        packaged = root / "shared/experiments/byt5_baseline"
        model_origin = "colab_single_seed_rebuild"
        if (packaged / "ranker.txt").exists() and (packaged / "manifest.json").exists():
            original = json.loads((packaged / "manifest.json").read_text())
            h = original["sha256"]
            assert h["split rows.csv"] == signature["split"]
            assert h["opus_lib.py"] == signature["opus_lib"] and h["ko_translit_argon.py"] == signature["matcher"]
            assert h["llm_scored_trainrole.csv"] == signature["llm"]
            assert h["ranker.txt"] == sha(packaged / "ranker.txt")
            shutil.copyfile(packaged / "ranker.txt", model_path)
            model_origin = "verified_opus_model_import"
        else:
            assert allow_train, "기준선이 아직 없습니다. 준비된 기준선 ZIP을 추가하거나 코랩 기준선 학습을 켜세요."
            print("코랩 CPU에서 단일 시드 기준 랭커를 만듭니다. Mac 빌드는 건드리지 않습니다.", flush=True)
            x, y, groups, _ = L.build(ix, pd.concat([L.augment_korean(tr, 0), L.augment_corrupt(tr, 0),
                         L.augment_rename(tr, 0), llm], ignore_index=True), True)
            params = dict(PARAMS, num_threads=2)
            m = lgb.train(params, lgb.Dataset(x, y, group=groups, feature_name=L.FEATURES), 600)
            m.save_model(str(model_path))
            del x, y, groups, m
        write_json(receipt_path, {"status": "BASELINE_TRAINED_OR_VERIFIED_IMPORT", "signature": signature,
                                 "model_sha256": sha(model_path), "origin": model_origin,
                                 "colab_rebuild_num_threads": 2 if model_origin.startswith("colab") else None})
    model = lgb.Booster(model_file=str(model_path))
    queries = pd.read_csv(root / "prepared/queries.csv")
    train_targets = set(tr.target_vessel_id)
    rows = []
    t0 = time.time()
    for qi, q in enumerate(queries.to_dict("records")):
        # Ground truth is kept for reporting but not passed to candidate/feature code.
        features_q = {k: v for k, v in q.items() if k not in ("target_vessel_id", "role", "variant")}
        idx, features = L.query_features(ix, features_q, L.query_candidates(ix, features_q))
        scores = model.predict(features, num_threads=2)
        ordered = sorted(range(len(idx)), key=lambda j: (-scores[j], idx[j]))[:config["top_k"]]
        assert ordered, f"No natural candidates: {q['query_sample_id']}"
        for rank, j in enumerate(ordered, 1):
            candidate = ix.ids[idx[j]]
            rows.append({"query_sample_id": q["query_sample_id"], "orig_sample_id": q["orig_sample_id"],
                "role": q["role"], "variant": q["variant"], "vessel_name": q["vessel_name"],
                "target_vessel_id": q["target_vessel_id"], "candidate_vessel_id": candidate,
                "candidate_name": ix.names_raw[idx[j]], "rank": rank, "ranker_score": float(scores[j]),
                "is_target": candidate == q["target_vessel_id"], "cand_is_train_role_target": candidate in train_targets})
        if (qi + 1) % 100 == 0:
            print(f"candidate queries {qi+1}/{len(queries)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "candidates.csv", index=False)
    write_json(output / "candidate_manifest.json", {"queries": len(queries), "pairs": len(frame),
        "seconds": time.time()-t0, "model_sha256": sha(model_path),
        "queries_sha256": sha(root / "prepared/queries.csv"), "candidates_sha256": sha(output / "candidates.csv"),
        "forced_target_insertion": False, "train_ship_exclusion": False})
    print(f"후보 저장 완료: {len(frame):,}쌍", flush=True)


def score(root, run, baseline_output):
    import torch
    from transformers import AutoModelForSeq2SeqLM, ByT5Tokenizer
    root, run, baseline_output = Path(root), Path(run), Path(baseline_output)
    verify_package(root)
    cfg = json.loads((root / "config.json").read_text())
    done = json.loads((run / "training_complete.json").read_text())
    assert sha(run / "best/model.safetensors") == done["best_weights_sha256"]
    assert torch.cuda.is_available(), "GPU 런타임이 필요합니다."
    data = pd.read_csv(baseline_output / "candidates.csv")
    expected = pd.read_csv(root / "prepared/queries.csv")
    assert set(data.query_sample_id) == set(expected.query_sample_id)
    assert not data.duplicated(["query_sample_id", "candidate_vessel_id"]).any()
    t0 = time.time()
    tokenizer = ByT5Tokenizer.from_pretrained(run / "best", local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(run / "best", local_files_only=True).to("cuda")
    model.eval(); torch.cuda.synchronize()
    load_secs = time.time() - t0
    unique = data[["candidate_name", "vessel_name"]].drop_duplicates().reset_index(drop=True)
    # Cache only identical input pairs; no frequencies/distributions are model features.
    scores = sequence_scores(model, tokenizer, unique.candidate_name.tolist(), unique.vessel_name.tolist(),
                             "cuda", cfg["score_batch_size"], cfg)
    unique["byt5_score"] = scores
    joined = data.merge(unique, on=["candidate_name", "vessel_name"], validate="many_to_one", how="left")
    assert joined.byt5_score.notna().all()
    joined.to_csv(run / "scored_candidates.csv", index=False)
    # Verify batching/order stability on a small deterministic probe, in actual precision.
    probe = unique.sample(min(16, len(unique)), random_state=cfg["seed"])
    together = sequence_scores(model, tokenizer, probe.candidate_name.tolist(), probe.vessel_name.tolist(), "cuda", 8, cfg)
    separately = sequence_scores(model, tokenizer, probe.candidate_name.tolist(), probe.vessel_name.tolist(), "cuda", 1, cfg)
    max_diff = float(np.max(np.abs(together-separately)))
    bench = unique.sample(min(1024, len(unique)), random_state=1)
    sequence_scores(model, tokenizer, bench.candidate_name.head(8).tolist(), bench.vessel_name.head(8).tolist(), "cuda", 8, cfg)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); bt = time.time()
    sequence_scores(model, tokenizer, bench.candidate_name.tolist(), bench.vessel_name.tolist(), "cuda", cfg["score_batch_size"], cfg)
    torch.cuda.synchronize(); elapsed = time.time() - bt
    speed = len(bench) / elapsed
    write_json(run / "scoring_manifest.json", {"status": "GPU_SCORED_PILOT_ONLY", "gpu": torch.cuda.get_device_name(),
        "pairs": len(data), "unique_pairs": len(unique), "model_load_seconds": load_secs,
        "seconds_total": time.time()-t0, "benchmark_pairs": len(bench), "benchmark_seconds": elapsed,
        "pairs_per_second": speed, "projection_60000_pairs_seconds": 60000/speed,
        "projection_is_not_measured_full_server_runtime": True,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(), "batch_probe_max_abs_difference": max_diff,
        "batch_probe_tolerance": 0.02, "batch_probe_pass": max_diff <= .02,
        "scored_candidates_sha256": sha(run / "scored_candidates.csv"),
        "candidate_manifest_sha256": sha(baseline_output / "candidate_manifest.json"),
        "best_weights_sha256": done["best_weights_sha256"]})
    print(f"조건부 점수 저장 완료, {speed:.1f}쌍/초 (실측 GPU: {torch.cuda.get_device_name()})", flush=True)


def zscore(v):
    return (v-v.mean()) / max(float(v.std(ddof=0)), 1e-6)


def predictions(data, alpha):
    d = data.copy()
    d["combined"] = (1-alpha)*d.base_z + alpha*d.neural_z
    return d.sort_values(["query_sample_id", "combined", "rank", "candidate_vessel_id"],
                         ascending=[True, False, True, True]).groupby("query_sample_id").head(1)


def report(run):
    run = Path(run)
    score_receipt = json.loads((run / "scoring_manifest.json").read_text())
    assert sha(run / "scored_candidates.csv") == score_receipt["scored_candidates_sha256"]
    d = pd.read_csv(run / "scored_candidates.csv")
    # Within-query normalization uses no other query/test row information.
    d["base_z"] = d.groupby("query_sample_id").ranker_score.transform(zscore)
    d["neural_z"] = d.groupby("query_sample_id").byt5_score.transform(zscore)
    tune = d[d.role.eq("tune")]
    assert set(d.role) == {"tune", "eval"}
    assert not set(tune.target_vessel_id) & set(d.loc[d.role.eq("eval"), "target_vessel_id"])
    baseline = predictions(tune, 0)
    baseline_orig = baseline[baseline.variant.eq("orig")].is_target.mean()
    trials = []
    for alpha in [0., .05, .1, .2, .35, .5, .75, 1.]:
        p = predictions(tune, alpha)
        macro = p.groupby(["variant", "target_vessel_id"]).is_target.mean().groupby("variant").mean().mean()
        orig = p[p.variant.eq("orig")].is_target.mean()
        trials.append({"alpha": alpha, "tune_scenario_ship_macro": float(macro), "tune_original_acc": float(orig),
                       "eligible": bool(orig >= baseline_orig-1e-12)})
    eligible = [x for x in trials if x["eligible"]]
    selected = sorted(eligible, key=lambda x: (-x["tune_scenario_ship_macro"], x["alpha"]))[0]
    # Alpha is fixed above without looking at evaluation labels.
    alpha = selected["alpha"]
    tune_outputs, final_outputs = [], []
    for a, label in [(0., "baseline"), (1., "neural"), (alpha, "fused")]:
        p = predictions(d, a).assign(method=label, selected_alpha=a)
        final_outputs.append(p)
    p = pd.concat(final_outputs, ignore_index=True)
    p.to_csv(run / "predictions.csv", index=False)
    metrics = []
    for (role, variant, method), part in p.groupby(["role", "variant", "method"]):
        recall = d[(d.role==role)&(d.variant==variant)].groupby("query_sample_id").is_target.any().mean()
        metrics.append({"role": role, "variant": variant, "method": method, "rows": len(part),
                        "independent_vessels": part.target_vessel_id.nunique(), "row_accuracy": float(part.is_target.mean()),
                        "ship_macro_accuracy": float(part.groupby("target_vessel_id").is_target.mean().mean()),
                        "candidate_recall_at20": float(recall)})
    m = pd.DataFrame(metrics); m.to_csv(run / "metrics.csv", index=False)
    pivot = p[p.role.eq("eval")].groupby(["target_vessel_id", "method"]).is_target.mean().unstack()
    delta = (pivot.fused-pivot.baseline).to_numpy()
    rng = np.random.default_rng(42)
    boot = delta[rng.integers(0, len(delta), size=(2000,len(delta)))].mean(1)
    result = {"status": "PILOT_COMPLETE_REVIEW_REQUIRED_NOT_FULL_CV_NOT_SUBMISSION", "selected_alpha": alpha,
              "alpha_selected_using": "tune only, original accuracy must not regress, ties prefer smaller alpha",
              "tune_trials": trials, "eval_independent_vessels": len(pivot),
              "eval_ship_macro_delta_averaging_scenarios": float(delta.mean()),
              "eval_paired_vessel_bootstrap_95": np.quantile(boot,[.025,.975]).tolist(),
              "scoring_batch_probe_pass": score_receipt["batch_probe_pass"],
              "score_manifest_sha256": sha(run / "scoring_manifest.json"), "metrics_sha256": sha(run / "metrics.csv"),
              "limitations": ["three-way holdout, not five-fold OOF", "only Korean rows evaluated",
                              "synthetic stress mix does not estimate hidden-test distribution",
                              "small 52-vessel evaluation; no automatic adoption or expansion",
                              "L4 server runtime still needs actual full-pipeline check"]}
    write_json(run / "pilot_report.json", result)
    print(m[m.role.eq("eval")].to_string(index=False), flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=["baseline", "score", "report"])
    p.add_argument("--root", type=Path, default=Path(__file__).parent)
    p.add_argument("--run", type=Path, required=True); p.add_argument("--baseline", type=Path)
    p.add_argument("--allow-baseline-training", action="store_true")
    a = p.parse_args(); baseline = a.baseline or a.run / "baseline"
    if a.stage == "baseline": baseline_candidates(a.root, baseline, a.allow_baseline_training)
    elif a.stage == "score": score(a.root, a.run, baseline)
    else: report(a.run)
