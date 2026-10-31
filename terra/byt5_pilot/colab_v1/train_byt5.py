# author: terra
"""GPU training, with tune-only checkpoint selection and resumable Drive state."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import time

import numpy as np
import pandas as pd

from runtime_common import make_batch, per_example_nll, precision_context, sequence_scores, sha, verify_package, write_json


def train(root, output, steps=None, batch_size=None):
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForSeq2SeqLM, ByT5Tokenizer, Adafactor, get_linear_schedule_with_warmup
    root, output = Path(root), Path(output)
    manifest = verify_package(root)
    cfg = json.loads((root / "config.json").read_text())
    if steps is not None:
        cfg["steps"] = steps
    if batch_size is not None:
        cfg["batch_size"] = batch_size
    assert torch.cuda.is_available(), "GPU 런타임을 선택한 뒤 다시 실행하세요."
    device = "cuda"
    if not torch.cuda.is_bf16_supported() and batch_size is None:
        cfg["batch_size"] = 1
        cfg["gradient_accumulation"] = 16
    output.mkdir(parents=True, exist_ok=True)
    binding = hashlib.sha256((json.dumps(cfg, sort_keys=True) + sha(root / "package_manifest.json")).encode()).hexdigest()
    lock = output / "run_binding.json"
    if lock.exists():
        assert json.loads(lock.read_text())["binding"] == binding, "설정/입력이 달라졌습니다. 새 RUN_NAME을 쓰세요."
    else:
        write_json(lock, {"binding": binding, "config": cfg})
    if (output / "training_complete.json").exists():
        done = json.loads((output / "training_complete.json").read_text())
        assert done["binding"] == binding
        assert sha(output / "best" / "model.safetensors") == done["best_weights_sha256"]
        print("이미 완료된 학습을 재사용합니다.", flush=True)
        return
    write_json(output / "training_status.json", {"status": "RUNNING", "binding": binding})
    try:
        origin_file = output / "model_origin.json"
        if origin_file.exists():
            origin = json.loads(origin_file.read_text())
        else:
            revision = HfApi().model_info(cfg["model_id"]).sha
            origin = {"model_id": cfg["model_id"], "revision": revision}
            write_json(origin_file, origin)
        tokenizer = ByT5Tokenizer.from_pretrained(origin["model_id"], revision=origin["revision"])
        t0 = time.time()
        model = AutoModelForSeq2SeqLM.from_pretrained(origin["model_id"], revision=origin["revision"])
        model.to(device)
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        optimizer = Adafactor(model.parameters(), lr=cfg["learning_rate"], relative_step=False,
                             scale_parameter=False, warmup_init=False, clip_threshold=1.0)
        scheduler = get_linear_schedule_with_warmup(optimizer, cfg["warmup_steps"], cfg["steps"])
        random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        pairs = pd.read_csv(root / "prepared/train_pairs.csv")
        tune = pd.read_csv(root / "prepared/tune_pairs.csv")
        assert set(pairs.role) == {"train"} and set(tune.role) == {"tune"}
        assert not set(pairs.vessel_id) & set(tune.vessel_id)
        weights = torch.tensor(pairs.sample_weight.to_numpy(), dtype=torch.double)
        start, best_loss, best_step = 0, float("inf"), 0
        state_path = output / "last.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            assert state["binding"] == binding
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            for st in optimizer.state.values():
                for key, value in st.items():
                    if torch.is_tensor(value):
                        st[key] = value.to(device)
            scheduler.load_state_dict(state["scheduler"])
            start, best_loss, best_step = state["step"], state["best_loss"], state["best_step"]
            torch.set_rng_state(state["rng_cpu"])
            torch.cuda.set_rng_state_all(state["rng_cuda"])
            print(f"checkpoint step {start}부터 이어갑니다.", flush=True)

        def save_state(step):
            # Write a complete file before replacing the previous valid checkpoint.
            local = Path("/content/byt5_last.tmp.pt") if Path("/content").is_dir() else output / "last.local.pt"
            torch.save({"binding": binding, "step": step, "best_loss": best_loss, "best_step": best_step,
                        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "rng_cpu": torch.get_rng_state(),
                        "rng_cuda": torch.cuda.get_rng_state_all()}, local)
            dest = output / "last.pt.tmp"
            shutil.copyfile(local, dest)
            os.replace(dest, state_path)
            local.unlink(missing_ok=True)

        for step in range(start, cfg["steps"]):
            model.train(); optimizer.zero_grad(set_to_none=True)
            generator = torch.Generator().manual_seed(cfg["seed"] + step)
            total_batch = cfg["batch_size"] * cfg["gradient_accumulation"]
            indices = torch.multinomial(weights, total_batch, replacement=True, generator=generator).tolist()
            avg = 0.
            for j in range(cfg["gradient_accumulation"]):
                bdf = pairs.iloc[indices[j*cfg["batch_size"]:(j+1)*cfg["batch_size"]]]
                batch = make_batch(tokenizer, bdf.english.tolist(), bdf.korean.tolist(), device,
                                   cfg["max_source"], cfg["max_target"])
                with precision_context(device):
                    loss = per_example_nll(model, batch).mean() / cfg["gradient_accumulation"]
                assert torch.isfinite(loss), "Nonfinite training loss"
                loss.backward(); avg += float(loss.detach())
            optimizer.step(); scheduler.step()
            if (step + 1) % 20 == 0:
                print(f"step {step+1}/{cfg['steps']} loss={avg:.4f} elapsed={(time.time()-t0)/60:.1f}m", flush=True)
            if (step + 1) % cfg["eval_every"] == 0 or step + 1 == cfg["steps"]:
                scores = sequence_scores(model, tokenizer, tune.english.tolist(), tune.korean.tolist(),
                                         device, cfg["score_batch_size"], cfg)
                # Each tune vessel has equal checkpoint-selection mass.
                tune_loss = float(pd.DataFrame({"vessel": tune.vessel_id, "nll": -scores}).groupby("vessel").nll.mean().mean())
                if tune_loss < best_loss:
                    best_loss, best_step = tune_loss, step + 1
                    best = output / "best"
                    best.mkdir(exist_ok=True)
                    # A disconnect during Drive copy must not truncate the previous weights.
                    with tempfile.TemporaryDirectory(prefix="byt5_best_") as temp:
                        model.save_pretrained(temp, safe_serialization=True)
                        tokenizer.save_pretrained(temp)
                        for src in Path(temp).iterdir():
                            if src.is_file():
                                pending = best / (src.name + ".tmp")
                                shutil.copyfile(src, pending)
                                os.replace(pending, best / src.name)
                    write_json(best / "selection.json", {"binding": binding, "step": best_step, "tune_nll": best_loss})
                save_state(step + 1)
                entry = {"step": step+1, "train_loss": avg, "tune_ship_macro_nll": tune_loss,
                         "best_step": best_step, "seconds": time.time()-t0}
                with (output / "training_log.jsonl").open("a") as f:
                    f.write(json.dumps(entry) + "\n")
                print(entry, flush=True)
        done = {"status": "GPU_TRAINING_COMPLETE_NOT_SUBMISSION", "binding": binding,
                "steps": cfg["steps"], "best_step": best_step, "best_tune_nll": best_loss,
                "seconds_this_session": time.time()-t0, "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__, "model_origin": origin,
                "best_weights_sha256": sha(output / "best" / "model.safetensors"),
                "split_sha256": manifest["files_sha256"]["terra/byt5_pilot/split_v1/rows.csv"]}
        write_json(output / "training_complete.json", done)
        write_json(output / "training_status.json", done)
        print("학습 결과를 저장했습니다. 다음 후보 점수화 셀을 실행하세요.", flush=True)
    except BaseException as exc:
        write_json(output / "training_status.json", {"status": "FAILED_OR_INTERRUPTED", "binding": binding,
                                                     "error": repr(exc)})
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True); p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int); p.add_argument("--batch-size", type=int)
    a = p.parse_args(); train(a.root, a.output, a.steps, a.batch_size)
