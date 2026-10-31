# author: terra
"""Shared, public-train-only ByT5 pilot utilities."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def text(s):
    return " ".join(unicodedata.normalize("NFKC", str(s)).split())


def verify_package(root):
    root = Path(root)
    manifest = json.loads((root / "package_manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        assert sha(root / name) == expected, f"Package hash mismatch: {name}"
    return manifest


def make_batch(tokenizer, english, korean, device, max_source=128, max_target=192):
    """No silent truncation. Labels include EOS, exclude padding from loss."""
    import torch
    sources = ["en2ko: " + text(x) for x in english]
    targets = [text(x) for x in korean]
    for s in sources:
        assert len(tokenizer(s).input_ids) <= max_source, f"Source exceeds {max_source}: {s}"
    for s in targets:
        assert len(tokenizer(s).input_ids) <= max_target, f"Target exceeds {max_target}: {s}"
    x = tokenizer(sources, padding=True, return_tensors="pt")
    y = tokenizer(targets, padding=True, return_tensors="pt").input_ids
    y[y == tokenizer.pad_token_id] = -100
    return {"input_ids": x.input_ids.to(device), "attention_mask": x.attention_mask.to(device),
            "labels": y.to(device)}


def per_example_nll(model, batch):
    import torch.nn.functional as F
    out = model(**batch, use_cache=False)
    labels = batch["labels"]
    token_loss = F.cross_entropy(out.logits.float().transpose(1, 2), labels,
                                 reduction="none", ignore_index=-100)
    mask = labels.ne(-100)
    return (token_loss * mask).sum(1) / mask.sum(1).clamp_min(1)


def precision_context(device):
    import torch
    if str(device).startswith("cuda") and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()  # T4 uses float32; no fragile T5 fp16 path


def sequence_scores(model, tokenizer, english, korean, device, batch_size=8, config=None):
    import torch
    config = config or {}
    model.eval()
    result = []
    with torch.inference_mode():
        for i in range(0, len(english), batch_size):
            b = make_batch(tokenizer, english[i:i+batch_size], korean[i:i+batch_size], device,
                           config.get("max_source", 128), config.get("max_target", 192))
            with precision_context(device):
                loss = per_example_nll(model, b)
            assert torch.isfinite(loss).all(), "Nonfinite conditional score"
            result.extend((-loss).float().cpu().tolist())
    return np.asarray(result)
