# author: opus
"""Can a public multilingual embedding model match Korean port-record names to English catalog names?

Index: all 78,595 catalog English names. Queries: the real Korean names of train's Korean rows
(one row per vessel), clean and corrupted with opus_lib._corrupt_name (seeded). Reports top-1 /
top-20 retrieval by cosine similarity, name only and restricted to catalog GT within +-10% of the
record's tonnage. Pretrained weights only; nothing is trained here.

usage: python3 opus/stage2/embed_probe.py intfloat/multilingual-e5-small [more models]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared/src"))
import opus_lib as L  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
PREFIX = {"intfloat": ("query: ", "query: ")}


def encode(tok, model, texts, bs=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=32, return_tensors="pt").to(DEV)
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
            out.append(torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy())
    return np.concatenate(out)


def main(names):
    cat = pd.read_csv(ROOT / "data/open/vessel_catalog.csv", encoding="utf-8-sig")
    tr = pd.read_csv(ROOT / "data/open/train.csv", encoding="utf-8-sig")
    ko = tr[tr.vessel_name.map(L.is_hangul)].drop_duplicates("target_vessel_id")
    id2i = {v: i for i, v in enumerate(cat.vessel_id)}
    tgt = ko.target_vessel_id.map(id2i).to_numpy()
    q_gt = ko.gross_tonnage.fillna(ko.international_gross_tonnage).to_numpy(float)
    c_gt = cat.gross_tonnage.to_numpy(float)
    rng = np.random.default_rng(0)
    variants = {"clean": ko.vessel_name.tolist(),
                "corrupt": [L._corrupt_name(str(s), rng) for s in ko.vessel_name]}
    for name in names:
        t = time.time()
        tok, model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval().to(DEV)
        pq, pc = PREFIX.get(name.split("/")[0], ("", ""))
        C = encode(tok, model, [pc + str(s) for s in cat.vessel_name])
        print(f"{name}: catalog encoded in {time.time() - t:.0f}s")
        for v, qs in variants.items():
            Q = encode(tok, model, [pq + s for s in qs])
            S = Q @ C.T
            top = np.argsort(-S, axis=1)[:, :20]
            r1 = (top[:, 0] == tgt).mean()
            r20 = np.mean([t in row for t, row in zip(tgt, top)])
            ok = np.abs(c_gt[None, :] / q_gt[:, None] - 1) <= 0.10
            St = np.where(ok, S, -9)
            topt = np.argsort(-St, axis=1)[:, :20]
            t1 = (topt[:, 0] == tgt).mean()
            t20 = np.mean([t in row for t, row in zip(tgt, topt)])
            print(f"  {v:8s} n={len(qs)}  name top1 {r1:.3f} top20 {r20:.3f} | +GT10% top1 {t1:.3f} top20 {t20:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
