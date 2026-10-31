# author: opus
"""LLM judge for Korean port records: which of K English catalog candidates is the same vessel?

One forward pass per ordering; the answer is read from the next-token logits over the option digits
"1".."K" (no sampling, no generation). Two orderings (ranker order and reversed) are averaged in
log-prob space to cancel position bias. Used both by the Colab evaluation notebook and by the
submission script (one row at a time, so a row's result never depends on other rows).
"""
from __future__ import annotations

import math

import numpy as np
import torch

INSTRUCTION = (
    "한국 항만 입출항 기록의 한글 선박명은 영문 선박명을 소리 나는 대로 적은 것입니다. "
    "한글 표기에는 오타, 받침·모음 혼동, 띄어쓰기 차이, 일부 단어 누락이 있을 수 있고 "
    "톤수도 기록마다 조금 다를 수 있습니다. 국적과 선종은 한국어로, 카탈로그는 영어로 적혀 있습니다.\n"
    "아래 기록과 같은 선박인 카탈로그 후보의 번호 하나만 답하세요."
)


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    return "-" if math.isnan(v) else f"{v:,.0f}"


def _txt(v):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)


def build_messages(q, cands):
    lines = [INSTRUCTION, "", "[기록]",
             f"선박명: {_txt(q.get('vessel_name'))}",
             f"총톤수: {_num(q.get('gross_tonnage'))} / 국제총톤수: {_num(q.get('international_gross_tonnage'))}",
             f"국적: {_txt(q.get('flag_name'))} / 선종: {_txt(q.get('vessel_type'))}",
             "", "[카탈로그 후보]"]
    for n, c in enumerate(cands, 1):
        lines.append(f"{n}. {_txt(c.get('vessel_name'))} | {_txt(c.get('vessel_type'))} | {_txt(c.get('flag_name'))}"
                     f" | GT {_num(c.get('gross_tonnage'))} | DWT {_num(c.get('deadweight_tonnage'))}")
    return [{"role": "user", "content": "\n".join(lines)}]


class Judge:
    def __init__(self, model_dir, dtype=torch.bfloat16, device_map="cuda", quant=None, max_k=9):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = dict(dtype=dtype, device_map=device_map)
        if quant is not None:
            kw["quantization_config"] = quant
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, **kw).eval()
        self.digit_ids = []
        for n in range(1, max_k + 1):
            ids = self.tok.encode(str(n), add_special_tokens=False)
            assert len(ids) == 1, (n, ids)
            self.digit_ids.append(ids[0])

    def _prompt(self, q, cands):
        msgs = build_messages(q, cands)
        kw = dict(tokenize=False, add_generation_prompt=True)
        try:
            text = self.tok.apply_chat_template(msgs, enable_thinking=False, **kw)
        except TypeError:
            text = self.tok.apply_chat_template(msgs, **kw)
        return text

    @torch.no_grad()
    def _logprobs(self, prompts, k):
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        logits = self.model(**enc).logits[:, -1, :].float()
        sel = logits[:, self.digit_ids[:k]]
        return torch.log_softmax(sel, dim=-1).cpu().numpy()

    def score_many(self, items, batch=8):
        """items: list of (query, cands). Returns a list of arrays: per-candidate log-prob (orderings averaged)."""
        jobs = []
        for qi, (q, cands) in enumerate(items):
            k = len(cands)
            fwd = list(range(k))
            for perm in (fwd, fwd[::-1]):
                jobs.append((qi, perm, self._prompt(q, [cands[j] for j in perm]), k))
        out = [np.zeros(len(c)) for _, c in items]
        jobs_by_k = {}
        for j in jobs:
            jobs_by_k.setdefault(j[3], []).append(j)
        for k, js in jobs_by_k.items():
            for s in range(0, len(js), batch):
                chunk = js[s:s + batch]
                lp = self._logprobs([c[2] for c in chunk], k)
                for (qi, perm, _, _), row in zip(chunk, lp):
                    for pos, j in enumerate(perm):
                        out[qi][j] += row[pos] / 2.0
        return out

    def score(self, q, cands):
        """Single row, both orderings in one batch of two: identical work whatever else is in the test file."""
        return self.score_many([(q, cands)], batch=2)[0]
