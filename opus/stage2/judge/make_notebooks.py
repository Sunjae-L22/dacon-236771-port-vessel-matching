# author: opus
"""Writes the Colab notebooks for the LLM judge into the Drive llm_judge folder.

usage: python3 make_notebooks.py DRIVE_LLM_JUDGE_DIR
"""
import json
import sys
from pathlib import Path

DRIVE = "/content/drive/MyDrive/Projects/항만문제해결을위한AI기술공모전/llm_judge"


def nb(cells):
    out = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return {"cells": out, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm"},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 0}


EVAL = [
    ("markdown", """
# 1단계: LLM 판정기 모델 비교 (항만 한글 행)
런타임 → 런타임 유형 변경 → **G4 / A100 / H100 중 가장 큰 GPU** 선택 후 위에서부터 순서대로 실행하세요.
결과는 드라이브 `llm_judge/results/`에 저장되고, Claude가 그 파일을 읽어 판정합니다.
"""),
    ("code", f"""
from google.colab import drive
drive.mount('/content/drive')
import os, sys
D = "{DRIVE}"
os.chdir(D); sys.path.insert(0, D)
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
    ("code", """
!pip -q install transformers==4.57.6 accelerate==1.9.0 bitsandbytes==0.46.1
"""),
    ("markdown", """
Gemma도 비교하려면 아래 칸에 Hugging Face 토큰을 넣으세요(google/gemma-3-12b-it 약관 동의 필요). 없으면 비워두면 Gemma만 건너뜁니다.
"""),
    ("code", """
HF_TOKEN = ""   # 선택
if HF_TOKEN:
    from huggingface_hub import login; login(HF_TOKEN)
MODELS = ["Qwen/Qwen3-14B", "Qwen/Qwen3-8B", "kakaocorp/kanana-1.5-8b-instruct-2505", "Qwen/Qwen2.5-14B-Instruct"]
if HF_TOKEN:
    MODELS.append("google/gemma-3-12b-it")
"""),
    ("code", """
import json, time, gc, torch, importlib
import judge_llm; importlib.reload(judge_llm)
K = 8
packets = [json.loads(l) for l in open(f"{D}/packets/eval_packets.jsonl")]
items = [(p["query"], p["cands"][:K]) for p in packets]
print(len(packets), "packets")
for name in MODELS:
    out = f"{D}/results/{name.replace('/', '__')}.jsonl"
    if os.path.exists(out):
        print("skip (done)", name); continue
    t = time.time()
    try:
        j = judge_llm.Judge(name)
    except Exception as e:
        print("LOAD FAIL", name, repr(e)[:300]); continue
    lp = j.score_many(items, batch=16)
    with open(out, "w") as f:
        for p, v in zip(packets, lp):
            f.write(json.dumps({"pid": p["pid"], "lp": [round(float(x), 5) for x in v]}) + "\\n")
    print(name, f"{time.time()-t:.0f}s")
    del j; gc.collect(); torch.cuda.empty_cache()
    import shutil; shutil.rmtree(os.path.expanduser("~/.cache/huggingface/hub/models--" + name.replace("/", "--")), ignore_errors=True)
"""),
    ("code", """
# 요약: 어려운 행(gap<4)에서 랭커 1위 vs LLM 선택 정확도 (조건별)
import numpy as np, collections, glob
for fn in sorted(glob.glob(f"{D}/results/*.jsonl")):
    res = {json.loads(l)["pid"]: json.loads(l)["lp"] for l in open(fn)}
    tab = collections.defaultdict(lambda: [0, 0, 0])
    for p in packets:
        if p["pid"] not in res: continue
        c = p["cands"][:K]; t = p["target"]
        key = (p["cond"], "hard" if p["gap"] < 4 else "easy")
        tab[key][0] += 1
        tab[key][1] += c[0]["i"] == t
        tab[key][2] += c[int(np.argmax(res[p["pid"]]))]["i"] == t
    print(os.path.basename(fn))
    for k in sorted(tab):
        n, a, b = tab[k]; print(f"   {k[0]:14s} {k[1]:4s} n={n:4d}  ranker {a/n:.3f}  LLM {b/n:.3f}  ({b-a:+d})")
"""),
]


BUILD = [
    ("markdown", """
# 2단계: 제출 ZIP 만들기 (선택된 모델 4비트 양자화 → 93 + LLM 판정기)
가능하면 런타임을 **L4**로(평가 서버와 같은 GPU) 선택하세요. 아니면 아무 GPU나 괜찮습니다.
아래 설정 칸은 Claude가 1단계 결과를 보고 채워 둡니다. 순서대로 실행하면 `llm_judge/submit/`에 ZIP이 생깁니다.
"""),
    ("code", f"""
from google.colab import drive
drive.mount('/content/drive')
import os, sys
D = "{DRIVE}"
sys.path.insert(0, D)
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip -q install transformers==4.57.6 accelerate==1.9.0 bitsandbytes==0.46.1
"""),
    ("code", """
# ---- 설정 (Claude가 채움) ----
NUM = 110
MODEL = "Qwen/Qwen3-14B"
QUANT = "nf4"          # "nf4" 또는 "none"
CONFIG = {"gap": 4.0, "k": 8, "beta": 0.0}
"""),
    ("code", """
import torch, shutil, json, time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
B = "/content/build"
shutil.rmtree(B, ignore_errors=True); os.makedirs(f"{B}/model", exist_ok=True)
kw = dict(dtype=torch.bfloat16, device_map="cuda")
if QUANT == "nf4":
    kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
m = AutoModelForCausalLM.from_pretrained(MODEL, **kw)
m.save_pretrained(f"{B}/model/llm", safe_serialization=True)
AutoTokenizer.from_pretrained(MODEL).save_pretrained(f"{B}/model/llm")
del m; torch.cuda.empty_cache()
!du -sh {B}/model/llm
"""),
    ("code", """
# 양자화 모델로 1단계 평가 패킷 재채점 (bf16 대비 손실 확인)
import importlib, judge_llm, numpy as np, collections
importlib.reload(judge_llm)
packets = [json.loads(l) for l in open(f"{D}/packets/eval_packets.jsonl")]
K = CONFIG["k"]
j = judge_llm.Judge(f"{B}/model/llm")
t = time.time()
lp = j.score_many([(p["query"], p["cands"][:K]) for p in packets], batch=16)
print(f"{time.time()-t:.0f}s for {len(packets)} packets")
out = f"{D}/results/{MODEL.replace('/', '__')}__{QUANT}.jsonl"
with open(out, "w") as f:
    for p, v in zip(packets, lp):
        f.write(json.dumps({"pid": p["pid"], "lp": [round(float(x), 5) for x in v]}) + "\\n")
tab = collections.defaultdict(lambda: [0, 0, 0])
for p, v in zip(packets, lp):
    c = p["cands"][:K]; key = (p["cond"], "hard" if p["gap"] < 4 else "easy")
    comb = np.array(v) + CONFIG["beta"] * np.array([x["score"] for x in c])
    tab[key][0] += 1; tab[key][1] += c[0]["i"] == p["target"]; tab[key][2] += c[int(np.argmax(comb))]["i"] == p["target"]
for k in sorted(tab):
    n, a, b = tab[k]; print(f"   {k[0]:14s} {k[1]:4s} n={n:4d}  ranker {a/n:.3f}  judge {b/n:.3f}  ({b-a:+d})")
del j; torch.cuda.empty_cache()
"""),
    ("code", """
# 93 ZIP + 판정기 조립
import zipfile
z = zipfile.ZipFile(f"{D}/base/93_submit.zip")
for n in z.namelist():
    if n != "script.py":
        z.extract(n, B)
shutil.copy(f"{D}/judge_llm.py", f"{B}/model/judge_llm.py")
shutil.copy(f"{D}/script_judge.py", f"{B}/script.py")
json.dump(CONFIG, open(f"{B}/model/judge_config.json", "w"))
open(f"{B}/requirements.txt", "w").write("bitsandbytes==0.46.1\\n" if QUANT == "nf4" else "# preinstalled only\\n")
!ls {B} {B}/model | head -40
"""),
    ("code", """
# 시험 실행: 로컬 3,000행(train 기반, 정답 제외되어 점수는 의미 없음) 전체 → 600행 부분집합 일치 확인, 시간 측정
import subprocess, pandas as pd
def trial(df):
    shutil.rmtree(f"{B}/data", ignore_errors=True); shutil.rmtree(f"{B}/output", ignore_errors=True)
    os.makedirs(f"{B}/data")
    df.to_csv(f"{B}/data/test.csv", index=False, encoding="utf-8-sig")
    shutil.copy(f"{D}/base/vessel_catalog.csv", f"{B}/data/vessel_catalog.csv")
    t = time.time()
    p = subprocess.run([sys.executable, "script.py"], cwd=B, capture_output=True, text=True)
    print(p.stdout[-600:], p.stderr[-3000:] if p.returncode else "")
    assert p.returncode == 0
    return pd.read_csv(f"{B}/output/submission.csv").set_index("sample_id").target_vessel_id, time.time() - t
full_df = pd.read_csv(f"{D}/base/test_local3000.csv", encoding="utf-8-sig")
full, sec = trial(full_df); print(f"full 3000 rows {sec:.0f}s")
sub_df = full_df.sample(frac=0.2, random_state=5)
part, _ = trial(sub_df)
print("row independence mismatches:", int((full.loc[part.index] != part).sum()), "/", len(part))
shutil.rmtree(f"{B}/data"); shutil.rmtree(f"{B}/output")
"""),
    ("code", """
# ZIP 저장 (가중치는 무압축 저장)
os.makedirs(f"{D}/submit", exist_ok=True)
dst = f"{D}/submit/{NUM}_submit.zip"
with zipfile.ZipFile(dst, "w") as w:
    for root, _, files in os.walk(B):
        for fn in files:
            p = os.path.join(root, fn); arc = os.path.relpath(p, B)
            comp = zipfile.ZIP_STORED if fn.endswith(".safetensors") else zipfile.ZIP_DEFLATED
            w.write(p, arc, compress_type=comp)
print(dst, round(os.path.getsize(dst) / 1e9, 2), "GB (limit 10GB)")
"""),
]


def main(dst):
    dst = Path(dst)
    for name, cells in (("1_eval_judge.ipynb", EVAL), ("2_build_submit.ipynb", BUILD)):
        (dst / name).write_text(json.dumps(nb(cells), ensure_ascii=False, indent=1))
        print("wrote", dst / name)


if __name__ == "__main__":
    main(sys.argv[1])
