# author: sol
"""Build a small Drive-input notebook without modifying Colab's installed packages."""
import ast
import argparse
import base64
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "shared" / "src"
p = argparse.ArgumentParser()
p.add_argument("--direction", choices=["forward", "reverse"], default="forward")
p.add_argument("--scoped", action="store_true", help="Embed the fixed input; no Colab Drive access")
ARGS = p.parse_args()
OUT = ROOT / "shared" / "experiments" / "llm_rerank" / ("port_llm_reverse_runall_v4.ipynb" if ARGS.scoped else "port_llm_rerank_reverse_v3.ipynb" if ARGS.direction == "reverse" else "port_llm_rerank_runall_v2.ipynb")


def cell(kind, source, ident):
    value = {"cell_type": kind, "metadata": {"id": ident}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        ast.parse(source)
        value.update(execution_count=None, outputs=[])
    return value


intro = """# 항만 LLM 후보 재평가 — 전체 실행 v2
**모두 실행**을 누르면 GPU 확인 → Drive 연결 → 고정 환경 준비 → 실제 모델 점검 → 전체 점수·평가가 순서대로 실행됩니다.
입력 CSV는 이미 `내 드라이브/항만_llm_aug/candidates_top20.csv`에 올려 두었습니다. 별도 업로드는 없습니다.
GPU가 없으면 먼저 **런타임 → 런타임 유형 변경 → L4 GPU**를 선택하세요. 런타임이 GPU를 자동 배정하지 못하면 코드로 강제 배정할 수는 없습니다.
결과는 같은 폴더의 `scores_Qwen2.5-3B-Instruct.csv`, `evaluation_Qwen2.5-3B-Instruct.csv`입니다. 중단 후 다시 모두 실행하면 완료된 쿼리부터 재개합니다.
3B가 eval 손상변형 평균 +2%p 및 원본1.000을 통과하지 못하면 MODEL_ID만 7B로 바꿔 한 번 실행합니다. 7B 원본 가중치는10GB를 넘으므로 통과해도 제출용 양자화 재검증이 필요합니다.
"""
setup = '''import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
import subprocess
import sys
import torch

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
BATCH_SIZE = 20 if "3B" in MODEL_ID else 10
if not torch.cuda.is_available():
    raise RuntimeError("GPU 미연결: 런타임 → 런타임 유형 변경 → L4 GPU → 저장 후 모두 실행하세요.")
print("GPU 확인:", torch.cuda.get_device_name(0), "torch:", torch.__version__)
from google.colab import drive
drive.mount("/content/drive")
WORK = Path("/content/drive/MyDrive/항만_llm_aug")
INPUT_CSV = WORK / "candidates_top20.csv"
if not INPUT_CSV.is_file():
    raise FileNotFoundError("Drive에서 항만_llm_aug/candidates_top20.csv를 찾지 못했습니다. 이 노트북과 같은 Google 계정으로 Drive를 연결하세요.")
import pandas as pd
input_check = pd.read_csv(INPUT_CSV, usecols=["query_sample_id"])
assert len(input_check) == 78540 and input_check.query_sample_id.nunique() == 3927
print("Drive CSV 확인: 78,540쌍 / 3,927쿼리")
RUN_DIR = Path("/content/port_llm_run_v2")
RUN_DIR.mkdir(parents=True, exist_ok=True)
'''
install = '''import venv
ENV_DIR = Path(f"/content/port_llm_env_v2_py{sys.version_info.major}{sys.version_info.minor}")
PYTHON = ENV_DIR / "bin" / "python"
if not PYTHON.exists():
    venv.EnvBuilder(with_pip=False, system_site_packages=True).create(ENV_DIR)
# GPU용 torch는 Colab 것을 공유하고, HF 패키지만 별도 환경에 고정합니다.
# 기본 Python의 pip --python으로 pip가 없는 분리 환경에 설치합니다.
# --no-deps이므로 Colab의 gradio/diffusers 패키지 버전을 바꾸지 않습니다.
def checked(command, **kwargs):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kwargs)
    print(result.stdout, flush=True)
    result.check_returncode()
checked([sys.executable, "-m", "pip", "--python", str(PYTHON), "install", "--disable-pip-version-check",
         "--ignore-installed", "--no-deps", "transformers==4.57.6", "accelerate==1.10.1",
         "huggingface_hub==0.36.2", "tokenizers==0.22.2", "safetensors==0.6.2"])
env = dict(os.environ, USE_TF="0", USE_FLAX="0", PYTHONNOUSERSITE="1")
checked([str(PYTHON), "-u", "-c",
               "import transformers,torch; from transformers import Qwen2ForCausalLM; "
               "assert transformers.__version__=='4.57.6'; assert torch.cuda.is_available(); "
               "print('고정 환경 확인:', transformers.__version__, torch.__version__, torch.cuda.get_device_name(0))"], env=env)
'''
sources = {name: (SRC / name).read_text() for name in
           ("score_llm_candidates.py", "evaluate_llm_rerank.py", "evaluate_llm_grid.py", "llm_rerank_run.py")}
write = "# 실행 코드를 자동 준비합니다. 사용자가 수정할 부분은 없습니다.\nSOURCES = " + repr(sources) + '''
for name, source in SOURCES.items():
    (RUN_DIR / name).write_text(source, encoding="utf-8")
print("실행 코드 준비 완료")
'''
run = '''command = [str(PYTHON), "-u", str(RUN_DIR / "llm_rerank_run.py"),
           "--work", str(WORK), "--model", MODEL_ID, "--batch-size", str(BATCH_SIZE)]
process = subprocess.Popen(command, cwd=RUN_DIR, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, bufsize=1)
try:
    for line in process.stdout:
        print(line, end="", flush=True)
    returncode = process.wait()
except BaseException:
    process.terminate()
    raise
if returncode != 0:
    raise RuntimeError("실행이 중단됐습니다. 바로 위의 마지막 오류가 원인입니다. 완료된 쿼리는 Drive에 남아 있습니다.")
'''
if ARGS.direction == "reverse":
    intro = """# 항만 7B 역방향 재점수 — 1회 실행
**모두 실행**하면 Drive의 같은78,540쌍에 log P(관측 한글 | 영문 후보)를 합계로 계산합니다. EOS는 붙이지 않습니다.
모델은 Qwen2.5-7B-Instruct만 사용합니다. 후보 데이터·기준 랭커는 그대로이며 재학습하지 않습니다.
K={10,20}, w={0,.125,.25,.5,1,2,4,8} 전 구간 원본·손상·가중점수를 저장합니다.
tune의 .5×원본+.5×손상 macro가 최대인 설정을 고르고 원본은1행 손실까지 허용합니다. eval은 선택에 쓰지 않습니다.
결과: 항만_llm_aug의 scores/evaluation/grid_Qwen2.5-7B-Instruct_reverse.csv. eval의 절단·끝단어삭제 이득과 가중평균을 보고합니다.
입력 CSV는 이미 Drive에 있습니다. GPU가 연결돼 있어야 하며 추가 모델·프롬프트 시도는 하지 않습니다.
"""
    setup = setup.replace("Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct")
    run = run.replace('"--work", str(WORK),', '"--direction", "reverse", "--work", str(WORK),')
if ARGS.scoped:
    assert ARGS.direction == "reverse"
    raw = (ROOT / "shared/experiments/llm_rerank/candidates_top20.csv").read_bytes()
    data = base64.b64encode(gzip.compress(raw, compresslevel=9, mtime=0)).decode("ascii")
    a = setup.index("from google.colab import drive")
    b = setup.index("import pandas as pd", a)
    setup = setup[:a] + '''import base64, gzip
WORK = Path("/content/port_llm_reverse_results")
WORK.mkdir(parents=True, exist_ok=True)
INPUT_CSV = WORK / "candidates_top20.csv"
INPUT_CSV.write_bytes(gzip.decompress(base64.b64decode(''' + repr(data) + ''')))
''' + setup[b:]
    setup = setup.replace("Drive CSV 확인:", "포함된 CSV 확인:")
    intro = intro.replace("Drive의 같은78,540쌍", "노트북에 포함된 같은78,540쌍")
    intro = intro.replace("결과: 항만_llm_aug의 scores/evaluation/grid_Qwen2.5-7B-Instruct_reverse.csv.",
                          "결과: runtime의 scores/evaluation/grid_Qwen2.5-7B-Instruct_reverse.csv와 결과ZIP.")
    intro = intro.replace("입력 CSV는 이미 Drive에 있습니다.", "입력은 Drive에 보관한 CSV와 바이트까지 같으며, 이 노트북은 Drive 연결 권한을 요청하지 않습니다.")
    intro += "\n**사용법:** GPU(L4 또는 A100)를 선택하고 **런타임 → 모두 실행**을 누르세요. 코드 수정·CSV 업로드는 필요 없습니다. 완료되면 결과 ZIP 다운로드가 시작됩니다. 다운로드가 안 뜨면 왼쪽 파일 패널에서 ZIP을 내려받으세요.\n동일 런타임에서 중단됐다면 모두 실행으로 이어갑니다. 런타임을 삭제·초기화하면 로컬 결과도 없어집니다.\n"
    run = run.replace("완료된 쿼리는 Drive에 남아 있습니다.", "완료된 쿼리는 현재 런타임의 port_llm_reverse_results 폴더에 남아 있습니다.")
    run += '''
import zipfile
RESULT_ZIP = WORK / "reverse7b_results.zip"
with zipfile.ZipFile(RESULT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for prefix in ("scores", "evaluation", "grid"):
        name = f"{prefix}_Qwen2.5-7B-Instruct_reverse.csv"
        z.write(WORK / name, arcname=name)
print("결과 준비 완료:", RESULT_ZIP)
print("왼쪽 파일 패널의 port_llm_reverse_results/reverse7b_results.zip에서 다운로드할 수 있습니다.")
from google.colab import files
try:
    files.download(str(RESULT_ZIP))
except Exception as download_error:
    print("계산은 완료됐습니다. 자동 다운로드 대신 위 파일 패널에서 ZIP을 내려받으세요.", download_error)
'''
nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"name": OUT.name, "gpuType": "L4", "provenance": []},
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "cells": [cell("markdown", intro, "intro"), cell("code", setup, "gpu-drive"),
                cell("code", install, "environment"), cell("code", write, "code"),
                cell("code", run, "score-evaluate")]}
if ARGS.scoped:
    nb["cells"][1]["metadata"]["cellView"] = "form"
    nb["cells"][1]["source"].insert(0, "#@title GPU 확인 · 입력 자동 준비 (Drive 권한 요청 없음)\n")
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
print(OUT, OUT.stat().st_size, "bytes")
