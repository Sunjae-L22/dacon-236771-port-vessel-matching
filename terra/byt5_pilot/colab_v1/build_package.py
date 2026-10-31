# author: terra
"""Freeze a self-contained Colab pilot; performs no full model training."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile, ZIP_DEFLATED

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DIST = HERE / "delivery"
PACKAGE = DIST / "port_byt5_pilot_v1"
DRIVE_FOLDER = "항만_ByT5_파일럿_v1_20261027"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cell(kind, source):
    return {"cell_type": kind, "id": hashlib.sha256(source.encode()).hexdigest()[:12], "metadata": {},
            "source": source.splitlines(True), **({"execution_count": None, "outputs": []} if kind == "code" else {})}


def main():
    assert not PACKAGE.exists(), "이미 만든 패키지는 보존합니다. 새 delivery 경로를 쓰거나 작업 중 출력만 명시적으로 재생성하세요."
    PACKAGE.mkdir(parents=True)
    frozen = ROOT / "shared/experiments/pinned/build31"
    copies = ["data/open/train.csv", "data/open/vessel_catalog.csv",
              "terra/byt5_pilot/split_v1/rows.csv", "terra/byt5_pilot/split_v1/vessels.csv",
              "terra/byt5_pilot/split_v1/manifest.json",
              "shared/experiments/byt5_baseline/llm_scored_trainrole.csv"]
    for name in ["opus_lib.py", "ko_translit_argon.py", "cv.py", "opus_cv.py", "opus_stress2.py", "opus_stress4.py"]:
        copies.append(f"shared/experiments/pinned/build31/{name}")
    for rel in copies:
        target = PACKAGE / rel; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    for name in ["runtime_common.py", "train_byt5.py", "evaluate_byt5.py", "smoke_test.py"]:
        shutil.copyfile(HERE / name, PACKAGE / name)
    baseline = ROOT / "shared/experiments/byt5_baseline"
    included_baseline = (baseline / "manifest.json").exists() and (baseline / "ranker.txt").exists()
    if included_baseline:
        for name in ["manifest.json", "ranker.txt"]:
            shutil.copyfile(baseline / name, PACKAGE / "shared/experiments/byt5_baseline" / name)
    train = pd.read_csv(PACKAGE / "data/open/train.csv")
    catalog = pd.read_csv(PACKAGE / "data/open/vessel_catalog.csv").set_index("vessel_id")
    roles = pd.read_csv(PACKAGE / "terra/byt5_pilot/split_v1/rows.csv").set_index("sample_id")
    train["role"] = train.sample_id.map(roles.role)
    assert train.role.notna().all()
    ids = {r: set(train.loc[train.role.eq(r), "target_vessel_id"]) for r in ["train", "tune", "eval"]}
    assert not ids["train"] & (ids["tune"] | ids["eval"]) and not ids["tune"] & ids["eval"]
    tr = train[train.role.eq("train")]
    real = tr[tr.vessel_name.str.contains("[가-힣]", regex=True)].copy()
    real = pd.DataFrame({"vessel_id": real.target_vessel_id, "english": real.target_vessel_id.map(catalog.vessel_name),
                         "korean": real.vessel_name, "source": "real", "role": "train", "sample_id": real.sample_id})
    g = pd.read_csv(PACKAGE / "shared/experiments/byt5_baseline/llm_scored_trainrole.csv")
    assert set(g.vessel_id) <= ids["train"]
    base = tr.drop_duplicates("target_vessel_id").set_index("target_vessel_id")
    g = g[(g.score >= .70) & g.vessel_id.map(base.gross_tonnage).notna()].copy()
    synthetic = pd.DataFrame({"vessel_id": g.vessel_id, "english": g.vessel_id.map(catalog.vessel_name),
                 "korean": g.ko, "source": "llm_v1", "role": "train", "sample_id": [f"LLM_TRAIN_{i:05d}" for i in range(len(g))]})
    pairs = pd.concat([real, synthetic], ignore_index=True).drop_duplicates(["vessel_id", "english", "korean", "source"])
    # Source-balanced sampling, then equal mass per vessel and per spelling within each source.
    for source in pairs.source.unique():
        mask = pairs.source.eq(source)
        count = pairs[mask].groupby("vessel_id").size()
        pairs.loc[mask, "sample_weight"] = .5 / len(count) / pairs.loc[mask, "vessel_id"].map(count)
    assert abs(pairs.sample_weight.sum()-1) < 1e-8
    tune = train[train.role.eq("tune") & train.vessel_name.str.contains("[가-힣]", regex=True)]
    tune = pd.DataFrame({"vessel_id": tune.target_vessel_id, "english": tune.target_vessel_id.map(catalog.vessel_name),
                         "korean": tune.vessel_name, "role": "tune"}).drop_duplicates()
    prepared = PACKAGE / "prepared"; prepared.mkdir()
    pairs.to_csv(prepared / "train_pairs.csv", index=False)
    tune.to_csv(prepared / "tune_pairs.csv", index=False)
    sys.path.insert(0, str(frozen))
    import opus_stress2 as S2
    import opus_stress4 as S4
    queries = []
    validation = train[train.role.isin(["tune", "eval"]) & train.vessel_name.str.contains("[가-힣]", regex=True)]
    for original in validation.to_dict("records"):
        seed = int(hashlib.sha256(original["sample_id"].encode()).hexdigest()[:8], 16)
        S2.R.seed(seed); S4.R.seed(seed)
        cname = catalog.loc[original["target_vessel_id"], "vessel_name"]
        variants = {"orig": original, "typo1": S2.V["typo1"](original), "typo2": S2.V["typo2"](original),
                    "jamo1": S4.V["jamo1"](original, cname), "trunc60": S2.V["trunc60%"](original),
                    "drop_last": S2.V["drop_last_word"](original), "reorder": S4.V["reorder"](original, cname)}
        for name, row in variants.items():
            row = dict(row)
            row.update(orig_sample_id=original["sample_id"], query_sample_id=original["sample_id"]+"__"+name,
                       variant=name, changed_query=row["vessel_name"] != original["vessel_name"])
            row["sample_id"] = row["query_sample_id"]
            queries.append(row)
    q = pd.DataFrame(queries)
    assert q.query_sample_id.is_unique and set(q.target_vessel_id).isdisjoint(ids["train"])
    q.to_csv(prepared / "queries.csv", index=False)
    cfg = {"model_id": "google/byt5-small", "seed": 42, "steps": 600, "learning_rate": .0003,
           "warmup_steps": 30, "batch_size": 4, "gradient_accumulation": 4, "eval_every": 100,
           "score_batch_size": 8, "max_source": 128, "max_target": 192, "top_k": 20}
    (PACKAGE / "config.json").write_text(json.dumps(cfg, indent=2))
    manifest = {"status": "LOCAL_PREPARED_GPU_UNEXECUTED", "baseline_model_included": included_baseline,
        "training_pairs": len(pairs), "training_sources": pairs.source.value_counts().to_dict(),
        "source_sampling_mass": pairs.groupby("source").sample_weight.sum().to_dict(),
        "tune_checkpoint_pairs": len(tune), "queries": len(q),
        "query_counts": q.groupby(["role", "variant"]).size().to_dict().__str__(),
        "independent_eval_korean_vessels": len(set(q.loc[q.role.eq("eval"), "target_vessel_id"])),
        "official_test_access": False,
        "files_sha256": {str(p.relative_to(PACKAGE)): sha(p) for p in sorted(PACKAGE.rglob("*")) if p.is_file()}}
    (PACKAGE / "package_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    zp = DIST / "port_byt5_pilot_v1.zip"
    with ZipFile(zp, "w", ZIP_DEFLATED) as z:
        for p in sorted(PACKAGE.rglob("*")):
            if p.is_file(): z.write(p, p.relative_to(PACKAGE))
    ziphash = sha(zp)
    cells = [cell("markdown", "# 항만 ByT5 독립 파일럿 v1\n\nL4 또는 A100 GPU를 선택한 뒤 위에서부터 실행하세요. 학습·평가 결과는 Drive에 저장됩니다. 전체 train 기반 33/36 모델을 검증 기준선으로 사용하지 않습니다. 공식 test 접근과 대회 제출은 하지 않습니다.\n\n진행: 환경 설치 → Drive/입력 확인 → GPU 학습 → 기준 후보 → 조건부 점수·결합 평가. 학습은 기본 600 optimizer steps이며 소요 시간은 GPU에서 확인합니다."),
      cell("code", "# 1. 필요한 라이브러리만 설치합니다. PyTorch는 코랩 기본 GPU 버전을 유지합니다.\nimport subprocess, sys\nsubprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.57.6', 'accelerate==1.9.0', 'safetensors', 'sentencepiece', 'lightgbm', 'rapidfuzz'])\nprint('설치 완료. 다음 셀을 실행하세요.')"),
      cell("code", f"# 2. Drive 연결과 입력 검증\nfrom google.colab import drive, files\ndrive.mount('/content/drive')\nfrom pathlib import Path\nimport hashlib, zipfile, json, os\nBASE = Path('/content/drive/MyDrive/{DRIVE_FOLDER}')\nBASE.mkdir(parents=True, exist_ok=True)\nZIP = BASE / 'port_byt5_pilot_v1.zip'\nif not ZIP.exists():\n    print('준비된 port_byt5_pilot_v1.zip을 선택하세요.')\n    uploaded = files.upload()\n    ZIP.write_bytes(uploaded['port_byt5_pilot_v1.zip'])\nassert hashlib.sha256(ZIP.read_bytes()).hexdigest() == '{ziphash}', '다른 버전의 ZIP입니다.'\nROOT = Path('/content/port_byt5_pilot_v1')\nROOT.mkdir(exist_ok=True)\nwith zipfile.ZipFile(ZIP) as z:\n    for n in z.namelist():\n        assert not Path(n).is_absolute() and '..' not in Path(n).parts\n    z.extractall(ROOT)\nos.environ['PYTHONHASHSEED']='0'\nos.environ['OMP_NUM_THREADS']='2'\nos.environ['OPENBLAS_NUM_THREADS']='2'\nsys.path.insert(0, str(ROOT))\nfrom runtime_common import verify_package\nmanifest = verify_package(ROOT)\nprint({{k: manifest[k] for k in ['training_pairs','training_sources','baseline_model_included','independent_eval_korean_vessels']}})\nsubprocess.check_call([sys.executable, str(ROOT/'smoke_test.py')])\nimport torch\nassert torch.cuda.is_available(), '런타임 유형에서 GPU를 선택하세요.'\nprint('GPU:', torch.cuda.get_device_name())"),
      cell("code", "# 3. 실행 설정 (이름이 같으면 완료 결과 또는 마지막 체크포인트를 이어 씁니다.)\nRUN_NAME = 'pilot_v1_s42_600'\nSTEPS = 600\nBUILD_BASELINE_IF_MISSING = True  # 패키지에 기준선이 없으면 코랩 CPU에서 단일 시드 생성\nRUN = BASE / 'runs' / RUN_NAME\nRUN.mkdir(parents=True, exist_ok=True)\nprint('결과 저장:', RUN)"),
      cell("code", "# 4. GPU 학습: 검증 선박을 학습에 쓰지 않으며 checkpoint 선택은 tune만 사용합니다.\nsubprocess.check_call([sys.executable, str(ROOT/'train_byt5.py'), '--root', str(ROOT), '--output', str(RUN), '--steps', str(STEPS)])\nprint(json.loads((RUN/'training_complete.json').read_text()))"),
      cell("code", "# 5. 자연 검색 상위 20후보. 정답을 강제로 추가하지 않습니다.\ncmd = [sys.executable, str(ROOT/'evaluate_byt5.py'), 'baseline', '--root', str(ROOT), '--run', str(RUN)]\nif BUILD_BASELINE_IF_MISSING: cmd.append('--allow-baseline-training')\nsubprocess.check_call(cmd)"),
      cell("code", "# 6. ByT5 조건부 점수와 실제 GPU 처리량을 계산합니다.\nsubprocess.check_call([sys.executable, str(ROOT/'evaluate_byt5.py'), 'score', '--root', str(ROOT), '--run', str(RUN)])"),
      cell("code", "# 7. 결합 가중은 tune에서만 고르고, 그 뒤 eval을 한 번 평가합니다.\nsubprocess.check_call([sys.executable, str(ROOT/'evaluate_byt5.py'), 'report', '--root', str(ROOT), '--run', str(RUN)])\nimport pandas as pd\nm = pd.read_csv(RUN/'metrics.csv')\ndisplay(m[m.role.eq('eval')])\nprint(json.loads((RUN/'pilot_report.json').read_text()))\nREPORT_ZIP = RUN / 'pilot_results.zip'\nwith zipfile.ZipFile(REPORT_ZIP, 'w', zipfile.ZIP_DEFLATED) as z:\n    for p in RUN.glob('*'):\n        if p.suffix in ['.json','.jsonl','.csv']: z.write(p,p.name)\n    for p in (RUN/'baseline').glob('*.json'): z.write(p,'baseline/'+p.name)\nprint('분석용 결과 ZIP:', REPORT_ZIP)\n# 필요하면 아래 주석을 풀어 다운로드하세요.\n# files.download(str(REPORT_ZIP))"),
      cell("markdown", "학습 가중치는 Drive의 runs/RUN_NAME/best에 남습니다. 분석에는 pilot_results.zip을 전달하면 됩니다. 이것은 한글 52척의 독립 holdout 파일럿이며, 전체 OOF·Public 점수·제출 완료를 뜻하지 않습니다. GPU 실행 전에는 로컬 점검만 완료된 상태입니다.")]
    nb = {"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"},"colab":{"name":"항만_ByT5_파일럿_v1.ipynb"},"accelerator":"GPU"},"cells":cells}
    nbpath = DIST / "항만_ByT5_파일럿_v1.ipynb"
    nbpath.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    (DIST / "delivery_manifest.json").write_text(json.dumps({"zip_sha256":ziphash,"notebook_sha256":sha(nbpath),"baseline_model_included":included_baseline,"package":manifest},ensure_ascii=False,indent=2))
    print(json.dumps({"zip":str(zp),"zip_bytes":zp.stat().st_size,"notebook":str(nbpath),"training_pairs":len(pairs),"baseline_model_included":included_baseline},ensure_ascii=False))


if __name__ == "__main__": main()
