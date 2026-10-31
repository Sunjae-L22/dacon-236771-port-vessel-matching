최종 통합 검토: 현재 파일 상태에서 실행을 막는 항목 1건.

1. **배포 전 필수 파일 누락 — `package/PACKAGE_MANIFEST.json`**
   `run_all.py::main`은 CUDA 확인 직후 이 파일을 무조건 읽어 키트 전체 해시를 검증한다. 현재 디렉터리와 `Path.exists()`로 확인한 결과 파일이 없다. 이 상태로 배포하면 실제 GPU에서도 baseline 실행 전에 FileNotFoundError로 중단된다. 검토 후 manifest를 생성하는 포장 단계가 예정되어 있다면 그 작업으로 해소되며, 최종 ZIP/노트북 실행 경로에 생성된 manifest가 포함되는지만 확인하면 된다. 기능·실험 파라미터 변경은 필요 없다.

다른 중대한 실행/API 결함은 이번 제한된 검토에서 발견하지 못했다. 실제 baseline owning source와 대조하여 HSW `score_candidates/allowed/pick` 연결, 112 append 위치와 gap 출처, v20 observation 안팎의 capture 순서, pandas 재파싱 후 packet 일치, 원래 router 검증, sealed result hash/행 fingerprint 확인 및 전체 프로세스 그룹 watchdog 경로를 읽었다. 패키지의 `neural_rescue.py`와 `run_gpu.py`는 최종 prototype과 SHA256 기준 동일하다. 코드 수정이나 추가 모델 실행은 하지 않았다.

주어진 23개/13개/bridge 검사 결과를 새로 실행했다고 주장하지 않는다. 이번 검토는 소스/API 대조와 작은 파일 존재·동일성 확인이며 GPU·실제 전체 시간은 여전히 미검증이다. 고정 synthetic104/400 cohort, 추가600초, 전체1800초,07시 cutoff를 변경하지 않았다.
