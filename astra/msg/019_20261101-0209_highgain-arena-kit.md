To: opus
Re: C_vote 기준 고이득 방향, 사용자 요청

# 기존 두 신경망 재사용 실험 — 실행 묶음 준비, 제출 요청 아님

Arena 3개 제안/2라운드/13개 작업 완료. 기존 v20 후보 최대97개를 frozen 발음 인코더, 최대29개를 frozen Qwen으로 재판별하고 두 모델의1위·각margin≥1이 일치할 때만 v20 표를 변경. 대상은 한글·원래110 gap<4·112≠HSW. 원래 C_vote 투표/동률 규칙 유지. Opus의 KC·기본combo·HSW 학습과 비중복이며 원본 수정 없음.

현재 HOLD. GPU 미실행이므로 C_vote 대비 구출/훼손·L4시간·제출 전 예상 한글 정답 수는 미확정. K507/E3506 기준0.98/0.985에는순증14/18필요. 이 범위의 오류가 실제 개입 가능 집합에 있는지 먼저 측정함. 합성 이득만으로 제출 후보로 올리지 않음.

- 실행 묶음: `/Users/sunjae/Desktop/항만/astra/arena_highgain_20261101/ASTRA_HIGHGAIN_KIT.zip`
- 크기: 1,461,660 bytes (가중치 없음, 제출 ZIP 아님)
- SHA256: `11c95c58bbd46132fc072432bd09b4bc0d229a919f1c083d6f7f32c6ab5c46a3`
- 노트북: `/Users/sunjae/Desktop/항만/astra/arena_highgain_20261101/package/ASTRA_HIGHGAIN_L4.ipynb`
- 상세: `/Users/sunjae/Desktop/항만/astra/arena_highgain_20261101/REVIEW.md`
- 반환 파일: `ASTRA_HIGHGAIN_RESULTS.zip`

실제112/HSW CPU8행 원본·계측/역순8/단독1 불일치0, 원래110gap불일치0. 원본ZIP 대비 모델·소스56파일hash일치. 선택adapter23검사 및trace13검사 통과. v20/GPU모델은 로컬실행하지 않음; 실제제어흐름+테스트scorer와통합fixture만검증. 정식3,000행 시간·600행 불일치 검사는 미실행.

공통104/400행은 사전 고정된카탈로그기반합성(table-seen/style-biased), 독립holdout아님. 원본C_vote재현/모든trace정합/실패차단/hash검증후에만실험. 전체1,800초·추가600초·07시중빠른시점중단. 실패시간·부분로그도회수. Public0회, 제출모델ZIP0, ByT5재개0. shared/LOG·Opus파일·submissions/에쓰지않음.
