From: Astra
To: 사용자 / Opus 검토용 파일 (자동 전송 아님)
Subject: 기존 neural-rescue 기각, CPU 관측 오류 채널 별도64척 검증 및 C_vote 원본 진단 묶음

- 이전 ASTRA_HIGHGAIN_KIT은 실행 권고 철회. Claude의 동일신호·개입상한·검증편향 지적을 수용.
- Opus72B/KoCat 재학습·F1/융합 파일 수정0, Public제출0, 새GPU학습0. astra만 작성.
- 현재방법: 전체카탈로그 한글alias→질의의0~2회 음절삭제/중복/교환 경로 확률 합산. 영문공식train으로 오류비율 학습, 후보별alias prior정규화. 기존HSW EM과 동일방법이라는 주장이 아니라, 관측오류 층을 분리한 가설.
- V2 개발64척400행은 실제112 대비45구출/3훼손으로 clean위반 기각. 정상 짧은 이름을 긴 선명에서 삭제된 결과로 잘못 본 것이 원인.
- V3는 winning vessel의 ANY alias에서 identity/순수삭제1~2회로 설명되는 경우 보류. .99질량/110gap<4는 고정. 개발39/0은 검증치로 세지 않음.
- 코드·파라미터·다음64척을 먼저고정한 후 실제1126랭커와HSW를 실행: 별도400행에서36구출/0훼손, 한글40변경/영문0, clean128변경0. 36구출은32척, 27구출은삭제+교환 복합조건에 집중.
- 구출10행9척은112=HSW 동일오답. C_vote 규칙으로 기존답을 확정할 수 있는 영역이라 해당 합성세트에서 비중복구출 근거. 원본전체C_vote/v20/GPU실행은 아직 안 함.
- 평가질의는KoCat표에서복사하지 않고 실제train의2척이상지지75%일치 단어대응으로 학습에없는catalog이름을 조합. train/알려진fewshot선박0겹침. 그러나 여전히합성이고train어휘공유편향존재. Public향상예상으로환산금지. .98/.985달성근거아직없음.
- CPU구현600회순서/중복재검사불일치0, 영어/보호행변경0; v20대역fixture임. 공식local_check3000/전체600및L4시간미측정. 제출후보아님.

실행물:
- astra/channel_20261101/ASTRA_CHANNEL_DIAGNOSTIC.ipynb
- astra/channel_20261101/ASTRA_CHANNEL_DIAGNOSTIC.zip (1,529,388 bytes)
- ZIP SHA256 cf9b05ca9a55e9e8a57633c74060674d02c36d701f51141821e4dd30d303183e
- 상세 astra/channel_20261101/REVIEW.md

F1완료 세션에서 원본C_vote3개를 한 번 실행하고최종/각투표자예측CSV를저장. 그후동일예측에서CPU채널을평가. 평가에서는112맞힌행의훼손을112-correct로표기하며110→112정정으로오인하지않음. 각GPU투표자실패시fallback을유효기준으로인정하지않고실패처리. 전체30분/07:00중먼저도달하면중단,실패결과도반환. 제출파일생성없음. 결과 ASTRA_CHANNEL_RESULTS.zip을 사용자에게받으면 실제C_vote구출/훼손및L4시간을다음판단근거로사용.
