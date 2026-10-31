To: opus
Re: 013_20261031-2210_byt5-result-HOLD.md

# 기존 가중치 재사용, Public0회 후속 검증

Opus112/113/72B/조합/HSW재현·3자투표와겹치지않는ByT5채널만진행. 사용자가기존가중치재활용·0.97목표연구계속명시.

## Rseq — 결과 보기 전에 계산법 고정

clean실패리퍼스호에서Lepus(mean logp−1.4597,6bytes inclEOS)보다River Spirit(−.8263,13bytes)이유리해졌다. 총logp는각−8.758/−10.741로순서가다르다. 평균출력길이정규화가쉬운영어완성을보상하는가설을검증한다. 이는이름예외아님.

기존반환mean logp에각canonical targetUTF8길이+EOS를곱해sequence logp복원. query마다기대target길이로만나눔: `len(NFC(query).encode(UTF8))+1` × fit전체학습쌍의`target_bytes/query_bytes`중앙값. fit2/fit4각기통계;heldout정답은계산에사용하지않음.고정융합계수2와110gap<4는유지. 임계값/길이승수스윕없음.

기존개발fold를본뒤도출한진단이므로좋아져도독립확증으로부르지않음. cleanharm0조건을완화하지않고,Public111실패를고려해합성CV만으로제출하지않음.

## Rretrieve — 기존모델에서검색후보생성

학습가중치변경없이한글query→영어이름검색. 공식카탈로그영어이름의prefix trie로beam16을제약하여허용catalog이름만생성. 기존fit정답선박은최종허용대상에서제외(검색trie는그허용이름집합). 동명이선ID는모두보존. 영문/110easy불변.

별도실험으로,새검색후보만110원래후보에추가하고원래110피처·랭커로선택하는retrieval-only를먼저측정. 이후sequence점수추가결과는별도열로분리하여한번에원인을섞지않음. GPU작업은새학습없이Drive의이전fold모델정확히재사용. 원래반환FINAL_STATUS의해시로run을찾고가중치파일은쓰기금지.

모든결과는개발검증.검증용새질의/원본holdout분리와추가미사용fold가필요하면선행준비. 현재제출추천0개,리더보드호출0회.07시후새후보없음.
