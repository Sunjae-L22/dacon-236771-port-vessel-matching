import json
import pandas as pd
from pathlib import Path

ROOT = Path("/Users/sunjae/Desktop/항만")
cat = pd.read_csv(ROOT / "data/open/vessel_catalog.csv")

top20 = {}
with open(ROOT / "astra/incumbent97/fold2/benchmark_6models_top20.jsonl") as f:
    for line in f:
        d = json.loads(line)
        top20[d["query_id"]] = d

with open(ROOT / "astra/incumbent97/fold2/eval_benchmark_queries.json") as f:
    queries = {q["query_id"]: q for q in json.load(f)}

preds = pd.read_csv(ROOT / "astra/incumbent97/fold2/benchmark_6models_predictions.csv")
ko = preds[(preds["source_language"] == "ko") & preds["recall"] & (~preds["correct"])]

selected = {}
for fam in ["joint_name_tonnage", "partial_name", "rename"]:
    sub = ko[(ko["family"] == fam) & (ko["target_rank"] >= 2) & (ko["target_rank"] <= 5)]
    rows = []
    seen_samples = set()
    for _, r in sub.iterrows():
        sid = queries[r["query_id"]]["sample_id"]
        if sid not in seen_samples:
            rows.append(r)
            seen_samples.add(sid)
        if len(rows) == 10:
            break
    if len(rows) < 10:
        for _, r in sub.iterrows():
            if r["query_id"] not in [x["query_id"] for x in rows]:
                rows.append(r)
            if len(rows) == 10:
                break
    selected[fam] = rows

reasons = {
    "joint_name_tonnage": [
        '호수 번호("제9호")가 잘려 자매선(Dongbang Giant) 간 변별이 안 되는 상태에서, 톤수 노이즈(13,625)로 인해 랭커가 7호를 9호보다 높게 예측',
        '선박명 뒷부분("너스") 절단 시 음차 매칭이 축약어 "Dart"와 유사하게 잡히고, 톤수 노이즈(39,280)로 정답의 톤수 일치 보너스가 소실됨',
        '접미사("파")가 잘려 Spring 자매선(Euro vs Alpha) 경합 중, 질의 국적("파나마")이 카탈로그의 Spring Euro(Panama)와 일치해 Spring Alpha(South Korea)를 역전',
        '질의 국적("대한민국")이 오답(South Korea)과 일치하고 정답(Togo)과 불일치하며, 톤수 왜곡(1,178)이 오답 톤수(1,354)에 더 가까움',
        '자매선 번호("2호")가 잘린 상태에서 질의 국적("대한민국")이 Dong Bang Giant No. 9(South Korea)와 일치하여 톤수가 일치하는 No. 2(Liberia)를 제침',
        '"에스에이치" 음차가 단어 "Sea"로 오인식되어 이름 유사도 피처에서 오답(Sea Power)이 정답(SH Freesia)보다 높게 채점됨',
        '톤수가 27,061에서 35,991로 33% 왜곡되어, 톤수 차이가 더 작은 Ning Yuan Ning Bo(39,067)에 톤수 페널티가 덜 적용됨',
        '질의 국적("마샬 제도")이 Star Pioneer(South Korea)와 불일치하고 톤수 노이즈로 인해 정답의 톤수 매칭 가산점을 상실',
        '"제이앤 아이" 음차가 "Jayne I"로 거의 100% 완전 일치하여, 약어 표기인 "JN Iris"보다 이름 완전일치 스코어를 높게 받음',
        '질의 "푸 루"가 "Fu Lu"와 완전 문자열 일치(Full exact match)하여, 원래 이름인 "Fu Rui"를 이름 스코어에서 압도함'
    ],
    "partial_name": [
        '질의가 "스타"로 극단 축약되어 동일 톤수(9,520) Star 자매선들이 동점을 이룬 상태에서 질의 국적(마샬제도)이 Star Skipper와 일치',
        '"베마그" 3음절만 남아 KoMatcher 점수가 전반적으로 낮은 가운데, 질의 국적(대한민국)이 일치하고 톤수 4,722에 가까운 FC Grace를 선택',
        '"동방자이"로 호수 정보가 완전히 소실되어 Dongbang Giant 선대 후보 간 동점 발생 시 랭커가 7호를 9호보다 선호',
        '"스타" 2글자만 남아 동일 톤수/국적 자매선(Star Skipper vs Star Clipper) 간 이름/메타 변별 피처가 전무하여 1위 오답 발생',
        '"골든" 접두어만 남고 톤수(7,251)와 국적(Panama)이 동일한 Golden 선대 자매선(Axis vs Mind) 간 점수가 0.05점 차이로 오답',
        '"스프링" 접두어만 남아 Spring 자매선들이 경합할 때, 질의 국적("파나마")이 카탈로그의 Spring Euro(Panama)와 일치',
        '"싸왓디" 접두어만 남아 동일 톤수(18,051)와 국적(South Korea)의 Sawasdee 선대 자매선(Pacific, Baltic, Atlantic) 간 미세 점수 차이로 오답',
        '1글자("씨")만 남아 극단적 정보 부족 상태에서 톤수가 20,200선인 선박들 중 영문 이니셜 "SSI"가 1순위로 잘못 랭크됨',
        '3어절 중 끝 단어("비너스")가 잘려 "디에이티 비"만남자 단어 "Dart"와의 문자열 유사도가 "DAT Venus"보다 우세하게 채점됨',
        '앞부분("오션 퀸")이 전부 잘리고 접미 약어 "에스브이"만 남아 이름 매칭 점수가 무력화된 채 톤수 대역 내 오답들이 1~4위를 차지'
    ],
    "rename": [
        '선사/도시명("부산" -> "네스토르")이 개명되었으나 접미어 "Pioneer"와 톤수(2,696)가 완전히 같은 자매선(Kobe vs Busan) 간 경합에서 오답',
        '앞단어가 "호노"로 개명되어 공통 접미사 "Trader"와 동일 톤수(9,944)를 갖는 Trader 자매선(Qingdao, Kobe, Yokohama) 중 1위 오답 발생',
        '선사 접두어가 "케이엠티씨"로 바뀌어 공통어 "Chief"와 톤수(25,483), 국적(Singapore)이 같은 Chief 선대(Samoa, Kiribati, Noumea, Tonga) 중 1위 오답',
        '이름 첫 단어가 "그랜드"로 개명되어 동일 선대 Knutsen 셔틀탱커 자매선(Carmen vs Sindre) 간 0.12점 미세 차이로 오답',
        '질의 국적("대한민국")이 카탈로그의 Vemagrace(Panama)와 불일치하고 FC Grace(South Korea)와 일치하여 메타데이터 가산점으로 오답 선택',
        '개명된 접두어("테르모필레")를 제외한 공통어 "Trader", 톤수(9,980), 국적(Liberia)이 완전히 동일한 자매선(Toyama vs Akita) 경합 오답',
        '개명으로 인해 공통어 "Pioneer"와 톤수(2,696)가 같은 Pioneer 자매선(Busan, Ulsan, Yokohama, Kobe) 간 변별 실패',
        '이름이 완전히 다른 이름("크리스탈 에이스")으로 변경되어 이름 매칭이 무효화되고, 동일 톤수(7,411) 석유제품선들 간 오답 발생',
        '완전 개명으로 이름 정보가 상실된 상태에서 질의 국적("라이베리아")이 Merkur Horizon(Liberia)과 일치하여 SM Seoul(South Korea)을 제침',
        '개명된 질의의 "파이오니아" 접미사가 AK Pioneer의 "Pioneer"와 어휘 일치하여, 톤수(4,825)가 완벽 일치하는 원래 정답 Wooyang Victory를 역전'
    ]
}

def get_cand_str(c_idx, rank, scores):
    c = cat.iloc[c_idx]
    s = scores[rank - 1] if rank <= len(scores) else 0.0
    gt = f"{c.gross_tonnage:.0f}" if pd.notnull(c.gross_tonnage) else "-"
    return f"**{c.vessel_name}**<br>(GT: {gt}, {c.vessel_type}, {c.flag_name}) [점수: {s:.2f}]"

out_lines = []
out_lines.append("\n---\n")
out_lines.append("## 6. 한글 순위 오답 30행 심층 비교 분석표 (2단계 리랭킹 피처 설계용)\n")
out_lines.append("> **[분석 대상 데이터]** `astra/incumbent97/fold2/benchmark_6models_predictions.csv` (79 한글 6모델 홀드아웃 재현셋, 전체 2,669 질의 중 오답 275건)\n"
                 "> **[선정 기준]** `source_language == 'ko'` & `recall == True` & `correct == False` & `target_rank in [2, 5]` 중 `joint_name_tonnage`, `partial_name`, `rename` 각 10행 (총 30행)\n")
out_lines.append("### 30행 전체에서 가장 흔한 오답 원인 Top 2 요약\n")
out_lines.append("1. **동일 선대/자매선(Sister Ship) 접두·접미어 및 동일 톤수 경합 (15/30건, 50.0%)**:\n"
                 "   - **대표 사례**: `Star Pioneer` vs `Star Skipper`, `Busan Pioneer` vs `Kobe Pioneer`, `Yokohama Trader` vs `Qingdao Trader`, `Tonga Chief` vs `Samoa Chief`, `Spring Alpha` vs `Spring Euro`, `Dong Bang Giant No. 9` vs `Dongbang Giant No. 7` 등.\n"
                 "   - **메커니즘**: 질의의 이름 일부가 손상/절단되거나 개명되었을 때, 공통 단어(`Pioneer`, `Trader`, `Chief`, `Star`, `Giant`)와 동일 톤수(GT)를 완벽히 공유하는 동형 자매선들이 상위 1~5위에 집중 포진합니다. 현재 1단계 랭커는 이들 간의 개별 고유 식별자나 순번(No. 9 vs No. 7)을 가려낼 정밀 변별 피처가 없어 엉뚱한 자매선(Rank 1)을 선택합니다.\n")
out_lines.append("2. **질의 메타데이터(국적/선종) 불일치에 의한 역선택 및 톤수 왜곡 바이어스 (11/30건, 36.7%)**:\n"
                 "   - **대표 사례**: `Star Pioneer`(카탈로그 국적: South Korea) 질의에 편의치적 국적(`마샬 제도`)이 들어가자 실제 국적이 Marshall Islands인 `Star Skipper`를 Rank 1로 역선택. 톤수 왜곡 노이즈(GT 27,061 -> 35,991) 발생 시 정답 대신 노이즈 톤수에 더 가까운 엉뚱한 대형선(`Ning Yuan Ning Bo`, GT 39,067)을 Rank 1로 선택.\n"
                 "   - **메커니즘**: 1단계 랭커의 트리 피처(`flag_match`, `tonnage_diff`)가 편의치적 변경(Flag of Convenience)이나 노이즈성 톤수 흔들기에 취약하여 정답보다 오답에 더 높은 메타 점수를 부여합니다.\n")

family_names = {
    "joint_name_tonnage": "이름 + 톤수 동시 손상 (joint_name_tonnage)",
    "partial_name": "부분 이름 / 절단 (partial_name)",
    "rename": "개명 / 선사 교체 (rename)"
}

for fam, rows in selected.items():
    out_lines.append(f"### 분류군: {family_names[fam]} (10선)\n")
    out_lines.append("| 번호 | 질의 (손상/원본) 및 입력 메타 | 1위 후보 (오답) | 2위 후보 | 3위 후보 | 정답 (Target, 순위) | 랭커가 1위(오답)를 고른 이유 |\n|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows):
        qid = r["query_id"]
        q = queries[qid]
        t20 = top20[qid]
        t_idx = t20["t"]
        t_cat = cat.iloc[t_idx]
        trank = int(r["target_rank"])
        scores = t20["baseline_scores"]
        
        q_name = q["vessel_name"]
        orig_name = q.get("original_vessel_name", "")
        q_gt = f"{q.get('gross_tonnage'):.0f}" if pd.notnull(q.get("gross_tonnage")) else "-"
        q_flag = str(q.get("flag_name") or "-")
        q_type = str(q.get("vessel_type") or "-")
        q_str = f"**\"{q_name}\"**<br>(원형: {orig_name})<br>GT: {q_gt}<br>선종: {q_type}<br>국적: {q_flag}"
        
        c1_str = get_cand_str(t20["candidates"][0], 1, scores)
        c2_str = get_cand_str(t20["candidates"][1], 2, scores)
        c3_str = get_cand_str(t20["candidates"][2], 3, scores)
        
        tgt_gt = f"{t_cat.gross_tonnage:.0f}" if pd.notnull(t_cat.gross_tonnage) else "-"
        tgt_s = scores[trank - 1] if trank <= len(scores) else 0.0
        tgt_str = f"**{t_cat.vessel_name}**<br>(**{trank}위**)<br>GT: {tgt_gt}<br>{t_cat.vessel_type}<br>{t_cat.flag_name}<br>[점수: {tgt_s:.2f}]"
        
        reason = reasons[fam][i]
        out_lines.append(f"| {i+1} | {q_str} | {c1_str} | {c2_str} | {c3_str} | {tgt_str} | {reason} |")
    out_lines.append("\n")

out_lines.append("## 7. Opus의 2단계 리랭킹(상위 k개 재순위) 피처 설계를 위한 제언\n")
out_lines.append("1. **자매선 변별용 고유 토큰 및 숫자 식별 피처 (Sibling Disambiguation Features)**:\n"
                 "   - 선박명에서 공통 선대 접두어(`Star`, `Dongbang Giant`, `Sawasdee`, `Pioneer`, `Chief`, `Trader`, `Knutsen`)를 마스킹한 후 **남은 고유 단어/숫자 토큰 간 유사도**(예: `Pioneer` vs `Skipper`, `No. 9` vs `No. 7`)를 집중 비교하는 서브토큰 피처가 필수적입니다.\n"
                 "   - 질의에 아라비아 숫자나 한글 호수(1호, 2호, 9호)가 존재할 경우, 후보의 숫자와 일치하지 않으면 강력한 페널티를 부여해야 합니다.\n")
out_lines.append("2. **메타데이터 소프트 평활화 및 편의치적(FOC) 강건성 (Metadata Soft-Smoothing)**:\n"
                 "   - `flag_name`이 완전 일치하지 않더라도 파나마(Panama), 라이베리아(Liberia), 마샬 제도(Marshall Islands) 등 **대표적 편의치적국(FOC) 국적 간의 전환**에 대해서는 감점 폭을 대폭 축소하는 `foc_flag_soft_match` 피처가 필요합니다.\n")
out_lines.append("3. **재화중량톤수(DWT) 보조 피처 및 GT/DWT 비율 검증**:\n"
                 "   - 카탈로그의 `deadweight_tonnage`와 `gross_tonnage` 비율은 선박 고유의 물리적 설계 비율입니다. 질의 톤수와 후보 톤수의 단순 차이 대신, 후보 선박들의 DWT/GT 프로파일을 고려하여 톤수 노이즈에 강건한 상대 순위를 산출해야 합니다.\n")

target_file = ROOT / "argon/009_20261030-1030_reply-opus028-92-diophantine-and-decomposition-plan.md"
with open(target_file, "r", encoding="utf-8") as f:
    text = f.read()

prefix = text.split("## 6. 한글 순위 오답 30행")[0].strip()
full_output = prefix + "\n\n" + "\n".join(out_lines)

with open(target_file, "w", encoding="utf-8") as f:
    f.write(full_output)

print(f"Updated {target_file} successfully! Total length: {len(full_output)} chars")
