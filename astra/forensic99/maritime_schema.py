"""Universal Korean <-> English Country/Flag & Maritime Vessel-Type Semantic Schema.

100% general Korean-English language translation (ISO country/territory names + general
maritime ship category taxonomy). Zero external vessel database or ship proper nouns.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
import numpy as np
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# 1. Korean <-> English Country & Maritime Flag Registry Translation
# ---------------------------------------------------------------------------
# Maps standard Korean country/territory names -> (primary English catalog flag names, secondary/affiliated registries)
KO_FLAG_TO_EN: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "가봉": (("Gabon",), ()),
    "가나": (("Ghana",), ()),
    "가이아나": (("Guyana",), ()),
    "감비아": (("Gambia",), ()),
    "과테말라": (("Guatemala",), ()),
    "그레나다": (("Grenada",), ()),
    "그리스": (("Greece",), ("Cyprus",)),
    "그린란드": (("Greenland",), ("Denmark", "Danish Int'l Register")),
    "기니": (("Guinea",), ()),
    "기니비사우": (("Guinea Bissau",), ()),
    "나미비아": (("Namibia",), ()),
    "나우루": (("Nauru",), ()),
    "나이지리아": (("Nigeria",), ()),
    "남아프리카 공화국": (("South Africa",), ()),
    "남아프리카공화국": (("South Africa",), ()),
    "남아공": (("South Africa",), ()),
    "네덜란드": (("Netherlands",), ("Curacao", "Aruba")),
    "노르웨이": (("Norway", "Norwegian Int'l Register"), ()),
    "뉴질랜드": (("New Zealand",), ("Cook Islands", "Niue")),
    "니우에": (("Niue",), ()),
    "니카라과": (("Nicaragua",), ()),
    "대만": (("Taiwan",), ()),
    "대한민국": (("South Korea",), ()),
    "한국": (("South Korea",), ()),
    "덴마크": (("Denmark", "Danish Int'l Register"), ("Faeroes", "Greenland")),
    "도미니카": (("Dominica", "Dominican Republic"), ()),
    "도미니카 공화국": (("Dominican Republic",), ()),
    "도미니카공화국": (("Dominican Republic",), ()),
    "독일": (("Germany",), ()),
    "동티모르": (("Timor-Leste",), ()),
    "라오스": (("Laos",), ()),
    "라이베리아": (("Liberia",), ()),
    "라트비아": (("Latvia",), ()),
    "러시아": (("Russia",), ()),
    "레바논": (("Lebanon",), ()),
    "레소토": (("Lesotho",), ()),
    "루마니아": (("Romania",), ()),
    "룩셈부르크": (("Luxembourg",), ()),
    "리비아": (("Libya",), ()),
    "리투아니아": (("Lithuania",), ()),
    "마다가스카르": (("Madagascar",), ()),
    "마샬 제도": (("Marshall Islands",), ()),
    "마셜 제도": (("Marshall Islands",), ()),
    "마데이라": (("Madeira",), ("Portugal",)),
    "말레이시아": (("Malaysia",), ()),
    "맨섬": (("Isle of Man",), ("United Kingdom",)),
    "멕시코": (("Mexico",), ()),
    "모나코": (("Monaco",), ()),
    "모로코": (("Morocco",), ()),
    "모리셔스": (("Mauritius",), ()),
    "모리타니": (("Mauritania",), ()),
    "모잠비크": (("Mozambique",), ()),
    "몬테네그로": (("Montenegro",), ()),
    "몰도바": (("Moldova",), ()),
    "몰디브": (("Republic of Maldives",), ()),
    "몰타": (("Malta",), ()),
    "몽골": (("Mongolia",), ()),
    "미국": (("United States",), ("Virgin Islands (U.S.)", "Puerto Rico")),
    "미얀마": (("Union of Myanmar",), ()),
    "미크로네시아": (("Micronesia",), ()),
    "바누아투": (("Vanuatu",), ()),
    "바레인": (("Bahrain",), ()),
    "바베이도스": (("Barbados",), ()),
    "바하마": (("Bahamas",), ()),
    "방글라데시": (("Bangladesh",), ()),
    "버뮤다": (("Bermuda",), ("United Kingdom",)),
    "베냉": (("Benin",), ()),
    "베네수엘라": (("Venezuela",), ()),
    "베트남": (("Vietnam",), ()),
    "벨기에": (("Belgium",), ("Luxembourg",)),
    "벨리즈": (("Belize",), ()),
    "보스니아 헤르체고비나": (("Bosnia & Herzegovina",), ()),
    "볼리비아": (("Bolivia",), ()),
    "북마케도니아": (("Republic of North Macedonia",), ()),
    "북한": (("North Korea",), ()),
    "불가리아": (("Bulgaria",), ()),
    "브라질": (("Brazil", "Reg. Especial Brasiliero"), ()),
    "브루나이": (("Brunei",), ()),
    "사모아": (("Independent State of Samoa",), ()),
    "사우디아라비아": (("Saudi Arabia",), ()),
    "산마리노": (("San Marino",), ()),
    "상투메 프린시페": (("Sao Tome & Principe",), ()),
    "세네갈": (("Senegal",), ()),
    "세이셸": (("Seychelles",), ()),
    "세인트 빈센트 그레나딘": (("St. Vincent & Grenadines",), ()),
    "세인트빈센트 그레나딘": (("St. Vincent & Grenadines",), ()),
    "세인트키츠 네비스": (("St. Kitts & Nevis",), ()),
    "소말리아": (("Somalia",), ()),
    "솔로몬 제도": (("Solomon Islands",), ()),
    "수단": (("Sudan",), ()),
    "수리남": (("Suriname",), ()),
    "스리랑카": (("Sri Lanka",), ()),
    "스웨덴": (("Sweden",), ()),
    "스위스": (("Switzerland",), ()),
    "스페인": (("Spain",), ()),
    "슬로베니아": (("Slovenia",), ()),
    "시리아": (("Syria",), ()),
    "시에라 리온": (("Sierra Leone",), ()),
    "시에라리온": (("Sierra Leone",), ()),
    "싱가포르": (("Singapore",), ()),
    "아랍에미리트": (("United Arab Emirates",), ()),
    "아루바": (("Aruba",), ("Netherlands",)),
    "아르헨티나": (("Argentina",), ()),
    "아이슬란드": (("Iceland",), ()),
    "아이티": (("Haiti",), ()),
    "아일랜드": (("Irish Republic",), ()),
    "아제르바이잔": (("Azerbaijan",), ()),
    "알바니아": (("Albania",), ()),
    "알제리": (("Algeria",), ()),
    "앙골라": (("Angola",), ()),
    "앤티가 바부다": (("Antigua & Barbuda",), ()),
    "앤틸리스제도": (("Curacao", "Aruba"), ("Netherlands",)),
    "앵귈라": (("Anguilla",), ("United Kingdom",)),
    "에리트레아": (("Eritrea",), ()),
    "에스와티니": (("Eswatini",), ()),
    "에스토니아": (("Estonia",), ()),
    "에콰도르": (("Ecuador",), ()),
    "에티오피아": (("Ethiopia",), ()),
    "영국": (("United Kingdom",), ("Isle of Man", "Gibraltar", "Cayman Islands", "Bermuda", "Channel Islands", "Virgin Islands (British)", "Falkland Islands", "Turks & Caicos Islands", "Anguilla")),
    "예멘": (("Republic of Yemen",), ()),
    "오만": (("Oman",), ()),
    "오스트레일리아": (("Australia",), ()),
    "호주": (("Australia",), ()),
    "온두라스": (("Honduras",), ()),
    "요르단": (("Jordan",), ()),
    "우루과이": (("Uruguay",), ()),
    "우크라이나": (("Ukraine",), ()),
    "이라크": (("Iraq",), ()),
    "이란": (("Iran",), ()),
    "이스라엘": (("Israel",), ()),
    "이집트": (("Egypt",), ()),
    "이탈리아": (("Italy",), ()),
    "인도": (("India",), ()),
    "인도네시아": (("Indonesia",), ()),
    "일본": (("Japan",), ()),
    "자메이카": (("Jamaica",), ()),
    "잠비아": (("Zambia",), ()),
    "적도 기니": (("Equatorial Guinea",), ()),
    "적도기니": (("Equatorial Guinea",), ()),
    "조지아": (("Georgia",), ()),
    "중국": (("Peoples' Republic of China",), ("Hong Kong",)),
    "지부티": (("Djibouti",), ()),
    "지브롤터": (("Gibraltar",), ("United Kingdom",)),
    "미상": (("Unknown",), ()),
    "체코": (("Czech Republic",), ()),
    "칠레": (("Chile",), ()),
    "카메룬": (("Cameroon",), ()),
    "카보베르데": (("Cape Verde",), ()),
    "카자흐스탄": (("Kazakhstan, Republic of",), ()),
    "카타르": (("Qatar",), ()),
    "캄보디아": (("Cambodia",), ()),
    "캐나다": (("Canada",), ()),
    "케냐": (("Kenya",), ()),
    "케이맨 제도": (("Cayman Islands",), ("United Kingdom",)),
    "코모로": (("Comoro Islands",), ()),
    "코스타리카": (("Costa Rica",), ()),
    "코트디부아르": (("Ivory Coast",), ()),
    "콜롬비아": (("Colombia",), ()),
    "콩고민주공화국": (("Democratic Republic of Congo",), ()),
    "쿠바": (("Cuba",), ()),
    "쿠웨이트": (("Kuwait",), ()),
    "쿡 제도": (("Cook Islands",), ()),
    "큐라소": (("Curacao",), ("Netherlands",)),
    "크로아티아": (("Croatia",), ()),
    "키리바시": (("Kiribati",), ()),
    "키프로스": (("Cyprus",), ()),
    "탄자니아": (("Tanzania",), ()),
    "태국": (("Thailand",), ()),
    "터키": (("Turkey",), ()),
    "튀르키예": (("Turkey",), ()),
    "토고": (("Togo",), ()),
    "통가": (("Tonga",), ()),
    "투르크메니스탄": (("Turkmenistan",), ()),
    "투발루": (("Tuvalu",), ()),
    "튀니지": (("Tunisia",), ()),
    "트리니다드 토바고": (("Trinidad",), ()),
    "파나마": (("Panama",), ()),
    "파라과이": (("Paraguay",), ()),
    "파키스탄": (("Pakistan",), ()),
    "파푸아뉴기니": (("Papua New Guinea",), ()),
    "팔라우": (("Palau",), ()),
    "페로 제도": (("Faeroes",), ("Denmark", "Danish Int'l Register")),
    "페루": (("Peru",), ()),
    "포르투갈": (("Portugal", "Madeira"), ()),
    "폴란드": (("Poland",), ()),
    "푸에르토리코": (("Puerto Rico",), ("United States",)),
    "프랑스": (("France", "Registre International Francais"), ("Kerguelen Islands", "Wallis & Futuna Islands (French)", "French Polynesia", "New Caledonia", "Reunion", "French Ant. T'tory")),
    "피지": (("Fiji",), ()),
    "핀란드": (("Finland",), ()),
    "필리핀": (("Philippines",), ()),
    "홍콩": (("Hong Kong",), ("Peoples' Republic of China",)),
}


# ---------------------------------------------------------------------------
# 2. Korean PORT-MIS Vessel Categories <-> 138 Catalog Maritime Ship Types
# ---------------------------------------------------------------------------
# Maps Korean vessel_type -> (primary exact types [1.0], closely related family types [0.45])
KO_VTYPE_TO_EN: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "산물선(벌크선)": (
        ("Bulk", "Open Hatch", "Ore", "Cement", "Aggregates", "Aggregate Carrier", "Chip", "Gypsum", "Limestone", "Salt", "Urea", "Nickel", "Misc. Bulk", "Slurry", "Gravel/Stone"),
        ("GCargo", "MPP", "MPP/Hvy. Lift", "Bulk/Oil", "Ore/Oil", "Bulk & C. Soda", "Bulk/Chem & Oil", "Ore & Sulph. Acid"),
    ),
    "석탄운반선": (
        ("Bulk", "Ore", "Misc. Bulk"),
        ("Open Hatch", "Bulk/Oil", "Ore/Oil", "GCargo"),
    ),
    "철강재 운반선": (
        ("Bulk", "MPP", "MPP/Hvy. Lift", "Open Hatch", "GCargo", "Heavy Lift", "Hvy Dk Cargo"),
        ("Deck Cargo Carr.", "Misc. Cargo"),
    ),
    "일반화물선": (
        ("GCargo", "MPP", "MPP/Hvy. Lift", "Open Hatch", "Hvy Dk Cargo", "Deck Cargo Carr.", "Heavy Lift", "Landing Craft", "Pallet Carrier", "Pulp", "Barge Carr.", "Misc. Cargo", "Livestock", "Fish Feed Carrier"),
        ("Bulk", "Chip", "Aggregates", "Aggregate Carrier", "Cement", "Ro-Ro/Lo-Lo", "Products/MPP", "Pass/Cargo", "Rfr/Gen'l"),
    ),
    "풀컨테이너선": (
        ("Container",),
        ("Ro-Ro/Cnt", "MPP", "Open Hatch"),
    ),
    "세미(혼재)컨테이너선": (
        ("MPP", "MPP/Hvy. Lift", "Open Hatch", "Ro-Ro/Cnt", "Container", "Products/MPP"),
        ("GCargo", "Ro-Ro/Lo-Lo", "Pallet Carrier"),
    ),
    "자동차운반선": (
        ("PCC", "Ro-Ro", "Ro-Ro/Lo-Lo", "Ro-Ro/Cnt", "Prod./RoRo"),
        ("RoPax", "Rfr/RoRo", "Landing Craft"),
    ),
    "원유운반선": (
        ("Tanker", "Shuttle", "FSO", "FPSO", "FPDSO"),
        ("Products", "Chem & Oil", "Bulk/Oil", "Ore/Oil", "Chemical Parcel", "Oil Transfer Vessel"),
    ),
    "석유제품 운반선": (
        ("Products", "Chemical Parcel", "Chem & Oil", "Bunkering Tanker", "Asp. & Bit.", "Chemical Bulk", "Tanker", "Oil & LPG", "Prod./RoRo", "Products/MPP"),
        ("Chemical Unknown", "Oil Transfer Vessel", "Oil Rec Tanker", "Slop Reception", "LNG & Oil Bunker", "Methanol Bunkering"),
    ),
    "케미칼 운반선": (
        ("Chemical Parcel", "Chem & Oil", "Chemical Bulk", "Chemical Unknown", "Methanol", "Methanol Bunkering", "Sulphuric A.", "M. Sulphur", "Edible Oil", "Bulk & C. Soda", "Bulk/Chem & Oil", "Ore & Sulph. Acid"),
        ("Products", "Bunkering Tanker", "Asp. & Bit.", "Wine", "Fruit Juice", "Water", "Tanker", "LCO2 Carrier"),
    ),
    "기타 유조선": (
        ("Chem & Oil", "Chemical Unknown", "Bunkering Tanker", "Oil Transfer Vessel", "Oil Rec Tanker", "Slop Reception", "Oil Poll Ctrl", "Water", "Edible Oil", "Wine", "Fruit Juice", "Asp. & Bit."),
        ("Tanker", "Products", "Chemical Parcel", "Chemical Bulk"),
    ),
    "LPG 운반선": (
        ("LPG", "Ethy/LPG", "Ethane/LPG", "LPG/FSO", "LPG/FPSO", "Oil & LPG"),
        ("LNG/Eth/LPG", "LCO2 Carrier", "CNG"),
    ),
    "LNG 운반선": (
        ("LNG", "LNG/Regas", "LNG/Eth/LPG", "LNG Bunkering", "LNG & Oil Bunker", "LNG/FSU", "LNG/FPSO", "FSRU", "FSU"),
        ("Ethane/LPG", "CNG", "LH2 Carrier"),
    ),
    "케미칼가스 운반선": (
        ("LPG", "Ethy/LPG", "Ethane/LPG", "LNG/Eth/LPG", "LCO2 Carrier", "LH2 Carrier", "CNG"),
        ("Chemical Parcel", "Chem & Oil", "Chemical Unknown", "Methanol"),
    ),
    "냉동.냉장선": (
        ("Reefer", "Rfr Fish C.", "Rfr/Gen'l", "Rfr/Pallets", "Rfr/RoRo", "Rfr/Pass/RoRo"),
        ("Fish Feed Carrier", "Fruit Juice", "GCargo"),
    ),
    "여객선": (
        ("Pass", "Pass/Car F.", "RoPax", "Pass/Cargo", "Pass/Cat", "Pass/Car Cat", "Pass/Tri", "Air Cush F.", "Rfr/Pass/RoRo"),
        ("Cruise", "Ro-Ro", "Crew Boat"),
    ),
    "크루즈선": (
        ("Cruise",),
        ("Pass", "Pass/Car F.", "RoPax", "Accom. Unit"),
    ),
    "견인용예선": (
        ("AHT", "AHTS", "O. Salv. Tug", "Ocean. Tug"),
        ("PSV", "Supply", "Crew/Fast Supply", "Crew Boat", "Offs. Crew Tender", "Utility", "MSV", "ERRV", "Standby Safety", "Misc Offshore"),
    ),
    "기타선": (
        ("Research", "Geo. Survey", "Seis. Survey", "Seis. Support", "Oceanograph.", "Hydrographic", "Cable Layer", "Pipe Layer", "Flex Layer", "Derrick/Lay Vessel", "Semi-Sub Hvy Lift", "Hvy L/Crane", "Drillship", "Diving Spt", "ROV/Sub Spt", "MSV", "PSV", "Supply", "Utility", "ERRV", "Standby Safety", "CSOV", "SOV", "EWTV", "Well Stimul.", "Maintenance", "Accom. Unit", "B/D/G Dredger", "CS/BW Dredger", "Stone Dump/Fallpipe", "Waste", "Misc Offshore", "CFPU", "FPU"),
        ("Open Hatch", "Landing Craft", "Crew Boat", "Crew/Fast Supply", "Offs. Crew Tender", "AHT", "AHTS", "O. Salv. Tug", "Ocean. Tug", "Ore & Sulph. Acid"),
    ),
}


class MaritimeSchemaMatcher:
    """Fast vectorized semantic + Dirichlet-smoothed flag & vessel_type scorer.
    Pure Python/NumPy data structures (zero pyarrow / pandas objects stored).
    """

    def __init__(self, cat_flags: list[str], cat_types: list[str], train_df=None, ko_matcher=None):
        self.cat_flags = [str(x or "") for x in cat_flags]
        self.cat_types = [str(x or "") for x in cat_types]
        self.ko_matcher = ko_matcher

        self.unique_cflags = sorted(set(self.cat_flags))
        self.unique_ctypes = sorted(set(self.cat_types))
        self.cflag2id = {f: i for i, f in enumerate(self.unique_cflags)}
        self.ctype2id = {t: i for i, t in enumerate(self.unique_ctypes)}
        self.cat_flag_ids = np.array([self.cflag2id[f] for f in self.cat_flags], dtype=np.int32)
        self.cat_type_ids = np.array([self.ctype2id[t] for t in self.cat_types], dtype=np.int32)

        # Empirical counts from train_df (if provided)
        self.emp_flag: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.emp_type: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        if train_df is not None and len(train_df) > 0:
            # Expect train_df with flag_name, vessel_type, and c_idx or target_vessel_id mapped outside
            pass

        self._flag_vec_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._type_vec_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit_empirical(self, q_flags: list[str], c_flags: list[str], q_types: list[str], c_types: list[str]):
        self.emp_flag.clear()
        self.emp_type.clear()
        self._flag_vec_cache.clear()
        self._type_vec_cache.clear()
        for qf, cf in zip(q_flags, c_flags):
            if qf and qf != "nan" and qf != "None" and cf:
                self.emp_flag[qf][cf] += 1
        for qt, ct in zip(q_types, c_types):
            if qt and qt != "nan" and qt != "None" and ct:
                self.emp_type[qt][ct] += 1

    def _flag_sem_vector(self, q_flag: str) -> np.ndarray:
        """Return semantic compatibility in [0.0, 1.0] across self.unique_cflags."""
        n = len(self.unique_cflags)
        if not q_flag or q_flag in ("nan", "None", "NaN"):
            return np.full(n, -1.0, dtype=np.float32)
        qf = q_flag.strip()
        vec = np.zeros(n, dtype=np.float32)

        # Direct English match (in case test ever has English flag_name)
        qf_low = qf.lower()
        for i, cf in enumerate(self.unique_cflags):
            if cf.lower() == qf_low:
                vec[i] = 1.0

        if qf in KO_FLAG_TO_EN:
            prim, sec = KO_FLAG_TO_EN[qf]
            for p in prim:
                if p in self.cflag2id:
                    vec[self.cflag2id[p]] = 1.0
            for s in sec:
                if s in self.cflag2id:
                    vec[self.cflag2id[s]] = max(vec[self.cflag2id[s]], 0.55)
            return vec

        # Strip '제도', '공화국', '연방' and retry or phonetic transliterate!
        qf_core = re.sub(r"(제도|공화국|연방|왕국)$", "", qf).strip()
        if qf_core in KO_FLAG_TO_EN:
            prim, sec = KO_FLAG_TO_EN[qf_core]
            for p in prim:
                if p in self.cflag2id:
                    vec[self.cflag2id[p]] = 1.0
            for s in sec:
                if s in self.cflag2id:
                    vec[self.cflag2id[s]] = max(vec[self.cflag2id[s]], 0.55)
            return vec

        # Phonetic transliteration fallback for any unseen Korean country spelling
        if self.ko_matcher is not None:
            q_info = self.ko_matcher._prepare_query(qf_core or qf)
            q_tr = q_info.get("trans_compact", "")
            if q_tr:
                for i, cf in enumerate(self.unique_cflags):
                    cf_clean = re.sub(r"[^a-z0-9]", "", cf.lower())
                    if cf_clean:
                        sim = fuzz.ratio(q_tr, cf_clean) / 100.0
                        if sim >= 0.72:
                            vec[i] = max(vec[i], float(sim))
        return vec

    def _type_sem_vector(self, q_type: str) -> np.ndarray:
        """Return semantic compatibility in [0.0, 1.0] across self.unique_ctypes."""
        n = len(self.unique_ctypes)
        if not q_type or q_type in ("nan", "None", "NaN"):
            return np.full(n, -1.0, dtype=np.float32)
        qt = q_type.strip()
        vec = np.zeros(n, dtype=np.float32)

        if qt in KO_VTYPE_TO_EN:
            prim, sec = KO_VTYPE_TO_EN[qt]
            for p in prim:
                if p in self.ctype2id:
                    vec[self.ctype2id[p]] = 1.0
            for s in sec:
                if s in self.ctype2id:
                    vec[self.ctype2id[s]] = max(vec[self.ctype2id[s]], 0.45)
            return vec

        # Keyword fallback for any unseen Korean PORT-MIS vessel_type string
        kw_rules = [
            (("여객", "페리", "카페리"), ("Pass", "Pass/Car F.", "RoPax", "Pass/Cargo", "Pass/Cat", "Pass/Car Cat")),
            (("크루즈",), ("Cruise", "Pass")),
            (("예선", "견인"), ("AHT", "AHTS", "O. Salv. Tug", "Ocean. Tug", "Utility")),
            (("시멘트",), ("Cement", "Bulk")),
            (("광석", "석탄"), ("Ore", "Bulk", "Ore/Oil")),
            (("모래", "골재"), ("Aggregates", "Aggregate Carrier", "B/D/G Dredger", "Bulk")),
            (("목재", "칩"), ("Chip", "Open Hatch", "Bulk")),
            (("냉동", "냉장", "어선", "수산"), ("Reefer", "Rfr Fish C.", "Rfr/Gen'l")),
            (("컨테이너",), ("Container", "MPP", "Ro-Ro/Cnt")),
            (("자동차", "로로"), ("PCC", "Ro-Ro", "RoPax")),
            (("LNG", "엘엔지"), ("LNG", "LNG/Regas", "LNG Bunkering")),
            (("LPG", "엘피지", "가스"), ("LPG", "Ethy/LPG", "Ethane/LPG", "LCO2 Carrier")),
            (("케미칼", "화학"), ("Chemical Parcel", "Chem & Oil", "Chemical Bulk", "Chemical Unknown")),
            (("유조", "석유", "원유", "급유"), ("Tanker", "Products", "Chem & Oil", "Bunkering Tanker", "Chemical Parcel")),
            (("벌크", "산물"), ("Bulk", "Open Hatch", "Ore")),
            (("화물", "잡화", "철강"), ("GCargo", "MPP", "MPP/Hvy. Lift", "Open Hatch", "Bulk")),
            (("준설",), ("B/D/G Dredger", "CS/BW Dredger")),
        ]
        for kws, targets in kw_rules:
            if any(k in qt for k in kws):
                for t in targets:
                    if t in self.ctype2id:
                        vec[self.ctype2id[t]] = max(vec[self.ctype2id[t]], 0.90)
        return vec

    def get_flag_vectors(self, q_flag: str, alpha: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
        """Return (sem_vec[n_unique_flags], smooth_prob_vec[n_unique_flags])."""
        qf = "" if q_flag in (None, "nan", "None", "NaN") else str(q_flag).strip()
        cached = self._flag_vec_cache.get(qf)
        if cached is not None:
            return cached
        n = len(self.unique_cflags)
        if not qf:
            res = (np.full(n, -1.0, dtype=np.float32), np.full(n, 1.0 / max(n, 1), dtype=np.float32))
            self._flag_vec_cache[qf] = res
            return res

        sem = self._flag_sem_vector(qf)
        sem_pos = np.maximum(sem, 0.0)
        prior_mass = float(sem_pos.sum())
        if prior_mass > 0:
            prior = (sem_pos + 0.005) / (prior_mass + 0.005 * n)
        else:
            prior = np.full(n, 1.0 / n, dtype=np.float32)

        emp = self.emp_flag.get(qf, {})
        tot_emp = sum(emp.values())
        counts = np.array([emp.get(cf, 0) for cf in self.unique_cflags], dtype=np.float32)
        smooth_p = (counts + alpha * prior) / (tot_emp + alpha)
        res = (sem, smooth_p.astype(np.float32))
        self._flag_vec_cache[qf] = res
        return res

    def get_type_vectors(self, q_type: str, alpha: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
        """Return (sem_vec[n_unique_types], smooth_prob_vec[n_unique_types])."""
        qt = "" if q_type in (None, "nan", "None", "NaN") else str(q_type).strip()
        cached = self._type_vec_cache.get(qt)
        if cached is not None:
            return cached
        n = len(self.unique_ctypes)
        if not qt:
            res = (np.full(n, -1.0, dtype=np.float32), np.full(n, 1.0 / max(n, 1), dtype=np.float32))
            self._type_vec_cache[qt] = res
            return res

        sem = self._type_sem_vector(qt)
        sem_pos = np.maximum(sem, 0.0)
        prior_mass = float(sem_pos.sum())
        if prior_mass > 0:
            prior = (sem_pos + 0.005) / (prior_mass + 0.005 * n)
        else:
            prior = np.full(n, 1.0 / n, dtype=np.float32)

        emp = self.emp_type.get(qt, {})
        tot_emp = sum(emp.values())
        counts = np.array([emp.get(ct, 0) for ct in self.unique_ctypes], dtype=np.float32)
        smooth_p = (counts + alpha * prior) / (tot_emp + alpha)
        res = (sem, smooth_p.astype(np.float32))
        self._type_vec_cache[qt] = res
        return res

    def score_candidates(self, q_flag: str, q_type: str, cand_idx: list[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (flag_sem, flag_p_smooth, type_sem, type_p_smooth) for candidate catalog indices."""
        f_sem_u, f_prob_u = self.get_flag_vectors(q_flag)
        t_sem_u, t_prob_u = self.get_type_vectors(q_type)
        fids = self.cat_flag_ids[cand_idx]
        tids = self.cat_type_ids[cand_idx]
        return f_sem_u[fids], f_prob_u[fids], t_sem_u[tids], t_prob_u[tids]
