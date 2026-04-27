import json
from pydantic import BaseModel, model_validator

# ── 동의어 정규화 ──
SYNONYMS = {
    "product_category": {"전세대출": "주택전세자금대출", "전세 대출": "주택전세자금대출",
                         "보금자리론": "주택구입자금대출", "디딤돌": "디딤돌대출", "버팀목": "버팀목전세자금대출"},
    "housing_type":     {"공공 임대": "공공임대", "매입 임대": "매입임대", "신희타": "신혼희망타운"},
    "target":           {"신혼": "신혼부부", "청년층": "청년", "노인": "고령자"},
    "region":           {"서울특별시": "서울", "서울시": "서울", "경기도": "경기"},
}

def norm(field, val):
    if not val: return None
    return SYNONYMS.get(field, {}).get(val.strip(), val.strip())

# ── 스키마 ───
class FinanceQuery(BaseModel):
    category: str | None = None
    region:   str | None = None
    target:   str | None = None
    product_category:  str | None = None
    special_condition: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _n(cls, d):
        d["product_category"] = norm("product_category", d.get("product_category"))
        return d

class HousingQuery(BaseModel):
    category: str | None = None
    region:   str | None = None
    target:   str | None = None
    housing_type:          str | None = None
    residence_requirement: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _n(cls, d):
        d["housing_type"] = norm("housing_type", d.get("housing_type"))
        return d

class Queries(BaseModel):
    finance: FinanceQuery
    housing: HousingQuery
    
    @model_validator(mode="after")
    def _infer_category(self):
        if not self.finance.category:
            self.finance.category = "대출" if (
                self.finance.product_category or self.finance.special_condition
            ) else None
        if not self.housing.category:
            self.housing.category = "주택" if (
                self.housing.housing_type or self.housing.residence_requirement
            ) else None
        return self

# ── 누락 필드 분석 ─────────────────────────────────────────
REQUIRED_COMMON = ["region", "target"]
REQUIRED_FINANCE = ["product_category", "special_condition"]  # 최소 1개
REQUIRED_HOUSING = ["housing_type", "residence_requirement"]  # 최소 1개

LABEL = {
    "category": "카테고리(대출/주택)", "region": "지역", "target": "대상(청년·신혼부부 등)",
    "product_category": "대출 상품 종류", "special_condition": "대출 특이조건",
    "housing_type": "주택 유형(임대/분양 등)", "residence_requirement": "거주 요건",
}
HINT = {
    "category": "예: 대출 / 주택 / 둘 다",
    "region": "예: 서울 / 경기 / 부산",
    "target": "예: 청년 / 신혼부부 / 고령자",
    "product_category": "예: 주택전세자금대출 / 디딤돌대출 / 버팀목전세자금대출",
    "special_condition": "예: 생애최초 / 소득 ○○원 이하",
    "housing_type": "예: 공공임대 / 행복주택 / 분양",
    "residence_requirement": "예: 서울 1년 이상 거주",
}

def missing_fields(q: Queries) -> dict:
    def check(block, specifics):
        common   = [f for f in REQUIRED_COMMON if not getattr(block, f)]
        specific = specifics if not any(getattr(block, f) for f in specifics) else []
        return common, specific
    fc, fs = check(q.finance, REQUIRED_FINANCE)
    hc, hs = check(q.housing, REQUIRED_HOUSING)
    return {"finance": {"common": fc, "any": fs}, "housing": {"common": hc, "any": hs}}

def is_complete(miss: dict) -> bool:
    return not any(v for block in miss.values() for v in block.values())

def clarification_msg(miss: dict) -> str:
    lines = ["답변을 드리려면 정보가 더 필요해요:\n"]
    common = set(miss["finance"]["common"]) | set(miss["housing"]["common"])
    if common:
        lines += [f"  • {LABEL[f]} — {HINT[f]}" for f in REQUIRED_COMMON if f in common]
    if miss["finance"]["any"]:
        lines += ["\n💰 대출 관점 (아래 중 하나):",
                  *[f"  • {LABEL[f]} — {HINT[f]}" for f in REQUIRED_FINANCE],
                  "  (불필요 시 '대출 제외' 입력)"]
    if miss["housing"]["any"]:
        lines += ["\n🏠 주택 관점 (아래 중 하나):",
                  *[f"  • {LABEL[f]} — {HINT[f]}" for f in REQUIRED_HOUSING],
                  "  (불필요 시 '주택 제외' 입력)"]
    return "\n".join(lines)

# ── Mock LLM ──────────────────────────────────────────────
def mock_llm(q: str) -> str:
    def pick(q, m): return next((v for k, v in m.items() if k in q), None)
    region  = pick(q, {"서울": "서울", "경기": "경기", "부산": "부산", "전국": "전국"})
    target  = pick(q, {"신혼": "신혼부부", "청년": "청년", "고령": "고령자", "다자녀": "다자녀가구"})
    product = pick(q, {"전세대출": "전세대출", "구입자금": "주택구입자금대출",
                       "디딤돌": "디딤돌대출", "버팀목": "버팀목전세자금대출"})
    housing = pick(q, {"공공임대": "공공임대", "행복주택": "행복주택",
                       "매입임대": "매입임대", "신희타": "신혼희망타운", "분양": "분양", "임대": "임대"})
    return json.dumps({
        "finance": {"category": "대출" if any(k in q for k in ["대출","융자"]) else None,
                    "region": region, "target": target,
                    "product_category": product,
                    "special_condition": "생애최초" if "생애최초" in q else None},
        "housing": {"category": "주택" if any(k in q for k in ["주택","임대","분양","주거","공공임대","행복주택","신희타"]) else None,
                    "region": region, "target": target,
                    "housing_type": housing, "residence_requirement": None},
    }, ensure_ascii=False)

# ── 파이프라인 ───
def run(initial_q: str, user_answers: list[str] = [], llm=mock_llm, max_turns: int = 3):
    q = initial_q
    print(f"\n{'='*50}\n👤질문: {initial_q}\n{'='*50}")
    for turn in range(1, max_turns + 1):
        queries = Queries.model_validate(json.loads(llm(q)))
        miss = missing_fields(queries)
        if is_complete(miss):
            print(f"✅ [시도횟수 {turn}] 완료")
            print(f" 들어온 금융 데이터: {queries.finance.model_dump(exclude_none=True)}")
            print(f" 들어온 주택 데이터: {queries.housing.model_dump(exclude_none=True)}")
            return queries
        msg = clarification_msg(miss)
        if not user_answers:
            print(f"❌ [시도횟수 {turn}] 정보 부족\n{msg}")
            return None
        ans = user_answers.pop(0)
        print(f"🤖 [시도횟수 {turn}]\n{msg}\n\n👤 추가: {ans}")
        q = f"{q}\n[추가] {ans}"
    print("❌ 최대 시도횟수 초과")
    return None

# ── 테스트 데이터 ──
TESTS = [
    ("서울 신혼부부 전세대출이랑 공공임대 조건",  []),
    ("부산 고령자 매입임대 알아봐줘",             ["고령자 주택 구입자금 대출도 같이"]),
    ("청년 지원",                                 ["서울", "전세대출이랑 행복주택"])
]

if __name__ == "__main__":
    for q, answers in TESTS:
        run(q, list(answers))