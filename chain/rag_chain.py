import json
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# ── TOP3 선정 프롬프트 ────────────────────────────────────
top3_prompt = ChatPromptTemplate.from_template("""
너는 서울시 청년 주거·금융 정책 추천 전문가다.
아래 후보 정책들 중 사용자 질문의 의도와 상황에 가장 잘 맞는 TOP3를 선정하라.
목표는 사용자가 "내가 왜 이 정책을 봐야 하는지", "무엇을 받을 수 있는지", "어떤 조건을 확인해야 하는지"를 바로 이해하게 하는 것이다.

[선정 기준]
1. 사용자 질문의 핵심 의도와 직접 연결되는 정책을 우선한다.
   - 전세/월세/보증금/대출/이자/금리 → 금융 정책 우선
   - 공공임대/청년주택/입주/청약/분양/주택 공급 → 주거 정책 우선
   - 포괄 추천 또는 주거+자금 문제가 함께 있으면 주거와 금융을 균형 있게 포함
2. 사용자 상황과 후보 정책의 대상, 지역, 소득조건, 지원유형, 신청기간이 많이 일치할수록 높은 순위로 둔다.
3. 신청 가능성 판단에 중요한 정보가 후보 문서에 있으면 반드시 반영한다.
4. 후보 문서에 없는 내용은 절대 추정하거나 보충하지 않는다.
5. 정책명이 비슷해도 후보의 [제목] 필드값을 정확히 그대로 사용한다.
6. 같은 목적의 정책이 여러 개면 사용자에게 더 직접적인 혜택, 조건 적합성, 실행 가능성이 큰 정책을 우선한다.
7. 근거가 부족한 정책을 억지로 좋게 포장하지 말고, 확인이 필요한 조건은 "확인 필요"라고 명시한다.

후보 정책들:
{candidates}

질문: {query}

[작성 지침]
- top3 배열은 반드시 3개를 목표로 하되, 후보가 부족하면 가능한 만큼만 출력한다.
- rank는 1, 2, 3 순서로 부여한다.
- type은 반드시 "주거" 또는 "금융" 중 하나만 사용한다.
- url 필드에는 후보 정책의 [신청주소] 또는 [URL] 정보를 정확히 기입한다. 정보가 없으면 "정보 없음"이라고 쓴다.
- 아래 JSON 예시의 type 값 "주거"는 예시일 뿐이며, 실제 출력에서는 후보 정책 유형에 맞춰 "주거" 또는 "금융" 중 정확히 하나를 넣는다.
- reason은 사용자 질문과 정책 조건이 맞는 지점을 2문장으로 구체적으로 설명한다.
- content는 대상, 지원 내용, 소득/자산 조건, 신청기간, 사용자가 확인할 사항을 3~4문장으로 압축한다.
- 문서에 없는 항목은 지어내지 말고 "문서상 확인 필요"라고 쓴다.

아래 JSON 형식으로만 출력하라. 다른 텍스트 없이 JSON만 출력:
{{
    "top3": [
        {{
            "rank": 1,
            "type": "주거",
            "title": "반드시 후보 정책의 [제목] 필드값을 그대로 사용",
            "url": "후보 문서에 포함된 신청주소 URL",
            "reason": "사용자 상황과 정책이 맞는 이유 2문장",
            "content": "대상, 지원 내용, 소득/자산 조건, 신청기간, 확인 필요 사항을 포함한 3~4문장",
            "fit_score": "상/중/하",
            "check_points": ["사용자가 확인해야 할 조건1", "조건2", "조건3"]
        }}
    ]
}}
""")

# ── 보고서 프롬프트 ───────────────────────────────────────
report_prompt = ChatPromptTemplate.from_template("""
너는 서울시 청년 주거·금융 정책 상담 보고서를 작성하는 전문가다.
반드시 아래 TOP3 정책만을 근거로 보고서를 작성하라.
문서에 없는 내용은 절대 추가하지 말고, 부족한 정보는 "문서상 확인 필요"라고 명시하라.
목표는 사용자가 정책별 차이, 본인에게 맞는 이유, 신청 전 확인사항, 다음 행동을 한 번에 파악하게 하는 것이다.

사용자 질문: {query}
TOP3 정책: {top3_context}

[작성 원칙]
1. 사용자 질문의 의도를 먼저 해석하고, TOP3 정책이 그 의도에 어떻게 대응하는지 설명한다.
2. 각 정책의 핵심 혜택, 대상, 조건, 신청기간/절차, 유의사항을 가능한 한 구체적으로 보여준다.
3. 주거 정책과 금융 정책이 함께 있으면 이용 순서와 조합 가능성을 설명하되, 중복 수혜 가능 여부가 문서에 없으면 확인 필요라고 쓴다.
4. "즉시 가능" 같은 표현은 신청기간 또는 접수 상태 근거가 있을 때만 사용한다. 근거가 없으면 "신청기간 확인 필요"라고 쓴다.
5. 소득, 자산, 무주택, 거주지, 연령, 혼인 여부 등 탈락 가능성이 큰 조건을 risks에 우선 배치한다.
6. 사용자가 바로 움직일 수 있도록 recommendation에는 1순위 정책, 확인할 서류/조건, 다음 행동 순서를 포함한다.
7. 문장은 친절하지만 단정적 과장 없이 쓴다.
8. 아래 JSON 예시의 type 값 "주거"는 예시일 뿐이며, 실제 출력에서는 각 정책 유형에 맞춰 "주거" 또는 "금융" 중 정확히 하나를 넣는다.

아래 JSON 형식으로만 출력하라. 다른 텍스트 없이 JSON만 출력:
{{
    "summary": "사용자 의도와 상황을 해석하고 TOP3 추천 방향을 요약한 4~5문장",
    "metrics": {{
        "추천정책수": "3개",
        "주거정책": "X개",
        "금융정책": "X개",
        "신청가능": "즉시 가능/신청기간 확인 필요/정책별 상이 중 하나"
    }},
    "policy_analysis": [
        {{
            "title": "정책명 (반드시 TOP3에서 받은 정확한 정책명 사용)",
            "type": "주거",
            "core": "지원 대상, 지원 내용, 주요 조건, 신청기간/확인사항을 포함한 3~4문장",
            "pros": ["사용자 상황에 맞는 구체적 장점1", "혜택 또는 활용 장점2", "다른 정책 대비 장점3"],
            "cons": ["소득/자산/무주택/거주지 등 확인할 조건1", "신청기간·서류·경쟁률 등 유의사항2", "문서상 부족하거나 추가 확인이 필요한 점3"]
        }}
    ],
    "combination": "TOP3를 함께 활용하는 전략 4~5문장. 주거 정책과 금융 정책의 신청 순서, 병행 가능성, 중복 수혜 확인 필요 여부, 우선순위를 포함",
    "risks": [
        "탈락 가능성이 큰 자격 조건",
        "신청 전에 확인해야 할 소득/자산/무주택/거주 요건",
        "준비 서류나 절차상 주의점",
        "신청기간 만료, 조기 마감, 경쟁률 등 일정 리스크",
        "문서상 확인이 필요한 정보"
    ],
    "recommendation": "사용자에게 가장 적합한 정책 조합과 행동 계획 5~6문장. 1순위 정책, 2순위 대안, 확인할 자격 조건, 준비 서류, 다음 검색/신청 행동 포함"
}}
""")

def build_chain(housing_retriever, finance_retriever):

    # ── 1. 후보군 수집 (주거 5개 + 금융 5개) ─────────────
    def get_candidates(inputs):
        query = inputs["query"]
        housing_docs = housing_retriever.invoke(query)[:5]
        finance_docs = finance_retriever.invoke(query)[:5]

        candidates = ""
        for i, doc in enumerate(housing_docs):
            # 메타데이터 + 본문 함께 넘기기
            title   = doc.metadata.get('title', f'주거정책{i+1}')
            region  = doc.metadata.get('region', '')
            target  = doc.metadata.get('target', '')
            income  = doc.metadata.get('income_condition', '')
            support = doc.metadata.get('support_type', '')
            period  = doc.metadata.get('application_period', '')
            url     = doc.metadata.get('source_url', doc.metadata.get('url', '정보 없음'))
            candidates += (
                f"[주거정책 {i+1}]\n"
                f"제목: {title}\n"
                f"지역: {region}\n"
                f"대상: {target}\n"
                f"소득조건: {income}\n"
                f"지원유형: {support}\n"
                f"신청기간: {period}\n"
                f"신청주소: {url}\n"
                f"내용: {doc.page_content[:700]}\n\n"
            )
        for i, doc in enumerate(finance_docs):
            title   = doc.metadata.get('title', f'금융정책{i+1}')
            region  = doc.metadata.get('region', '')
            target  = doc.metadata.get('target', '')
            income  = doc.metadata.get('income_condition', '')
            support = doc.metadata.get('support_type', '')
            period  = doc.metadata.get('application_period', '')
            url     = doc.metadata.get('source_url', doc.metadata.get('url', '정보 없음'))
            candidates += (
                f"[금융정책 {i+1}]\n"
                f"제목: {title}\n"
                f"지역: {region}\n"
                f"대상: {target}\n"
                f"소득조건: {income}\n"
                f"지원유형: {support}\n"
                f"신청기간: {period}\n"
                f"신청주소: {url}\n"
                f"내용: {doc.page_content[:700]}\n\n"
            )

        return {
            "query":      query,
            "candidates": candidates,
            "all_docs":   housing_docs + finance_docs
        }

    # ── 2. TOP3 선정 체인 ─────────────────────────────────
    top3_chain = (
        RunnableLambda(get_candidates)
        | top3_prompt
        | llm
        | StrOutputParser()
    )

    # ── 3. 보고서 체인 (TOP3 기반) ────────────────────────
    def get_top3_context(inputs):
        import json
        query    = inputs["query"]
        top3_raw = top3_chain.invoke(inputs)

        try:
            if "```" in top3_raw:
                top3_raw = top3_raw.split("```")[1]
                if top3_raw.startswith("json"):
                    top3_raw = top3_raw[4:]
            top3_data = json.loads(top3_raw.strip())
        except:
            top3_data = {"top3": []}

        top3_context = "\n".join([
            "\n".join([
                f"[{p.get('rank', '-')}위 / {p.get('type', '-')}] {p.get('title', '')}",
                f"적합도: {p.get('fit_score', '문서상 확인 필요')}",
                f"선정 이유: {p.get('reason', '')}",
                f"핵심 내용: {p.get('content', '')}",
                f"확인 사항: {', '.join(p.get('check_points', [])) if isinstance(p.get('check_points', []), list) else p.get('check_points', '')}",
            ])
            for p in top3_data.get("top3", [])
        ])

        return {
            "query":        query,
            "top3_context": top3_context,
            "top3_data":    top3_data
        }

    report_chain = (
        RunnableLambda(get_top3_context)
        | report_prompt
        | llm
        | StrOutputParser()
    )

    return top3_chain, report_chain




client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

state = {
    "category": None, "region": None, "target": None,
    "product_category": None, "special_condition": None,
    "housing_type": None, "residence_requirement": None
}

def is_pass():
    # 최소 조건: 대상(target)만 확인되면 나머지는 기본값으로 검색 진행
    if not state["target"]:
        return False
    # 기본값 채우기
    if not state["region"]:
        state["region"] = "서울"
    if not state["category"]:
        state["category"] = "주택,금융"
    return True

def get_missing():
    # target이 없을 때만 질문 (가장 중요한 하나의 핵심 정보만 요청)
    missing = []
    if not state["target"]:
        missing.append("현재 상황(예: 사회초년생, 신혼부부, 대학원생 등)")
    return missing

def call(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "당신은 JSON 형식으로 답변하도록 설계된 유용한 어시스턴트입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    # response.content는 이미 JSON 문자열입니다.
    return response.choices[0].message.content

def extract_info(user_text):
    prompt = f"""너는 서울시 청년 주거 정책 추천 AI의 '사용자 의도 분석기'다.
사용자 발화를 정책 검색에 필요한 구조화 정보로 변환하라.
목표는 사용자가 원하는 정책 유형, 생애/가구 상황, 거주/관심 지역, 주거 문제를 정확히 파악하는 것이다.

[핵심 원칙]
1. 사용자가 명시한 정보는 그대로 우선한다.
2. 암묵적 정보는 근거가 충분할 때만 추론한다. 애매하면 null로 둔다.
3. 서울시 청년 주거 정책 추천 서비스 맥락상, 타 지역이 명시되지 않으면 region은 "서울"로 둔다.
4. "추천해줘", "알려줘", "뭐 받을 수 있어"처럼 포괄적으로 묻는 경우 category는 "주택,금융"으로 둔다.
5. 전세/월세/보증금/대출/이자처럼 돈 문제를 말하면 금융 의도가 있다.
6. 임대주택/청년주택/공공임대/입주/분양/청약처럼 주택 공급 문제를 말하면 주택 의도가 있다.
7. 사용자가 여러 의도를 함께 말하면 버리지 말고 쉼표로 함께 기록한다. 예: "주택,금융"
8. 반환값은 반드시 아래 허용값 또는 자연어 요약만 사용하고, 모르는 값은 null로 둔다.

[필드별 추출 기준]
- category: 사용자가 찾는 정책 범주.
  - 허용값: "주택", "금융", "주택,금융"
  - 전세대출, 월세지원, 보증금, 이자, 금리, 버팀목, 디딤돌 → "금융"
  - 청년안심주택, 공공임대, 장기전세, 행복주택, 역세권청년주택, 입주, 청약, 분양 → "주택"
  - 집을 구하는 상황에서 대출/지원금도 함께 필요하거나 범주가 불분명한 추천 요청 → "주택,금융"

- region: 정책 적용 또는 관심 지역.
  - 서울 전체 문의 또는 지역 미언급 → "서울"
  - 서울 구/동/역세권 언급 → "서울 [구체 지역]" 예: "서울 관악구", "서울 왕십리역 인근"
  - 타 지역 명시 → 사용자가 말한 지역 그대로 기록

- target: 사용자의 생애/가구 유형.
  - 취준생, 구직중, 무직, 알바, 사회초년생, 직장인, 대학생, 대학원생, 1인가구, 자취 예정 → "청년"
  - 신혼, 예비부부, 결혼 예정, 결혼 N년차, 출산 예정, 아이 있음, 육아 중 → "신혼부부"
  - 한부모, 다자녀, 장애인 등은 target을 바꾸기보다 special_condition에 기록하되, 청년/신혼부부 여부가 보이면 target도 함께 추론

- product_category: 금융상품 세부 의도.
  - 전세자금, 전세대출, 보증금 대출, 임차보증금 → "주택전세자금대출"
  - 월세, 월세 지원, 월세 보조, 월세 대출 → "월세지원"
  - 주택구입, 매매, 내집마련, 디딤돌, 집 사고 싶다 → "주택구입자금대출"
  - 이자 지원, 금리 지원만 명시되고 용도가 불명확 → "이자지원"
  - 금융 의도가 없거나 세부 용도가 불명확 → null

- special_condition: 우대/제약/위험 조건.
  - 저소득, 중위소득, 소득 없음, 연봉, 월소득, 자산, 무주택, 한부모, 다자녀, 장애, 군복무, 외국인, 프리랜서, 비정규직, 사업자, 반려동물, 보증금 부족 등 정책 자격 판단에 중요한 조건을 짧게 기록
  - 여러 조건은 쉼표로 연결

- housing_type: 원하는 주거 형태 또는 거래 유형.
  - 전세, 월세, 반전세, 임대, 공공임대, 장기전세, 행복주택 → "임대"
  - 분양, 청약, 매매, 내집마련 → "분양"
  - 고시원, 쉐어하우스, 원룸, 오피스텔 등 구체 주거 형태는 그대로 기록
  - 불명확 → null

- residence_requirement: 현재 거주 또는 이주 조건.
  - 현재 서울 거주, 서울 살고 있음, 서울 전입 가능 → "서울 거주자"
  - 서울로 이사 예정, 서울 직장/학교 예정 → "서울 이주 예정"
  - 타 지역 거주 명시 → "[지역] 거주자"
  - 불명확 → null

[사용자 의도 판별 예시]
- "관악구에서 자취하려는 사회초년생인데 전세대출 뭐 있어?" →
  category="금융", region="서울 관악구", target="청년", product_category="주택전세자금대출", housing_type="임대"
- "신혼부부가 들어갈 수 있는 서울 공공임대 알려줘" →
  category="주택", region="서울", target="신혼부부", housing_type="임대"
- "서울에서 받을 수 있는 청년 주거 정책 추천해줘" →
  category="주택,금융", region="서울", target="청년"

[반환 형식]
반드시 JSON 객체만 반환한다. 설명, 마크다운, 코드블록은 금지한다.
모든 필드를 포함한다.
{{
  "category": "주택|금융|주택,금융|null",
  "region": "지역 또는 null",
  "target": "청년|신혼부부|null",
  "product_category": "주택전세자금대출|월세지원|주택구입자금대출|이자지원|null",
  "special_condition": "조건 요약 또는 null",
  "housing_type": "임대|분양|구체 주거 형태|null",
  "residence_requirement": "거주/이주 조건 또는 null"
}}

사용자 발화: "{user_text}"
"""
    text = call(prompt)
    cleaned_text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned_text)

def decompose_query():
    prompt = f"""너는 서울시 청년 주거 정책 검색 쿼리 작성기다.
아래 사용자 의도 분석 결과를 바탕으로 벡터 검색에 사용할 쿼리 2개를 작성하라.

사용자 정보: {json.dumps(state, ensure_ascii=False)}

[작성 원칙]
- housing_query는 region, target, housing_type, special_condition을 반영한다.
- finance_query는 region, target, product_category, housing_type, special_condition을 반영한다.
- category가 "주택"이어도 finance_query는 보조적으로 작성하되, "금융 지원 가능성"을 넓게 검색한다.
- category가 "금융"이어도 housing_query는 사용자의 주거 상황과 맞는 주택 정책을 넓게 검색한다.
- 사용자가 말한 핵심 키워드(전세, 월세, 신혼부부, 사회초년생, 무주택 등)는 쿼리에 보존한다.
- 각 쿼리는 20~45자 사이의 자연스러운 한국어 검색 문장으로 작성한다.
- region이 "서울"이면 "서울시"를 포함한다.
- null 값은 억지로 언급하지 않는다.

반드시 JSON만 반환: {{"finance_query": "...", "housing_query": "..."}}"""
    text = call(prompt)
    cleaned_text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned_text)

def ask_missing(missing, current_state):
    known_info = {k: v for k, v in current_state.items() if v is not None}
    friendly_names = {
        "category": "문의 분류", "region": "관심 지역", "target": "현재 상황",
        "product_category": "금융 상품 종류", "special_condition": "우대 조건",
        "housing_type": "주택 형태", "residence_requirement": "거주 요건"
    }
    known_summary = ", ".join([f"{friendly_names.get(k, k)}: {v}" for k, v in known_info.items()]) or "없음"

    prompt = f"""너는 서울 청년 주거 정책 상담사다.
사용자에게 딱 하나의 질문만 해서 정책 추천을 바로 시작할 수 있도록 해야 한다.

[현재까지 파악된 정보]
{known_summary}

[아직 파악 안 된 핵심 정보]
{', '.join(missing)}

[작성 지침]
- 이미 파악된 정보가 있으면 자연스럽게 언급하며 맥락을 이어간다.
- 부족한 정보 중 정책 추천에 가장 큰 영향을 주는 한 가지를 우선 질문한다.
- 사용자의 상황을 묻는 경우 직업/가구/거주계획을 한 문장 안에서 답할 수 있게 묻는다.
- 사용자가 어떻게 답해야 할지 알 수 있도록 구체적인 예시를 괄호 안에 제시한다.
- 예시: "현재 어떤 상황이신가요? (예: 사회초년생으로 서울에서 첫 자취를 준비 중이에요)"
- 친근하고 간결하게, 2문장 이내로 작성
- JSON 필드명은 절대 언급하지 마

반드시 JSON으로만 반환: {{"msg": "질문 내용"}}"""

    try:
        result_json = json.loads(call(prompt))
        return result_json.get("msg", f"어떤 상황이신지 알려주시면 바로 맞춤 정책을 찾아드릴게요! (예: {missing[0]})")
    except:
        return "어떤 상황이신지 간단히 알려주시면 맞춤 정책을 바로 추천해드릴게요! (예: 사회초년생, 신혼부부 등)"

def ask_llm(user_text):
    extracted = extract_info(user_text)
    for k, v in extracted.items():
        if v and k in state:
            state[k] = v
    
    print(f"질문: {user_text}")
    print(f"[수집 상태] {state}")

    if is_pass():
        queries = decompose_query()
        return True, queries
    else:
        # 수정된 부분: state를 함께 전달
        msg = ask_missing(get_missing(), state) 
        return False, msg



# -==============================================================================================================
import re
from duckduckgo_search import DDGS

def get_web_search_url(policy_name):
    if not policy_name or policy_name == "주거 정책":
        return "https://www.myhome.go.kr/"

    # 1. 텍스트에서 키워드만 추출 (유사도 검사용)
    # 숫자(연도), 특수문자, 의미 없는 단어(모집, 공고 등)를 제거
    clean_name = re.sub(r'\d+', '', policy_name) # 숫자 제거
    clean_name = re.sub(r'[^\w\s]', ' ', clean_name) # 특수문자 제거
    
    # 2글자 이상의 핵심 키워드만 리스트로 만듦
    stop_words = ['모집', '공고', '안내', '수시', '입주자', '신청', '기준']
    keywords = [w for w in clean_name.split() if len(w) > 1 and w not in stop_words]
    
    search_query = " ".join(keywords) # 검색용 키워드 조합

    trusted_domains = ["go.kr", "or.kr", "seoul.go.kr", "i-sh.co.kr", "lh.or.kr"]
    
    try:
        with DDGS() as ddgs:
            # 검색어에 '공식 홈페이지'를 붙여 결과 10개를 가져옴
            results = list(ddgs.text(f"{search_query} 공식 홈페이지", max_results=10))
            
            best_res = None
            max_score = 0

            for res in results:
                title = res['title'].lower()
                url = res['href'].lower()
                
                # 유사도 점수 계산 로직
                current_score = 0
                for kw in keywords:
                    if kw in title: current_score += 1 # 제목에 키워드 포함 시 +1점
                
                # 가점: 공식 도메인인 경우 대폭 가산점 (+5점)
                if any(domain in url for domain in trusted_domains):
                    current_score += 5
                
                # 감점: 스팸 의심 단어 포함 시 감점 (-10점)
                if any(bad in title for bad in ["달력", "휴일", "운세"]):
                    current_score -= 10

                # 가장 높은 점수를 받은 결과 선택
                if current_score > max_score:
                    max_score = current_score
                    best_res = res['href']
            
            if best_res:
                return best_res

    except Exception as e:
        print(f"Search Error: {e}")
    
    return "https://www.myhome.go.kr/"