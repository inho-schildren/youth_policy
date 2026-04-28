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
너는 청년 주거 및 금융 정책 전문가야.
아래 후보 정책들 중에서 사용자 질문에 가장 적합한 TOP3를 선정해줘.
문서에 없는 내용은 절대 추가하지 마.

후보 정책들:
{candidates}

질문: {query}

아래 JSON 형식으로만 출력해줘. 다른 텍스트 없이 JSON만 출력:
{{
    "top3": [
        {{
            "rank": 1,
            "type": "주거 or 금융",
            "title": "정책명",
            "reason": "선정 이유",
            "content": "핵심 내용"
        }}
    ]
}}
""")

# ── 보고서 프롬프트 ───────────────────────────────────────
report_prompt = ChatPromptTemplate.from_template("""
너는 청년 주택 정책 전문가야.
반드시 아래 TOP3 정책만을 근거로 보고서를 작성해줘.
문서에 없는 내용은 절대 추가하지 마.

사용자 질문: {query}
TOP3 정책: {top3_context}

아래 JSON 형식으로만 출력해줘. 다른 텍스트 없이 JSON만 출력:
{{
    "summary": "사용자 상황 요약 (2~3줄)",
    "metrics": {{
        "추천정책수": "3개",
        "주거정책": "X개",
        "금융정책": "X개",
        "신청가능": "즉시 가능"
    }},
    "policy_analysis": [
        {{
            "title": "정책명",
            "type": "주거 or 금융",
            "core": "핵심 내용",
            "pros": "장점",
            "cons": "단점 및 유의사항"
        }}
    ],
    "combination": "정책 조합 전략",
    "risks": "주의사항 및 리스크",
    "recommendation": "종합 추천 및 행동 계획"
}}
""")

def build_chain(housing_retriever, finance_retriever):

    # ── 1. 후보군 수집 (주거 5개 + 금융 5개) ─────────────
    def get_candidates(inputs):
        query        = inputs["query"]
        housing_docs = housing_retriever.invoke(query)[:5]
        finance_docs = finance_retriever.invoke(query)[:5]

        # 후보군 텍스트 구성
        candidates = ""
        for i, doc in enumerate(housing_docs):
            candidates += f"[주거정책 {i+1}]\n{doc.page_content}\n\n"
        for i, doc in enumerate(finance_docs):
            candidates += f"[금융정책 {i+1}]\n{doc.page_content}\n\n"

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
            f"[{p['rank']}위 / {p['type']}] {p['title']}\n{p['content']}"
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
    prompt = f"""사용자 문장에서 서울 청년 주거 정책 추천에 필요한 정보를 최대한 추론하여 추출하라.
명시적 언급뿐 아니라 문맥에서 암묵적으로 파악할 수 있는 정보도 적극 추론할 것.

[추론 규칙]
- target: "취준생/구직중/백수" → "청년", "사회초년생/신입/직장초년생" → "청년", "대학생/대학원생/학생" → "청년", "갓 결혼/신혼/예비부부/결혼한 지 N년" → "신혼부부", "아이 있는/육아중" → "신혼부부"
- category: "전세/월세/보증금/대출/금리/이자/디딤돌/버팀목" 언급 시 "금융" 포함, "아파트/주택/임대/분양/청약/입주" 언급 시 "주택" 포함, 두 유형 모두 언급 시 "금융,주택"
- region: 서울 구/동 이름 언급 시 "서울 [구명]"으로 기록. 구체적 지역 언급 없어도 서울 관련 정책 문의면 "서울"로 추론. 타 지역 명시 시 해당 지역 기록
- housing_type: "전세/월세/임대/보증금/반전세" 언급 시 "임대", "분양/청약/내집마련/집 사고 싶다" 언급 시 "분양"
- product_category: "전세자금/보증금 대출/전세대출" → "주택전세자금대출", "주택구입/집 구입/내집마련 대출" → "주택구입자금대출"
- residence_requirement: "서울에 살고 있다/거주중/현재 서울" → "서울 거주자", 타 지역 거주 언급 시 해당 지역
- special_condition: 특수 조건(장애인, 한부모, 다자녀, 저소득 등) 언급 시 기록
- 명확히 추론 불가한 경우만 null

[반환 필드]
category, region, target, product_category, special_condition, housing_type, residence_requirement

JSON만 반환.
문장: "{user_text}"
"""
    text = call(prompt)
    cleaned_text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned_text)

def decompose_query():
    prompt = f"""아래 사용자 정보를 바탕으로 주거 정책 검색에 사용할 쿼리 2개를 작성해라.

사용자 정보: {json.dumps(state, ensure_ascii=False)}

[작성 원칙]
- housing_query: 사용자의 target, region, housing_type을 반영한 주거 정책 검색 쿼리 (임대/분양 구분 포함)
- finance_query: 사용자의 target, product_category, income 수준을 반영한 금융 지원 검색 쿼리 (대출/보조금 포함)
- 각 쿼리는 30자 내외의 자연스러운 한국어 문장으로 작성
- region이 "서울"이면 "서울시" 또는 "서울 거주" 표현 포함

JSON만 반환: {{"finance_query": "...", "housing_query": "..."}}"""
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

    prompt = f"""너는 서울 청년 주거 정책 상담사야. 사용자에게 딱 하나의 질문만 해서 정책 추천을 바로 시작할 수 있도록 해야 해.

[현재까지 파악된 정보]
{known_summary}

[아직 파악 안 된 핵심 정보]
{', '.join(missing)}

[작성 지침]
- 파악된 정보가 있으면 자연스럽게 언급하며 맥락을 이어가
- 모르는 정보를 한 문장으로 물어봐 (여러 항목이어도 하나의 자연스러운 질문으로 통합)
- 사용자가 어떻게 답해야 할지 알 수 있도록 구체적인 예시를 괄호 안에 제시
- 예시: "어떤 상황이신가요? (예: 사회초년생으로 서울에서 첫 자취를 준비 중이에요)"
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