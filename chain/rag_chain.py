import json
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    streaming=True,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# ── 1. TOP3 선정 프롬프트 ────────────────────────────────────
top3_prompt = ChatPromptTemplate.from_template("""
너는 서울시 청년 주거·금융 정책 추천 전문가다.
아래 후보 정책들 중 사용자 질문의 의도와 상황에 가장 잘 맞는 TOP3를 선정하라.

[후보 정책들]
{candidates}

[사용자 질문]
{query}

[선정 원칙]
1. 사용자 상황과 가장 잘 맞는 정책 3개를 엄선한다.
2. 이유(reason)와 내용(content)은 각각 **단 1문장**으로 매우 짧게 핵심만 쓴다.
3. URL은 제공된 정보가 없으면 '정보없음'으로 표시하거나, 확실한 공식 도메인(lh.or.kr 등)만 짧게 기재한다.

아래 JSON 형식으로만 출력하라:
{{
    "top3": [
        {{
            "rank": 1,
            "type": "주거|금융",
            "title": "제목",
            "url": "URL",
            "reason": "1문장 이유",
            "content": "1문장 요약"
        }}
    ]
}}
""")

# ── 2. 보고서 프롬프트 ───────────────────────────────────────
report_prompt = ChatPromptTemplate.from_template("""
너는 선정된 TOP3 정책을 바탕으로 사용자에게 최적의 주거/금융 가이드를 제공하는 보고서 작성 전문가다.
아래 정보를 바탕으로 전문가 수준의 종합 보고서를 작성하라.

[선정된 TOP3 정책 정보]
{top3_context}

[사용자 상황 및 질문]
{query}

[작성 가이드라인]
1. summary: 사용자 상황을 요약하고 왜 이 정책들이 추천되었는지 3문장으로 설명하라.
2. policy_analysis: 각 정책별로 상세 분석을 제공하라.
   - title: 정책명
   - type: 주거 또는 금융
   - core: 해당 정책의 핵심 혜택 (1문장)
   - pros: 해당 정책의 장점 3가지를 리스트로 작성
   - cons: 해당 정책의 단점이나 주의사항 2가지를 리스트로 작성
3. combination: 선정된 정책들을 어떻게 조합하거나 어떤 순서로 신청하면 좋을지 전략을 제시하라.
4. risks: 신청 시 놓치기 쉬운 서류나 자격 요건 등 리스크를 설명하라.
5. recommendation: 마지막으로 사용자가 바로 실행해야 할 Action Plan을 제시하라.

아래 JSON 형식으로만 출력하라:
{{
    "summary": "종합 요약",
    "policy_analysis": [
        {{
            "title": "정책명",
            "type": "주거|금융",
            "core": "핵심 혜택",
            "pros": ["장점1", "장점2", "장점3"],
            "cons": ["주의사항1", "주의사항2"]
        }}
    ],
    "combination": "정책 조합 전략 및 순서",
    "risks": ["리스크1", "리스크2"],
    "recommendation": "종합 추천 및 행동 계획"
}}
""")

def build_chain(housing_retriever, finance_retriever):

    def get_candidates(inputs):
        query = inputs["query"]
        with ThreadPoolExecutor(max_workers=2) as executor:
            h_future = executor.submit(housing_retriever.invoke, query)
            f_future = executor.submit(finance_retriever.invoke, query)
            h_docs = h_future.result()[:3]
            f_docs = f_future.result()[:3]

        candidates = ""
        for i, doc in enumerate(h_docs):
            m = doc.metadata
            candidates += (
                f"[주거정책 {i+1}]\n제목: {m.get('title','')}\n"
                f"내용: {doc.page_content[:300]}\n\n"
            )
        for i, doc in enumerate(f_docs):
            m = doc.metadata
            candidates += (
                f"[금융정책 {i+1}]\n제목: {m.get('title','')}\n"
                f"내용: {doc.page_content[:300]}\n\n"
            )
        print("\n=== [DEBUG] AI에게 전달되는 후보 문서 정보 ===")
        print(candidates)
        print("==========================================\n")
        return {"query": query, "candidates": candidates}

    top3_chain = (
        RunnableLambda(get_candidates)
        | top3_prompt
        | llm
        | StrOutputParser()
    )

    report_chain = (
        report_prompt
        | llm
        | StrOutputParser()
    )

    return top3_chain, report_chain

# ── 3. 의도 분석 및 질문 생성 (기존 함수들 유지) ─────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_user_intent(query, current_state):
    prompt = f"""
    너는 청년 주거 정책 상담원이다. 사용자의 질문을 분석하여 정보를 추출하고 정책 검색용 쿼리를 생성하라.
    
    [현재까지 파악된 정보]
    {json.dumps(current_state, ensure_ascii=False)}
    
    [사용자 질문]
    {query}
    
    아래 JSON 형식으로만 응답하라:
    {{
        "extracted": {{
            "category": "금융|주거|미지정",
            "region": "서울|경기|...|미지정",
            "target": "청년|신혼부부|...|미지정",
            "product_category": "대출|전세|...|미지정"
        }},
        "is_pass": true|false,
        "queries": {{
            "housing_query": "주거 정책 검색용 쿼리",
            "finance_query": "금융 정책 검색용 쿼리"
        }}
    }}
    - 필수 정보가 충분하면 is_pass를 true로 설정하라.
    """
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return json.loads(res.choices[0].message.content)

def ask_missing_info(missing_fields, current_state):
    prompt = f"사용자에게 {missing_fields} 정보를 정중하게 물어보는 메시지를 작성하라."
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return res.choices[0].message.content

def get_web_search_url(title):
    return f"https://www.google.com/search?q={title}"