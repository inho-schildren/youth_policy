import json
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_template("""
너는 청년 주택 정책 전문가야. 아래 문서를 참고해서 질문에 답해줘.
문서에 없는 내용은 모른다고 해. 답변은 항목별로 정리해서 보기 좋게 작성해줘.

문서:
{context}

질문: {question}
""")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

def build_chain(retriever):
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain




client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

state = {
    "category": None, "region": None, "target": None,
    "product_category": None, "special_condition": None,
    "housing_type": None, "residence_requirement": None
}

REQ = {
    "common": ["category", "region", "target"],
    "finance": ["product_category", "special_condition"],
    "housing": ["housing_type", "residence_requirement"]
}

def is_pass():
    common_ok = all(state[k] for k in REQ["common"])
    finance_ok = any(state[k] for k in REQ["finance"])
    housing_ok = any(state[k] for k in REQ["housing"])
    return common_ok and finance_ok and housing_ok

def get_missing():
    # 기술 용어를 사용자 친화적인 말로 매핑
    friendly_names = {
        "category": "문의 분류",
        "region": "관심 지역",
        "target": "본인의 현재 상황(청년, 신혼부부 등)"
    }
    
    missing = [friendly_names.get(k, k) for k in REQ["common"] if not state[k]]
    
    if not any(state[k] for k in REQ["finance"]):
        missing.append("대출이나 금융 조건")
    if not any(state[k] for k in REQ["housing"]):
        missing.append("주택 형태나 거주 요건")
        
    return missing

def call(prompt):
    response = client.chat.completions.create(
        model="gpt-4o", 
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "당신은 JSON 형식으로 답변하도록 설계된 유용한 어시스턴트입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    # response.content는 이미 JSON 문자열입니다.
    return response.choices[0].message.content

def extract_info(user_text):
    prompt = f"""사용자 문장에서 아래 필드를 추출하라. 없으면 null.
필드: category(대출/주택), region(서울/경기/전국 등), target(청년/신혼부부 등),
product_category(주택전세자금대출/주택구입자금대출/기타), special_condition(횟수제한 등),
housing_type(임대/분양), residence_requirement(서울 거주자 등)

JSON만 반환.
문장: "{user_text}" """
    text = call(prompt)
    cleaned_text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned_text)

def decompose_query():
    prompt = f"""아래 정보로 금융쿼리와 주택쿼리를 자연어로 작성해라.
정보: {json.dumps(state, ensure_ascii=False)}
JSON만 반환: {{"finance_query": "...", "housing_query": "..."}}"""
    text = call(prompt)
    cleaned_text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned_text)

def ask_missing(missing, current_state):
    # 이미 알고 있는 정보만 필터링 (값이 None이 아닌 것들)
    known_info = {k: v for k, v in current_state.items() if v is not None}
    
    # 딕셔너리 키를 예쁜 이름으로 매핑 (get_missing에 있던 것과 동일하게 맞춤)
    friendly_names = {
        "category": "문의 분류", "region": "관심 지역", "target": "현재 상황",
        "product_category": "금융 상품 종류", "special_condition": "우대 조건",
        "housing_type": "주택 형태", "residence_requirement": "거주 요건"
    }
    
    # "문의 분류는 대출, 지역은 서울로 확인했습니다." 식의 요약 텍스트 생성
    summary_text = ", ".join([f"{friendly_names.get(k, k)}: {v}" for k, v in known_info.items()])

    prompt = f"""
    현재까지 사용자가 제공한 정보: {summary_text}
    
    부족한 정보: {', '.join(missing)}
    
    사용자에게 다음과 같이 말해줘:
    1. 먼저 위에서 정리한 '제공해주신 정보'를 요약해서 확인시켜줘.
    2. 그 다음 부족한 정보({', '.join(missing)})를 친절하게 물어봐.
    
    주의사항:
    - 괄호, JSON 필드명 같은 기술 용어 절대 금지.
    - 상담사가 말하듯이 자연스럽게.
    - 반드시 JSON 형식으로 'msg' 키에 담아서 반환해.
    """
    
    result_json = json.loads(call(prompt))
    return result_json.get("msg", f"제공해주신 내용은 잘 확인했습니다. 나머지 {', '.join(missing)}에 대해서도 알려주세요.")

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