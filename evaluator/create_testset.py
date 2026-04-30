import json
import os
import time
from openai import OpenAI
from config import OPENAI_API_KEY, FINANCE_DOCS_PATH, DOCS_PATH

client = OpenAI(api_key=OPENAI_API_KEY)

# ── 질문 유형 정의 ──────────────────────────────────────────────
# 실제 청년들이 검색할 법한 5가지 유형
QUESTION_TYPES = [
    "자격조건",   # "나 해당되는지" 류
    "금액/한도",  # "얼마나 받을 수 있는지" 류
    "신청방법",   # "어떻게 신청하는지" 류
    "기간/일정",  # "언제까지 신청해야 하는지" 류
    "비교/추천",  # "이거랑 저거 중 뭐가 나은지" 류
]

# ── 프롬프트 ────────────────────────────────────────────────────
SYSTEM_PROMPT = """당신은 서울에 거주하는 다양한 상황의 청년입니다.
아래 페르소나 중 하나를 랜덤으로 골라서 그 사람 입장에서 질문을 만들어주세요.

페르소나 목록:
- 23살 대학생, 서울 성북구 거주, 월세가 너무 부담됨
- 29살 직장인, 서울 마포구 거주, 전세로 이사하고 싶음
- 32살 신혼부부, 서울에서 살고 싶은데 집값이 너무 비쌈
- 27살 취준생, 보증금도 없고 월세도 빠듯함
- 35살 한부모가족, 서울 거주중, 아이랑 살 집이 필요함
- 26살 사회초년생, 월급 230만원, 서울 첫 독립 준비중
- 31살 무주택자, 서울 금천구, 내집마련 꿈꾸는 중

규칙:
1. 자신의 상황을 설명하면서 특정 정책 유형을 언급하는 방식으로 질문
   (예: "월세 지원", "전세자금 대출", "공공임대", "행복주택" 등 키워드 포함)
2. 딱딱한 공문서 말투 금지
3. 구어체/일상어 사용
4. 구체적인 상황 포함 (나이, 소득, 지역 등)
5. 반드시 JSON 형식으로만 응답
"""

def make_user_prompt(policy_text: str, policy_title: str, q_type: str, domain: str) -> str:
    return f"""아래 정책 내용을 바탕으로, 이 정책을 찾고 있는 청년이 자신의 상황과 함께 정책 유형을 언급하며 물어볼 법한 "{q_type}" 관련 질문 1개와 그 정답을 만들어주세요.

중요: 
- 질문에 정책의 핵심 키워드를 자연스럽게 포함시켜주세요 (예: "월세 지원", "전세자금", "공공임대", "행복주택" 등)
- 정확한 정책명은 언급하지 마세요
- 자신의 상황(나이, 소득, 지역 등)을 같이 설명해주세요

정책명: {policy_title}
정책 내용:
{policy_text[:2000]}

도메인: {domain} 정책

반드시 아래 JSON 형식으로만 응답하세요:
{{
  "question": "자연스러운 질문 (구어체, 정책 키워드 포함, 정확한 정책명 제외)",
  "answer": "정책 내용 기반 정확한 답변 (2-3문장)",
  "source": "{policy_title}",
  "domain": "{domain}",
  "question_type": "{q_type}"
}}"""


def load_documents(path: str) -> list:
    """저장된 documents JSON 로드"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_policy_info(doc: dict) -> tuple[str, str]:
    """document에서 제목과 본문 추출"""
    # LangChain Document 형식 처리
    if isinstance(doc, dict):
        content = doc.get("page_content", doc.get("text", ""))
        metadata = doc.get("metadata", {})
        title = metadata.get("title", metadata.get("source", "정책문서"))
    else:
        content = str(doc)
        title = "정책문서"
    return title, content


def generate_question(policy_title: str, policy_text: str, q_type: str, domain: str) -> dict | None:
    """LLM으로 질문 1개 생성"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_user_prompt(
                    policy_text, policy_title, q_type, domain
                )}
            ],
            temperature=0.8,  # 다양한 질문을 위해 약간 높게
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"  ⚠️ 생성 실패 ({policy_title} / {q_type}): {e}")
        return None


def create_eval_set(docs_path: str, domain: str, target_count: int = 25) -> list:
    print(f"\n{'='*50}")
    print(f"📝 {domain} 테스트셋 생성 시작 (목표: {target_count}개)")
    print(f"{'='*50}")

    docs = load_documents(docs_path)
    print(f"  로드된 문서 수: {len(docs)}개")

    # 문서별 고유 정책 추출 (중복 제거)
    policies = {}
    for doc in docs:
        title, content = extract_policy_info(doc)
        if title not in policies:
            policies[title] = content
        else:
            policies[title] += "\n" + content  # 같은 정책 내용 합치기

    policy_list = list(policies.items())
    print(f"  고유 정책 수: {len(policy_list)}개")

    eval_set = []
    q_type_cycle = QUESTION_TYPES * (target_count // len(QUESTION_TYPES) + 1)

    for i, (title, content) in enumerate(policy_list):
        if len(eval_set) >= target_count:
            break

        q_type = q_type_cycle[i % len(QUESTION_TYPES)]
        print(f"  [{len(eval_set)+1}/{target_count}] {title[:30]}... | {q_type}")

        qa = generate_question(title, content, q_type, domain)
        if qa:
            qa["id"] = f"{domain}_{len(eval_set)+1:03d}"
            eval_set.append(qa)
            print(f"    Q: {qa['question'][:50]}...")

        time.sleep(0.5)  # API 레이트 리밋 방지

    # 부족하면 중요 정책에서 추가 생성
    if len(eval_set) < target_count:
        print(f"\n  ⚡ {target_count - len(eval_set)}개 추가 생성 중...")
        for i, (title, content) in enumerate(policy_list):
            if len(eval_set) >= target_count:
                break
            q_type = q_type_cycle[(i + 2) % len(QUESTION_TYPES)]  # 다른 유형으로
            qa = generate_question(title, content, q_type, domain)
            if qa:
                qa["id"] = f"{domain}_{len(eval_set)+1:03d}"
                eval_set.append(qa)
            time.sleep(0.5)

    print(f"\n  ✅ {domain} 완료: {len(eval_set)}개 생성")
    return eval_set


def save_eval_set(eval_set: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, ensure_ascii=False, indent=2)
    print(f"  💾 저장 완료: {path}")


def preview_eval_set(eval_set: list, n: int = 3):
    """생성된 테스트셋 미리보기"""
    print(f"\n{'='*50}")
    print("📋 샘플 미리보기")
    print(f"{'='*50}")
    for item in eval_set[:n]:
        print(f"\n[{item['id']}] [{item['question_type']}]")
        print(f"Q: {item['question']}")
        print(f"A: {item['answer'][:100]}...")
        print(f"출처: {item['source']}")


# ── 메인 실행 ───────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. 금융 정책 테스트셋
    finance_eval = create_eval_set(
        docs_path=FINANCE_DOCS_PATH,  # data/finance_documents.json
        domain="finance",
        target_count=25
    )
    save_eval_set(finance_eval, "./evaluator/eval_finance.json")
    preview_eval_set(finance_eval)

    # 2. 주거 정책 테스트셋
    housing_eval = create_eval_set(
        docs_path=DOCS_PATH,          # data/output_v2.json
        domain="housing",
        target_count=25
    )
    save_eval_set(housing_eval, "./evaluator/eval_housing.json")
    preview_eval_set(housing_eval)

    # 3. 통합 저장 (전체 50개)
    all_eval = finance_eval + housing_eval
    save_eval_set(all_eval, "./evaluator/eval_all.json")

    print(f"\n{'='*50}")
    print(f"🎉 최종 완료: 총 {len(all_eval)}개")
    print(f"  - finance: {len(finance_eval)}개 → evaluator/eval_finance.json")
    print(f"  - housing: {len(housing_eval)}개 → evaluator/eval_housing.json")
    print(f"  - 전체:    {len(all_eval)}개  → evaluator/eval_all.json")
    print(f"{'='*50}")