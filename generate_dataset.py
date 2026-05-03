import json
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_openai import ChatOpenAI

# -----------------------------
# 0. 환경 설정
# -----------------------------
load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# -----------------------------
# 1. Document 로드
# -----------------------------
def load_documents(path: str) -> List[Document]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        Document(
            page_content=item["page_content"],
            metadata=item.get("metadata", {})
        )
        for item in data
        if str(item.get("page_content", "")).strip()
    ]


# -----------------------------
# 2. JSON 추출 유틸
# -----------------------------
def extract_json(text: str):
    text = text.strip()

    # ```json ... ``` 제거
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 배열 또는 객체만 추출 시도
    match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    raise ValueError("JSON 파싱 실패")


# -----------------------------
# 3. 질문 정제
# -----------------------------
def clean_question(q: str) -> str:
    q = str(q).strip()

    q = re.sub(r"^\s*[-*•]\s*", "", q)
    q = re.sub(r"^\s*\d+\s*[\.\)\-]\s*", "", q)

    return q.strip()


# -----------------------------
# 4. 답변 불가 / 품질 낮은 답변 필터
# -----------------------------
def is_bad_answer(answer: str) -> bool:
    answer = str(answer).strip()

    if not answer:
        return True

    bad_patterns = [
        "문서에서 확인되지 않습니다",
        "문서에 명시되어 있지 않습니다",
        "문서에는 해당 내용이 없습니다",
        "제공된 문서에는",
        "확인할 수 없습니다",
        "알 수 없습니다",
        "추측할 수 없습니다",
        "답변할 수 없습니다",
        "정보가 없습니다",
        "포함되어 있지 않습니다",
    ]

    return any(pattern in answer for pattern in bad_patterns)


def is_bad_question(question: str) -> bool:
    question = str(question).strip()

    if len(question) < 10:
        return True

    bad_patterns = [
        "어떤 조건이 필요한가요",
        "어떻게 신청하나요",
        "무엇인가요",
        "알려주세요",
    ]

    return any(pattern in question for pattern in bad_patterns)


# -----------------------------
# 5. 질문 생성
# -----------------------------
def generate_questions(context: str) -> List[Dict[str, str]]:
    prompt = f"""
다음 문서를 기반으로 총 3개의 질문을 생성해줘.

[목표]
실제 사람이 검색할 때 입력할 법한 구체적인 질문 생성

[난이도]
- easy 1개
- medium 1개
- hard 1개

[절대 규칙]
- 질문은 반드시 문서 안에서 답할 수 있어야 함
- 질문은 반드시 구체적인 대상/조건을 포함해야 함
- 질문만 보고도 의미가 명확해야 함
- 문서에 없는 내용을 묻는 질문 금지
- 너무 포괄적인 질문 금지

[좋은 예]
- 서대문구 꿈꾸는 다락방에 신청하려면 어떤 대학생이 대상인가요?
- 서울금천 행복주택 산업단지근로자로 신청하려면 근무지가 어떤 지역이어야 하나요?
- 국민임대주택 신청 시 금융정보 등 제공 동의서를 제출하지 않으면 어떤 불이익이 있나요?

[출력 형식]
반드시 JSON 배열만 출력해.
설명 문장, 마크다운, 코드블록은 출력하지 마.

[
  {{
    "difficulty": "easy",
    "question": "질문"
  }},
  {{
    "difficulty": "medium",
    "question": "질문"
  }},
  {{
    "difficulty": "hard",
    "question": "질문"
  }}
]

[문서]
{context[:5000]}
"""

    response = llm.invoke(prompt)

    try:
        data = extract_json(response.content)
    except Exception as e:
        print("⚠️ 질문 JSON 파싱 실패:", e)
        print(response.content[:500])
        return []

    questions = []

    if not isinstance(data, list):
        return []

    for item in data:
        if not isinstance(item, dict):
            continue

        difficulty = str(item.get("difficulty", "")).strip()
        question = clean_question(item.get("question", ""))

        if difficulty not in ["easy", "medium", "hard"]:
            difficulty = "unknown"

        if is_bad_question(question):
            continue

        questions.append({
            "difficulty": difficulty,
            "question": question
        })

    return questions[:3]


# -----------------------------
# 6. 정답 + 근거 생성
# -----------------------------
def generate_answer_and_evidence(context: str, question: str) -> Dict[str, str]:
    prompt = f"""
다음 문서를 기반으로 질문에 답변해줘.

[조건]
- 반드시 문서 기반으로만 답변
- 없는 내용은 추측 금지
- 답변은 간결하지만 구체적으로 작성
- evidence는 답변의 근거가 되는 원문 문장을 문서에서 그대로 가져와야 함
- evidence는 문서에 실제로 존재하는 문장이어야 함
- 문서에서 답할 수 없으면 answer와 evidence를 빈 문자열로 출력

[출력 형식]
반드시 JSON 객체만 출력해.
설명 문장, 마크다운, 코드블록은 출력하지 마.

{{
  "answer": "정답",
  "evidence": "문서 원문 근거"
}}

[문서]
{context[:5000]}

[질문]
{question}
"""

    response = llm.invoke(prompt)

    try:
        data = extract_json(response.content)
    except Exception as e:
        print("⚠️ 정답 JSON 파싱 실패:", e)
        print(response.content[:500])
        return {
            "answer": "",
            "evidence": ""
        }

    if not isinstance(data, dict):
        return {
            "answer": "",
            "evidence": ""
        }

    return {
        "answer": str(data.get("answer", "")).strip(),
        "evidence": str(data.get("evidence", "")).strip()
    }


# -----------------------------
# 7. Evidence 검증
# -----------------------------
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def is_valid_evidence(context: str, evidence: str) -> bool:
    evidence = normalize_text(evidence)
    context = normalize_text(context)

    if not evidence:
        return False

    # 너무 짧은 근거는 신뢰 낮음
    if len(evidence) < 15:
        return False

    # 원문 그대로 포함되어 있는지 확인
    if evidence in context:
        return True

    # LLM이 줄바꿈/공백만 조금 바꾼 경우 대비
    evidence_compact = re.sub(r"\s+", "", evidence)
    context_compact = re.sub(r"\s+", "", context)

    return evidence_compact in context_compact


# -----------------------------
# 8. Dataset 생성
# -----------------------------
def build_dataset(documents: List[Document]) -> List[Dict[str, Any]]:
    dataset = []
    skipped = 0

    for idx, doc in enumerate(documents):
        context = doc.page_content

        print(f"\n📄 Doc {idx + 1}/{len(documents)} 질문 생성 중...")

        question_items = generate_questions(context)

        if not question_items:
            print("⚠️ 질문 생성 실패 → 스킵")
            skipped += 1
            continue

        added_count = 0

        for item in question_items:
            question = item["question"]
            difficulty = item["difficulty"]

            result = generate_answer_and_evidence(context, question)

            answer = result["answer"]
            evidence = result["evidence"]

            if is_bad_answer(answer):
                print("⚠️ 답변 불가/품질 낮음 → 제외:", question)
                skipped += 1
                continue

            if not is_valid_evidence(context, evidence):
                print("⚠️ evidence 검증 실패 → 제외:", question)
                skipped += 1
                continue

            dataset.append({
                "question": question,
                "ground_truth": answer,
                "evidence": evidence,
                "difficulty": difficulty,
                "metadata": doc.metadata
            })

            added_count += 1

        print(f"✅ Doc {idx + 1} 완료: {added_count}개 추가")

    print(f"\n📊 생성 완료: {len(dataset)}개 / 제외 {skipped}개")

    return dataset


# -----------------------------
# 9. 저장
# -----------------------------
def save_dataset(dataset: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)


# -----------------------------
# 10. 실행
# -----------------------------
if __name__ == "__main__":
    DOC_PATH = "experiments/_base/housing_docs.json"
    SAVE_PATH = "experiments/datasets/housing_qa_dataset_upgrade.json"

    documents = load_documents(DOC_PATH)

    # 테스트용
    # documents = documents[:3]

    dataset = build_dataset(documents)

    save_dataset(dataset, SAVE_PATH)

    print(f"\n🎉 총 {len(dataset)}개 데이터셋 저장 완료")
    print(f"💾 저장 위치: {SAVE_PATH}")