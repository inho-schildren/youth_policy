import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from core.retriever import get_basic_retriever

# =========================
# 환경 변수
# =========================
load_dotenv()

# =========================
# 경로 설정
# =========================
BASE_DIR = Path("experiments")

DATASET_PATH = BASE_DIR / "datasets" / "housing_qa_dataset_upgrade.json"
VECTORSTORE_BASE_DIR = BASE_DIR / "vectorstores"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# ✅ 수정: 빠른 1차 평가 옵션
# =========================
RUN_TAG = "retrieval_fast"   # 결과 파일명 구분용
MAX_DATASET_ROWS = 10        # 전체 dataset 중 앞 10개만 사용
TOP_K = 3                    # 검색할 chunk 개수
SKIP_LLM_ANSWER = True       # 1차 평가는 LLM 답변 생성 생략

# =========================
# 모델 설정
# =========================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# =========================
# vectorstore 자동 로드
# =========================
VECTORSTORE_CONFIGS = []

for folder in VECTORSTORE_BASE_DIR.iterdir():
    if folder.is_dir():
        chroma_path = folder / "openai_chroma"

        if chroma_path.exists():
            VECTORSTORE_CONFIGS.append({
                "name": folder.name,
                "path": chroma_path
            })
        else:
            print(f"⚠️ openai_chroma 없음 → 스킵: {folder}")

# =========================
# dataset 로드
# =========================
def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# =========================
# vectorstore 로드
# =========================
def load_chroma_vectorstore(vectorstore_path, collection_name):
    return Chroma(
        persist_directory=str(vectorstore_path),
        collection_name=collection_name,
        embedding_function=embedding
    )

# =========================
# RAG 답변 생성
# =========================
def generate_rag_answer(question, contexts):
    context_text = "\n\n".join(contexts)

    prompt = f"""
너는 청년 주거 정책 문서를 기반으로 답변하는 RAG 챗봇이다.

[답변 규칙]
- 반드시 제공된 context만 기반으로 답변해라.
- context에 없는 내용은 추측하지 말고 "문서에서 확인되지 않습니다"라고 답해라.
- 간결하고 이해하기 쉽게 답변해라.

[context]
{context_text}

[질문]
{question}
"""

    response = llm.invoke(prompt)
    return response.content.strip()

# =========================
# RAG 실행
# =========================
def run_rag_for_vectorstore(dataset, vectorstore, k=TOP_K):
    retriever = get_basic_retriever(vectorstore)

    results = []

    for idx, item in enumerate(dataset):
        question = item["question"]

        print(f"  - 질문 {idx + 1}/{len(dataset)}")

        retrieved_docs = retriever.invoke(question)

        # ✅ 수정: TOP_K만 사용
        retrieved_docs = retrieved_docs[:k]

        contexts = [doc.page_content for doc in retrieved_docs]

        # ✅ 수정: 1차 평가는 LLM 답변 생성 생략
        if SKIP_LLM_ANSWER:
            answer = item.get("ground_truth", "")
        else:
            answer = generate_rag_answer(question, contexts)

        results.append({
            "question": question,
            "ground_truth": item.get("ground_truth", ""),
            "answer": answer,
            "contexts": contexts,
            "evidence": item.get("evidence", ""),      # ✅ 수정: evidence 보존
            "metadata": item.get("metadata", {}),
            "top_k": k,                                # ✅ 수정: 실험 조건 기록
            "skip_llm_answer": SKIP_LLM_ANSWER         # ✅ 수정: 실험 조건 기록
        })

    return results

# =========================
# 결과 저장
# =========================
def save_results(results, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

# =========================
# 실행
# =========================
if __name__ == "__main__":
    dataset = load_dataset(DATASET_PATH)

    # ✅ 수정: 빠른 1차 평가용 dataset 개수 제한
    if MAX_DATASET_ROWS is not None:
        dataset = dataset[:MAX_DATASET_ROWS]

    print(f"\n📊 평가 데이터셋 개수: {len(dataset)}")

    for config in VECTORSTORE_CONFIGS:
        name = config["name"]
        path = config["path"]

        # ✅ 수정: RUN_TAG로 기존 결과 파일과 충돌 방지
        save_path = RESULTS_DIR / f"rag_result_{RUN_TAG}_{name}.json"

        if save_path.exists():
            print(f"⏭️ 이미 완료됨 → 스킵: {name}")
            continue

        print(f"\n🚀 실행 시작: {name}")

        if not path.exists():
            print(f"❌ 경로 없음: {path}")
            continue

        vectorstore = load_chroma_vectorstore(
            vectorstore_path=path,
            collection_name=name
        )

        results = run_rag_for_vectorstore(
            dataset=dataset,
            vectorstore=vectorstore,
            k=TOP_K
        )

        save_results(results, save_path)

        print(f"✅ 저장 완료: {save_path}")

    print("\n🎉 모든 실험 완료")