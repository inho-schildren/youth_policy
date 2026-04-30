import os
import json
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from datasets import Dataset
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from core.embedder_vectorstore import get_openai_embedder_small
from config import FINANCE_DOCS_PATH, DOCS_PATH

# ── LangSmith 설정 ─────────────────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = "TRUE"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "youth_policy")

# ── 경로 설정 ──────────────────────────────────────────────────
CHROMA_DIR   = "./db/chroma_experiment"
EVAL_PATH    = "./evaluator/eval_all.json"
RESULT_DIR   = "./evaluator/ragas_results"
os.makedirs(RESULT_DIR, exist_ok=True)

# ── 평가할 컬렉션 ──────────────────────────────────────────────
FINANCE_COLLECTIONS = [
    "finance_recursive_size1200",
    "finance_character_size800",
    # "finance_recursive_size500",
]
HOUSING_COLLECTIONS = [
    # "housing_recursive_size300",
    "housing_semantic_t85",
    "housing_recursive_size800",
]

# ── 프롬프트 ───────────────────────────────────────────────────
PROMPT = PromptTemplate.from_template("""당신은 서울시 청년 주거 정책 전문 상담사입니다.
아래 검색된 정책 문서를 바탕으로 질문에 답변해주세요.
문서에 없는 내용은 답변하지 마세요. 반드시 출처 정책명을 포함해서 답변하세요.

#검색된 문서:
{context}

#질문:
{question}

#답변:""")

# ── 문서 로드 ──────────────────────────────────────────────────
def load_json_docs(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_reference_contexts(source: str, domain: str, finance_docs: list, housing_docs: list) -> list:
    docs = finance_docs if domain == "finance" else housing_docs
    contexts = [
        doc["page_content"]
        for doc in docs
        if doc["metadata"].get("title", "") == source
    ]
    return contexts[:3] if contexts else ["관련 문서 없음"]

# ── 컬렉션별 평가 ──────────────────────────────────────────────
def evaluate_collection(collection_name: str, eval_set: list,
                        finance_docs: list, housing_docs: list,
                        embedder, llm) -> pd.DataFrame:

    domain = "finance" if collection_name.startswith("finance") else "housing"
    domain_eval = [item for item in eval_set if item["domain"] == domain]

    print(f"\n{'='*50}")
    print(f"📊 {collection_name} 평가 중... ({len(domain_eval)}개 질문)")
    print(f"{'='*50}")

    # 컬렉션 로드
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
        embedding_function=embedder
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # RAG 체인
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )

    questions        = []
    answers          = []
    retrieved_ctxs   = []
    reference_ctxs   = []

    for item in domain_eval:
        q      = item["question"]
        source = item["source"]

        # 검색
        docs_found = retriever.invoke(q)
        ret_ctx    = [doc.page_content for doc in docs_found]

        # 답변 생성
        answer = chain.invoke(q)

        # reference_contexts
        ref_ctx = get_reference_contexts(source, domain, finance_docs, housing_docs)

        questions.append(q)
        answers.append(answer)
        retrieved_ctxs.append(ret_ctx)
        reference_ctxs.append(ref_ctx)

        print(f"  ✅ Q: {q[:40]}...")

    # RAGAS Dataset 구성
    ragas_data = Dataset.from_dict({
        "user_input":          questions, # 질문
        "response":            answers, # LLM 답변
        "retrieved_contexts":  retrieved_ctxs, # RAG가 검색한 것
        "reference":  [ctx[0] if ctx else "" for ctx in reference_ctxs], # 정답 문서 텍스트
    })

    # RAGAS 평가
    ragas_llm        = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    result = evaluate(
        dataset=ragas_data,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    result_df = result.to_pandas()
    result_df["collection"] = collection_name

    # 저장
    save_path = os.path.join(RESULT_DIR, f"{collection_name}.csv")
    result_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  💾 저장 완료: {save_path}")

    return result_df


if __name__ == "__main__":

    print("\n📂 데이터 로드 중...")
    eval_set     = json.load(open(EVAL_PATH, encoding="utf-8"))
    finance_docs = load_json_docs(FINANCE_DOCS_PATH)
    housing_docs = load_json_docs(DOCS_PATH)
    print(f"  테스트셋: {len(eval_set)}개")

    embedder = get_openai_embedder_small()
    llm      = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    all_results = []

    # Finance 평가
    print("\n\n🏦 Finance 컬렉션 평가 시작")
    for col in FINANCE_COLLECTIONS:
        df = evaluate_collection(col, eval_set, finance_docs, housing_docs, embedder, llm)
        all_results.append(df)

    # Housing 평가
    print("\n\n🏠 Housing 컬렉션 평가 시작")
    for col in HOUSING_COLLECTIONS:
        df = evaluate_collection(col, eval_set, finance_docs, housing_docs, embedder, llm)
        all_results.append(df)

    # ── 최종 비교표 ────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("📊 RAGAS 최종 비교 결과")
    print(f"{'='*60}")

    METRICS = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]

    print(f"\n{'컬렉션':<35} {'CP':>6} {'CR':>6} {'FF':>6} {'AR':>6} {'avg':>6}")
    print("-" * 65)

    summary = []
    for df in all_results:
        col  = df["collection"].iloc[0]
        scores = {m: df[m].mean() for m in METRICS}
        avg  = sum(scores.values()) / len(scores)
        summary.append({"collection": col, **scores, "average": avg})
        print(f"  {col:<33} {scores['context_precision']:>6.2f} {scores['context_recall']:>6.2f} "
              f"{scores['faithfulness']:>6.2f} {scores['answer_relevancy']:>6.2f} {avg:>6.2f}")

    # 전체 요약 저장
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(RESULT_DIR, "ragas_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 요약 저장 완료: {summary_path}")