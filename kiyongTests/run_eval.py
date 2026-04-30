"""
실행: langchain_v0_basics 루트에서
    python youth_policy/kiyongTests/run_eval.py
"""
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from datasets import Dataset
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

PERSIST_DIR = "./db/chroma_finance"
COLLECTION = "youth_finance_policy"
EMBED = "text-embedding-3-small"
MODEL = "gpt-4o-mini"

# 테스트셋 로드
df = pd.read_csv("./kiyongTests/testset.csv")
TESTS = [{"question": r["user_input"], "ground_truth": r["reference"]} for _, r in df.iterrows()]

# RAG 준비
llm = ChatOpenAI(model=MODEL, temperature=0)
emb = OpenAIEmbeddings(model=EMBED)
ragas_llm = LangchainLLMWrapper(llm)
ragas_emb = LangchainEmbeddingsWrapper(emb)

# ── 벡터스토어 & 공통 리소스 ──────────────────────────────────────────────────
vectorstore = Chroma(collection_name=COLLECTION, persist_directory=PERSIST_DIR, embedding_function=emb)
docs_for_bm25 = [Document(page_content=d) for d in vectorstore.get()["documents"]]

# ── 리트리버 정의 ─────────────────────────────────────────────────────────────
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
bm25_retriever.k = 4

ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.2, 0.8],
)

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_retriever,
    llm=llm,
)

RETRIEVERS = {
    "vector":     vector_retriever,
    "ensemble":   ensemble_retriever,
    "multiquery": multiquery_retriever,
}

# ── 평가 메트릭 ───────────────────────────────────────────────────────────────
METRICS = [
    Faithfulness(llm=ragas_llm),
    AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
    ContextPrecision(llm=ragas_llm),
    ContextRecall(llm=ragas_llm),
]

# ── 리트리버별 RAG 실행 & 평가 ────────────────────────────────────────────────
summary_rows = []

for name, retriever in RETRIEVERS.items():
    print(f"\n{'='*50}")
    print(f"▶ [{name}] 리트리버 평가 시작")
    print(f"{'='*50}")

    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for t in TESTS:
        ctx = [d.page_content for d in retriever.invoke(t["question"])]
        ans = llm.invoke(f"컨텍스트:\n{chr(10).join(ctx)}\n\n질문: {t['question']}").content
        rows["question"].append(t["question"])
        rows["answer"].append(ans)
        rows["contexts"].append(ctx)
        rows["ground_truth"].append(t["ground_truth"])

    result = evaluate(Dataset.from_dict(rows), metrics=METRICS)
    result_df = result.to_pandas()

    # 개별 결과 저장
    out_path = f"./kiyongTests/eval_{name}.csv"
    result_df.to_csv(out_path, index=False)
    print(f"  저장 완료: {out_path}")
    print(result)

    # 요약용 평균 수집
    summary_rows.append({
        "retriever":        name,
        "faithfulness":     result_df["faithfulness"].mean(),
        "answer_relevancy": result_df["answer_relevancy"].mean(),
        "context_precision":result_df["context_precision"].mean(),
        "context_recall":   result_df["context_recall"].mean(),
    })

# ── 비교 요약 출력 & 저장 ─────────────────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows).set_index("retriever")

print("\n" + "="*60)
print("📊 리트리버 성능 비교 요약")
print("="*60)
print(summary_df.to_string(float_format="{:.4f}".format))

summary_df.to_csv("./kiyongTests/eval_summary.csv")
print("\n저장 완료: kiyongTests/eval_summary.csv")