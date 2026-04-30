import json
import os
from langchain.schema import Document
from evaluator.chunker_exp import chunking_by_size, chunking_semantic_by_threshold
from core.embedder_vectorstore import embed_and_save_chroma, get_openai_embedder_small
from config import FINANCE_DOCS_PATH, DOCS_PATH
from langchain_community.vectorstores.utils import filter_complex_metadata

# ── 경로 설정 ──────────────────────────────────────────────────
CHUNKS_DIR  = "./data/chunks"
CHROMA_DIR  = "./db/chroma_experiment"
RAW_HOUSING = "./data/raw"
RAW_FINANCE = "./data/raw_data"

os.makedirs(CHUNKS_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# ── 실험 파라미터 ──────────────────────────────────────────────
FINANCE_SIZES = [500, 800, 1200]
HOUSING_SIZES = [300, 500, 800]
METHODS       = ["recursive", "character", "markdown"]

def load_documents(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]

def save_chunks_json(chunks: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks],
            f, ensure_ascii=False, indent=2
        )
    print(f"  💾 JSON 저장 완료: {path} ({len(chunks)}개)")

def run_experiment(name: str, chunks: list, domain: str):
    print(f"\n{'='*50}")
    print(f"🧪 실험: {domain}_{name}")
    print(f"{'='*50}")

    json_path = os.path.join(CHUNKS_DIR, f"{domain}_{name}.json")
    save_chunks_json(chunks, json_path)

    chunks = filter_complex_metadata(chunks)

    collection_name = f"{domain}_{name}"
    embedder = get_openai_embedder_small()
    embed_and_save_chroma(
        chunks=chunks,
        embedder=embedder,
        persist_dir=CHROMA_DIR,
        collection_name=collection_name
    )
    print(f"  ✅ 컬렉션 생성 완료: {collection_name}")
    return collection_name


if __name__ == "__main__":

    print("\n📂 문서 로드 중...")
    finance_docs = load_documents(FINANCE_DOCS_PATH)
    housing_docs = load_documents(DOCS_PATH)
    print(f"  finance: {len(finance_docs)}개 | housing: {len(housing_docs)}개")

    all_collections = []

    # ── Finance 실험 ───────────────────────────────────────────
    # print("\n\n🏦 Finance 청킹 실험 시작")
    # for method in METHODS:
    #     for size in FINANCE_SIZES:
    #         chunks = chunking_by_size(
    #             all_pages=finance_docs,
    #             chunk_size=size,
    #             method=method,
    #             domain="finance",
    #             pdf_folder=RAW_FINANCE if method == "markdown" else None
    #         )
    #         col = run_experiment(f"{method}_size{size}", chunks, "finance")
    #         all_collections.append(col)

    # ── Housing 실험 ───────────────────────────────────────────
    # print("\n\n🏠 Housing 청킹 실험 시작")
    # for method in METHODS:
    #     for size in HOUSING_SIZES:
    #         chunks = chunking_by_size(
    #             all_pages=housing_docs,
    #             chunk_size=size,
    #             method=method,
    #             domain="housing",
    #             pdf_folder=RAW_HOUSING if method == "markdown" else None
    #         )
    #         col = run_experiment(f"{method}_size{size}", chunks, "housing")
    #         all_collections.append(col)

    SEMANTIC_THRESHOLDS = [70, 85, 95]

    print("\n\n💡 Finance Semantic 청킹 실험 시작")
    for threshold in SEMANTIC_THRESHOLDS:
        chunks = chunking_semantic_by_threshold(finance_docs, threshold, "finance")
        col = run_experiment(f"semantic_t{threshold}", chunks, "finance")
        all_collections.append(col)

    print("\n\n💡 Housing Semantic 청킹 실험 시작")
    for threshold in SEMANTIC_THRESHOLDS:
        chunks = chunking_semantic_by_threshold(housing_docs, threshold, "housing")
        col = run_experiment(f"semantic_t{threshold}", chunks, "housing")
        all_collections.append(col)

    # ── 결과 요약 ──────────────────────────────────────────────
    print(f"\n\n{'='*50}")
    print("🎉 실험 완료 요약")
    print(f"{'='*50}")
    finance_cols = [c for c in all_collections if c.startswith("finance")]
    housing_cols = [c for c in all_collections if c.startswith("housing")]
    print(f"\nFinance 컬렉션 ({len(finance_cols)}개):")
    for col in finance_cols:
        print(f"  - {col}")
    print(f"\nHousing 컬렉션 ({len(housing_cols)}개):")
    for col in housing_cols:
        print(f"  - {col}")
    print(f"\nChroma DB 위치: {CHROMA_DIR}")
    print(f"청크 JSON 위치: {CHUNKS_DIR}")