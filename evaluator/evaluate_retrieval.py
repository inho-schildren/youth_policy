import json
import os
from langchain_community.vectorstores import Chroma
from core.embedder_vectorstore import get_openai_embedder_small

# ── 경로 설정 ──────────────────────────────────────────────────
CHROMA_DIR  = "./db/chroma_experiment"
EVAL_PATH   = "./evaluator/eval_all.json"
RESULT_PATH = "./evaluator/retrieval_results.json"
TOP_K       = 5  # 검색 결과 몇 개 볼지

# ── 실험할 컬렉션 목록 ─────────────────────────────────────────
COLLECTIONS = [
    "finance_recursive_size500",
    "finance_recursive_size800",
    "finance_recursive_size1200",
    "finance_character_size500",
    "finance_character_size800",
    "finance_character_size1200",
    "finance_markdown_size500",
    "finance_markdown_size800",
    "finance_markdown_size1200",
    "finance_semantic_t70",
    "finance_semantic_t85",
    "finance_semantic_t95",
    "housing_recursive_size300",
    "housing_recursive_size500",
    "housing_recursive_size800",
    "housing_character_size300",
    "housing_character_size500",
    "housing_character_size800",
    "housing_markdown_size300",
    "housing_markdown_size500",
    "housing_markdown_size800",
    "housing_semantic_t70",
    "housing_semantic_t85",
    "housing_semantic_t95"
]

def load_eval_set(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_source_from_chunk(doc) -> str:
    """청크 메타데이터에서 정책명(title) 추출"""
    return doc.metadata.get("title", doc.metadata.get("source", ""))

# ── 정책 유형 키워드 ────────────────────────────────────────────
HOUSING_KEYWORDS = [
    "행복주택", "매입임대", "전세임대", "국민임대", "영구임대",
    "장기전세", "안심주택", "청년주택", "신혼희망타운", "공공임대",
    "보증금지원", "든든주택", "청년누리", "도전숙"
]

FINANCE_KEYWORDS = [
    "버팀목", "디딤돌", "청년월세", "임차보증금", "전세자금",
    "보증부월세", "청약통장", "특별공급", "전세보증금반환보증",
    "중개보수", "이사비", "구입자금"
]

def is_hit_check(answer_src: str, retrieved_sources: list, domain: str) -> bool:
    """정책 유형 키워드 기반 Hit 판단"""
    keywords = HOUSING_KEYWORDS if domain == "housing" else FINANCE_KEYWORDS
    
    # 정답 source에서 키워드 추출
    matched_keyword = next((kw for kw in keywords if kw in answer_src), None)
    
    if matched_keyword:
        # 키워드가 검색 결과에 포함되면 Hit
        return any(matched_keyword in src for src in retrieved_sources)
    
    # 키워드 없으면 기존 방식 (완전 일치)
    return any(answer_src in src or src in answer_src for src in retrieved_sources)

def evaluate_collection(collection_name: str, eval_set: list, embedder, top_k: int = 5) -> dict:
    print(f"\n  📊 {collection_name} 평가 중...")

    # 컬렉션 로드
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
        embedding_function=embedder
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    hits = 0
    misses = 0
    miss_examples = []

    for item in eval_set:
        question   = item["question"]
        answer_src = item["source"]   # 정답 출처 (title)
        domain     = item["domain"]

        # 도메인이 맞는 컬렉션에서만 평가
        # finance 질문은 finance 컬렉션에서만, housing은 housing에서만
        if not collection_name.startswith(domain):
            continue

        # 검색
        retrieved_docs = retriever.invoke(question)
        retrieved_sources = [get_source_from_chunk(doc) for doc in retrieved_docs]

        # Hit 판단: 정답 출처가 검색 결과에 있는지
        is_hit = is_hit_check(answer_src, retrieved_sources, domain)

        if is_hit:
            hits += 1
        else:
            misses += 1
            miss_examples.append({
                "question": question,
                "expected": answer_src,
                "retrieved": retrieved_sources
            })

    total = hits + misses
    hit_rate = hits / total if total > 0 else 0

    print(f"    Hit: {hits}/{total} | Hit Rate: {hit_rate:.2f}")

    return {
        "collection": collection_name,
        "total":      total,
        "hits":       hits,
        "misses":     misses,
        "hit_rate":   round(hit_rate, 4),
        "miss_examples": miss_examples[:3]  # 미스 샘플 3개만 저장
    }


if __name__ == "__main__":
    print("\n📂 테스트셋 로드 중...")
    eval_set = load_eval_set(EVAL_PATH)
    print(f"  총 {len(eval_set)}개 질문")

    embedder = get_openai_embedder_small()
    all_results = []

    print("\n🔍 Hit Rate 측정 시작")
    for collection_name in COLLECTIONS:
        result = evaluate_collection(collection_name, eval_set, embedder, TOP_K)
        all_results.append(result)

    # ── 결과 저장 ──────────────────────────────────────────────
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # ── 결과 출력 ──────────────────────────────────────────────
    print(f"\n\n{'='*50}")
    print("📊 Hit Rate 비교 결과")
    print(f"{'='*50}")
    print(f"{'컬렉션':<25} {'Hit Rate':>10} {'Hit':>6} {'Miss':>6} {'Total':>6}")
    print("-" * 55)

    # finance / housing 분리해서 출력
    finance_results = [r for r in all_results if r["collection"].startswith("finance")]
    housing_results = [r for r in all_results if r["collection"].startswith("housing")]

    print("\n[Finance]")
    for r in sorted(finance_results, key=lambda x: x["hit_rate"], reverse=True):
        print(f"  {r['collection']:<23} {r['hit_rate']:>10.2f} {r['hits']:>6} {r['misses']:>6} {r['total']:>6}")

    print("\n[Housing]")
    for r in sorted(housing_results, key=lambda x: x["hit_rate"], reverse=True):
        print(f"  {r['collection']:<23} {r['hit_rate']:>10.2f} {r['hits']:>6} {r['misses']:>6} {r['total']:>6}")

    print(f"\n💾 결과 저장 완료: {RESULT_PATH}")