# experiment_pipeline.py
import os
from dataclasses import dataclass
from typing import Literal

from core.metadata import save_documents, load_documents
from core.chunker import (
    housing_chunking_recur, housing_chunking_semantic,
    housing_chunking_character, finance_chunking_recur
)
from core.embedder_vectorstore import (
    get_openai_embedder, get_huggingface_embedder,
    embed_and_save_chroma, load_chroma,
)
from core.retriever import (
    get_basic_retriever,
    get_ensemble_retriever,
    get_contextual_compression_retriever,
    get_selfquery_retriever,
    HOUSING_METADATA_FIELD_INFO,
    FINANCE_METADATA_FIELD_INFO,
)
from core.reranker import get_cross_encoder_reranker, get_cohere_reranker

EXP_ROOT          = "./experiments"
BASE_DIR          = os.path.join(EXP_ROOT, "_base")
HOUSING_DOCS_PATH = os.path.join(BASE_DIR, "housing_docs.json")
FINANCE_DOCS_PATH = os.path.join(BASE_DIR, "finance_docs.json")

SELFQUERY_DOCUMENT_CONTENTS = {
    "housing": "청년·신혼부부 대상 주거 지원 정책 문서",
    "finance": "청년·신혼부부 대상 금융 지원 상품 문서",
}


@dataclass
class ExpConfig:
    name:          str
    domain:        Literal["housing", "finance"] = "housing"
    loader: Literal["pdfplumber", "pypdf", "upstage"] = "pdfplumber"
    chunker:       Literal["recur", "semantic", "char"] = "recur"
    embedder:      Literal["openai", "hf"] = "openai"
    vectorstore: Literal["chroma", "faiss"] = "chroma"
    retriever:     Literal["basic", "ensemble", "contextual_compression", "selfquery"]  = "ensemble"
    reranker:      Literal["cross_encoder", "cohere", "none"] = "cross_encoder"
    
    # chunk_size:    int = 500
    # chunk_overlap: int = 50

    breakpoint_threshold_type: str | None = None
    breakpoint_threshold_amount: float | None = None

    pre_chunk_size: int | None = None
    pre_chunk_overlap: int | None = None


def _load_base_docs(domain: str) -> list:
    path = FINANCE_DOCS_PATH if domain == "finance" else HOUSING_DOCS_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"❌ {path} 없음. 먼저 build_base_docs.py를 실행하세요."
        )
    print(f"📂 [공통] {os.path.basename(path)} 로드")
    return load_documents(path)


def _paths(cfg: ExpConfig) -> dict:
    base = os.path.join(EXP_ROOT, cfg.name)
    # chunk_size, overlap을 파일명에 반영해 실험별로 구분
    chunk_tag = f"{cfg.chunker}_{cfg.breakpoint_threshold_type}_{cfg.breakpoint_threshold_amount}"
    return {
        "base":   base,
        "chunks": os.path.join(base, f"{chunk_tag}_chunks.json"),
        "chroma": os.path.join(base, f"{cfg.embedder}_chroma"),
    }


def _chunk(docs, cfg: ExpConfig):
    if cfg.chunker == "semantic":
        return housing_chunking_semantic(
            documents=docs,
            breakpoint_threshold_type=cfg.breakpoint_threshold_type,
            breakpoint_threshold_amount=cfg.breakpoint_threshold_amount,
            pre_chunk_size=cfg.pre_chunk_size,
            pre_chunk_overlap=cfg.pre_chunk_overlap
        )

    if cfg.chunker == "char":
        return housing_chunking_character(
            docs,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap
        )
    # "recur"
    return (
        finance_chunking_recur(docs)
        if cfg.domain == "finance"
        else housing_chunking_recur(docs, cfg.chunk_size, cfg.chunk_overlap)
    )


def _embedder(cfg: ExpConfig):
    return get_huggingface_embedder() if cfg.embedder == "hf" else get_openai_embedder()


def _build_retriever(chunks, vs, cfg: ExpConfig):
    if cfg.retriever == "basic":
        return get_basic_retriever(vs)

    if cfg.retriever == "contextual_compression":
        return get_contextual_compression_retriever(vs)

    if cfg.retriever == "selfquery":
        metadata_field_info = (
            FINANCE_METADATA_FIELD_INFO
            if cfg.domain == "finance"
            else HOUSING_METADATA_FIELD_INFO
        )
        document_contents = SELFQUERY_DOCUMENT_CONTENTS[cfg.domain]
        return get_selfquery_retriever(vs, metadata_field_info, document_contents)

    # "ensemble" (기본)
    return get_ensemble_retriever(chunks, vs)


def _build_reranker(retriever, cfg: ExpConfig):
    if cfg.reranker == "none":
        return retriever
    if cfg.reranker == "cohere":
        return get_cohere_reranker(retriever)
    # "cross_encoder" (기본)
    return get_cross_encoder_reranker(retriever)


def run_experiment(cfg: ExpConfig):
    p = _paths(cfg)
    os.makedirs(p["base"], exist_ok=True)
    print(f"\n🧪 실험: {cfg.name}\n")

    # ── 1. 공통 docs 로드 ────────────────────────────────
    docs = _load_base_docs(cfg.domain)

    # ── 2. 실험별 청킹 ───────────────────────────────────
    if os.path.exists(p["chunks"]):
        print("📂 chunks 로드")
        chunks = load_documents(p["chunks"])
    else:
        print("🔄 청킹")
        chunks = _chunk(docs, cfg)
        save_documents(chunks, p["chunks"])
        print(f"  ✅ {len(chunks)}개 → {p['chunks']}")

    # ── 3. 실험별 임베딩 ─────────────────────────────────
    embedder = _embedder(cfg)
    if os.path.exists(p["chroma"]):
        print("📂 벡터스토어 로드")
        vs = load_chroma(embedder, persist_dir=p["chroma"], collection_name=cfg.name)
    else:
        print("🔄 임베딩")
        vs = embed_and_save_chroma(chunks, embedder, persist_dir=p["chroma"], collection_name=cfg.name)

    # ── 4. Retriever ─────────────────────────────────────
    retriever = _build_retriever(chunks, vs, cfg)

    # ── 5. Reranker ──────────────────────────────────────
    reranker = _build_reranker(retriever, cfg)

    print(f"✅ [{cfg.name}] 준비 완료  "
        f"(retriever={cfg.retriever}, reranker={cfg.reranker})\n")
    return reranker