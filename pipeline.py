import os
import json
from langchain.schema import Document
from core.housing_loader import load_pdf, get_text_by_pages
from core.finance_loader import collect_documents
from core.metadata import (
    extract_metadata, housing_normalize_meta,
    save_metadata, save_documents, load_documents
)
from core.chunker import chunk_documents, finance_chunking_recur
from core.embedder_vectorstore import (
    housing_embed_and_save, housing_load_vectorstore,
    finance_embed_and_save, finance_load_vectorstore)
from core.retriever import housing_retriever, finance_retriever
from core.reranker import get_cross_encoder_reranker
from config import (
    PDF_FOLDER, META_PATH, DOCS_PATH,
    CHUNKS_PATH, CHROMA_DIR,
    MAX_PAGES, MAX_TEXT_LENGTH,
    FINANCE_PDF_FOLDER, FINANCE_META_PATH, FINANCE_DOCS_PATH,
    FINANCE_CHUNKS_PATH, FINANCE_CHROMA_DIR
)

def run_pipeline():
    # ── 1. PDF → Document ─────────────────────────────────
    if not os.path.exists(DOCS_PATH):
        print("🔄 PDF에서 Document 생성 시작")
        documents = []
        meta_list = []
        pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

        for file in pdf_files:
            print(f"📄 {file}")
            pages = load_pdf(os.path.join(PDF_FOLDER, file))
            if not pages:
                continue

            full_text = get_text_by_pages(pages, MAX_PAGES)[:MAX_TEXT_LENGTH]
            meta = housing_normalize_meta(extract_metadata(full_text), file)
            meta_list.append(meta)

            for p in pages:
                if not p["page_content"].strip():
                    continue
                documents.append(Document(
                    page_content=p["page_content"],
                    metadata={**meta, "page": p["page"]}
                ))

        save_metadata(meta_list, META_PATH)
        save_documents(documents, DOCS_PATH)
        print(f"✅ Document 생성 완료: {len(documents)}개\n")

    else:
        print("📂 기존 output_v2.json 로드")
        documents = load_documents(DOCS_PATH)

    # ── 2. 청킹 ───────────────────────────────────────────
    if not os.path.exists(CHUNKS_PATH):
        print("🔄 청킹 시작")
        chunks = chunk_documents(documents)
        save_documents(chunks, CHUNKS_PATH)
    else:
        print("📂 기존 chunks_v2.json 로드")
        chunks = load_documents(CHUNKS_PATH)

    # ── 3. 임베딩 ─────────────────────────────────────────
    if not os.path.exists(CHROMA_DIR):
        print("🔄 임베딩 시작")
        housing_embed_and_save(chunks)
    else:
        print("📂 기존 벡터스토어 로드")
        housing_load_vectorstore()

    # ── 4. Retriever + Reranker ───────────────────────────
    retriever = housing_retriever(chunks)
    reranker  = get_cross_encoder_reranker(retriever)

    print("✅ 파이프라인 준비 완료\n")
    return reranker

def run_finance_pipeline():
    # ── 1. PDF → Document ─────────────────────────────────
    if not os.path.exists(FINANCE_DOCS_PATH):
        print("🔄 금융 PDF 로드 시작")

        all_pages = collect_documents(FINANCE_PDF_FOLDER)

        # 메타데이터 로드
        with open(FINANCE_META_PATH, encoding="utf-8") as f:
            all_metadata = json.load(f)
        meta_dict = {m["group_name"]: m for m in all_metadata}

        for page in all_pages:
            group_name = page.metadata.get("group_name", "")
            meta = meta_dict.get(group_name, {})
            page.metadata.update({
                k: ", ".join(str(i) for i in v) if isinstance(v, list) else v
                for k, v in meta.items()
            })

        save_documents(all_pages, FINANCE_DOCS_PATH)
    else:
        print("📂 기존 finance_documents.json 로드")
        all_pages = load_documents(FINANCE_DOCS_PATH)

    # ── 2. 청킹 ───────────────────────────────────────────
    if not os.path.exists(FINANCE_CHUNKS_PATH):
        print("🔄 금융 청킹 시작")
        chunks = finance_chunking_recur(all_pages)
        save_documents(chunks, FINANCE_CHUNKS_PATH)
    else:
        print("📂 기존 finance_chunks.json 로드")
        chunks = load_documents(FINANCE_CHUNKS_PATH)

    # ── 3. 임베딩 ─────────────────────────────────────────
    if not os.path.exists(FINANCE_CHROMA_DIR):
        print("🔄 금융 임베딩 시작")
        finance_embed_and_save(chunks)
    else:
        print("📂 기존 금융 벡터스토어 로드")
        finance_load_vectorstore()

    # ── 4. Retriever ──────────────────────────────────────
    retriever = finance_retriever(chunks)
    reranker = get_cross_encoder_reranker(retriever)

    print("✅ 금융 파이프라인 준비 완료\n")
    return reranker