import os
from langchain.schema import Document
from loader import load_pdf, get_text_by_pages, show_metadata
from metadata import (
    extract_metadata, normalize_meta,
    save_metadata, load_metadata,
    save_documents, load_documents
)
from chunker import chunk_documents
from embedder import embed_and_save, load_vectorstore
from retriever import get_retriever

PDF_FOLDER  = "./realdata"
META_PATH   = "metadata.json"
DOCS_PATH   = "output.json"
CHUNKS_PATH = "chunks.json"
CHROMA_DIR  = "./db/chroma_db"

# ── 1. PDF → Document 생성 (최초 1회) ────────────────────
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

        full_text = get_text_by_pages(pages, 30)[:5000]
        meta = normalize_meta(extract_metadata(full_text), file)
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

else:
    print("📂 기존 output.json 로드")
    documents = load_documents(DOCS_PATH)

# ── 2. 청킹 (최초 1회) ───────────────────────────────────
if not os.path.exists(CHUNKS_PATH):
    print("🔄 청킹 시작")
    chunks = chunk_documents(documents)
    save_documents(chunks, CHUNKS_PATH)
else:
    print("📂 기존 chunks.json 로드")
    chunks = load_documents(CHUNKS_PATH)

# ── 3. 임베딩 (최초 1회) ─────────────────────────────────
if not os.path.exists(CHROMA_DIR):
    print("🔄 임베딩 시작")
    vectorstore = embed_and_save(chunks)
else:
    print("📂 기존 벡터스토어 로드")
    vectorstore = load_vectorstore()

# ── 4. 확인 ──────────────────────────────────────────────
show_metadata(documents)
print(f"\n✅ 총 Document 수: {len(documents)}")
print(f"✅ 총 Chunk 수: {len(chunks)}")

# ── 5. Retriever 생성 ─────────────────────────────────────
retriever = get_retriever(chunks)

# ── 테스트 ────────────────────────────────────────────────
query = "나는 금천구에서 거주 예정인 신혼부부인데 시행중인 주거 정책이 있으면 알려줘"
results = retriever.invoke(query)

print(f"\n🔍 검색 결과 ({len(results)}개)")
for i, doc in enumerate(results, 1):
    print(f"\n[{i}] {doc.metadata.get('title')} / 페이지 {doc.metadata.get('page')}")
    print(f"    {doc.page_content[:100]}...")