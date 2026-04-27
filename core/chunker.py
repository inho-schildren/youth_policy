from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from config import EMBEDDING_MODEL, OPENAI_API_KEY

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=50
# )

# def chunk_documents(documents):
#     chunks = []
#     for doc in documents:
#         if not doc.page_content.strip():
#             continue
#         split_texts = splitter.split_text(doc.page_content)
#         for text in split_texts:
#             chunks.append(Document(
#                 page_content=text,
#                 metadata=doc.metadata
#             ))
#     print(f"✅ 청킹 완료: {len(documents)}개 → {len(chunks)}개 청크")
#     return chunks


embedding = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY
)

semantic_splitter = SemanticChunker(
    embeddings=embedding,
    breakpoint_threshold_type="percentile",  # 의미 변화 감지 방식
    breakpoint_threshold_amount=85           # 85% 이상 의미 변화 시 청크 분리
)

def chunk_documents(documents):
    chunks = []
    for doc in documents:
        if not doc.page_content.strip():
            continue
        try:
            split_texts = semantic_splitter.split_text(doc.page_content)
            for text in split_texts:
                chunks.append(Document(
                    page_content=text,
                    metadata=doc.metadata
                ))
        except Exception as e:
            print(f"  ⚠️ 청킹 실패: {e}")
            chunks.append(doc)  # 실패 시 원본 그대로

    print(f"✅ 청킹 완료: {len(documents)}개 → {len(chunks)}개 청크")
    return chunks