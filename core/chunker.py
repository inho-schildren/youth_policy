from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from config import EMBEDDING_MODEL, OPENAI_API_KEY
from core.data_loader import collect_documents

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

def finance_chunking_recur(raw_data_path: str) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n■", "\n□", "\n\n", "\n▪",
            "\nㅇ", "\n※", "\n1.", "\n2.", "\n3.",
            "\n-", "\n", ". ", " ",
        ]
    )

    documents = collect_documents(raw_data_path)
    all_pages = []
    for doc in documents:
        all_pages.extend(doc["pages"])

    chunks = text_splitter.split_documents(all_pages)
    print(f"총 청크 수: {len(chunks)}")
    return chunks

def finance_chunking_character(raw_data_path: str) -> list:
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    documents = collect_documents(raw_data_path)
    all_pages = []
    for doc in documents:
        all_pages.extend(doc["pages"])

    chunks = text_splitter.split_documents(all_pages)
    print(f"글자수 기반 총 청크 수: {len(chunks)}")
    return chunks

def run_finance_chunking_semantic(raw_data_path: str) -> list:
    text_splitter = SemanticChunker(OpenAIEmbeddings())

    documents = collect_documents(raw_data_path)
    all_pages = []
    for doc in documents:
        all_pages.extend(doc["pages"])

    chunks = text_splitter.split_documents(all_pages)
    print(f"의미 기반 총 청크 수: {len(chunks)}")
    return chunks