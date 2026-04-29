from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from config import EMBEDDING_MODEL, OPENAI_API_KEY

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

# 기본 recursive
def finance_chunking_recur(all_pages):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n■", "\n□", "\n\n", "\n▪",
            "\nㅇ", "\n※", "\n1.", "\n2.", "\n3.",
            "\n-", "\n", ". ", " ",
        ]
    )

    chunks = text_splitter.split_documents(all_pages)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 100]
    print(f"✅ 금융 recursive 청킹 완료: {len(all_pages)}개 → {len(chunks)}개 청크")
    return chunks

# 파일별 다른 구분자 활용 recursive
def finance_chunking_recur_v2(all_pages):
    """recursive v2 - 파일명으로 문서 유형 감지 후 다른 separator 적용"""

    NOTICE_KEYWORDS = ["공고문", "FAQ"]
    notice_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=[
            "\nQ", "\n[", "\n□", "\n■",
            "\n\n", "\nㅇ", "\n◦", "\n※",    
            "\n①", "\n②", "\n③", "\n④", "\n⑤",
            "\n1.", "\n2.", "\n3.", "\n4.", "\n5.",
            "\n-", "\n", ". ", " ",
        ]
    )

    # 일반 금융상품 separator
    product_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=[
            "\n■", "\n\n", "\n▪",
            "\n•", "\n-", "\n",
            ". ", " ",
        ]
    )

    def is_notice(page):
        source = page.metadata.get("source", "")
        return any(kw in source for kw in NOTICE_KEYWORDS)

    chunks = []
    for page in all_pages:
        splitter = notice_splitter if is_notice(page) else product_splitter
        split = splitter.split_documents([page])
        chunks.extend(split)

    chunks = [c for c in chunks if len(c.page_content.strip()) > 100]
    print(f"✅ recursive v2 청킹 완료: {len(all_pages)}개 → {len(chunks)}개 청크")
    return chunks
def finance_chunking_character(all_pages):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=150,
        length_function=len
    )

    chunks = text_splitter.split_documents(all_pages)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 100]
    print(f"글자수 기반 총 청크 수: {len(chunks)}")
    return chunks

def finance_chunking_semantic(all_pages):
    text_splitter = SemanticChunker(OpenAIEmbeddings())

    chunks = text_splitter.split_documents(all_pages)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 100]
    print(f"의미 기반 총 청크 수: {len(chunks)}")
    return chunks