from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from config import EMBEDDING_MODEL, OPENAI_API_KEY
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter
import re
import os

def bold_to_header(md_text: str) -> str:
    """**섹션명** 패턴을 ## 헤더로 변환"""
    return re.sub(r'^\*\*([^*]+)\*\*$', r'## \1', md_text, flags=re.MULTILINE)

def housing_chunking_semantic(documents: list) -> list:
    embedding = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )

    semantic_splitter = SemanticChunker(
        embeddings=embedding,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85
    )

    chunks = semantic_splitter.split_documents(documents)
    print(f"✅ 청킹 완료: {len(documents)}개 → {len(chunks)}개 청크")
    return chunks

def housing_chunking_recur(all_pages: list) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n제", "\n①", "\n②", "\n③", "\n④", "\n⑤",
            "\n○", "\n◎", "\n◆", "\n▶", "\n【", "\n[",
            "\n\n", "\n-", "\n", ". ", " ",
        ]
    )

    chunks = text_splitter.split_documents(all_pages)
    print(f"총 청크 수: {len(chunks)}")
    return chunks


def housing_chunking_character(documents: list) -> list:
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)
    print(f"글자수 기반 총 청크 수: {len(chunks)}")
    return chunks

def housing_chunking_markdown(pdf_folder: str) -> list:
    headers_to_split = [
        ("#", "header1"),
        ("##", "header2"),
        ("###", "header3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split,
        strip_headers=False
    )
    recur_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50
    )

    chunks = []
    for root, dirs, files in os.walk(pdf_folder):
        for fname in files:
            if not fname.endswith(".pdf"):
                continue
            pdf_path = os.path.join(root, fname)
            try:
                md_text = pymupdf4llm.to_markdown(pdf_path)
                md_text = bold_to_header(md_text)
                splits = md_splitter.split_text(md_text)
                splits = recur_splitter.split_documents(splits)
                for s in splits:
                    s.metadata["source"] = fname
                chunks.extend(splits)
            except Exception as e:
                print(f"  ⚠️ {fname} 변환 실패: {e}")

    chunks = [c for c in chunks if len(c.page_content.strip()) > 100]
    print(f"✅ housing 마크다운 청킹 완료: {len(chunks)}개 청크")
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

def finance_chunking_markdown(pdf_folder: str) -> list:
    import os
    headers_to_split = [("#", "header1"), ("##", "header2"), ("###", "header3")]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split,
        strip_headers=False
    )
    recur_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50
    )

    chunks = []
    # 하위 폴더까지 재귀적으로 탐색
    for root, dirs, files in os.walk(pdf_folder):
        for fname in files:
            if not fname.endswith(".pdf"):
                continue
            pdf_path = os.path.join(root, fname)
            try:
                md_text = pymupdf4llm.to_markdown(pdf_path)
                md_text = bold_to_header(md_text)        # bold → 헤더 변환
                splits = md_splitter.split_text(md_text)  # 1차: 헤더 기준
                splits = recur_splitter.split_documents(splits)  # 2차: 크기 기준
                for s in splits:
                    s.metadata["source"] = fname
                chunks.extend(splits)
            except Exception as e:
                print(f"  ⚠️ {fname} 변환 실패: {e}")

    chunks = [c for c in chunks if len(c.page_content.strip()) > 100]
    print(f"✅ finance 마크다운 청킹 완료: {len(chunks)}개 청크")
    return chunks