import os
import re
import pymupdf4llm
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

# finance separator
FINANCE_SEPARATORS = [
    "\n■", "\n□", "\n\n", "\n▪",
    "\nㅇ", "\n※", "\n1.", "\n2.", "\n3.",
    "\n-", "\n", ". ", " ",
]

# housing separator
HOUSING_SEPARATORS = [
    "\n제", "\n①", "\n②", "\n③", "\n④", "\n⑤",
    "\n○", "\n◎", "\n◆", "\n▶", "\n【", "\n[",
    "\n\n", "\n-", "\n", ". ", " ",
]

def bold_to_header(md_text: str) -> str:
    """**섹션명** 패턴을 ## 헤더로 변환"""
    return re.sub(r'^\*\*([^*]+)\*\*$', r'## \1', md_text, flags=re.MULTILINE)


def chunking_by_size(
    all_pages: list,
    chunk_size: int,
    method: str,
    domain: str,
    pdf_folder: str = None
) -> list:
    
    chunk_overlap = int(chunk_size * 0.15)

    # Markdown
    if method == "markdown":
        if not pdf_folder:
            raise ValueError("markdown 방식은 pdf_folder가 필요해요")

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
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        chunks = []
        for root, dirs, files in os.walk(pdf_folder):
            for fname in files:
                if not fname.endswith(".pdf"):
                    continue
                try:
                    md_text = pymupdf4llm.to_markdown(os.path.join(root, fname))
                    md_text = bold_to_header(md_text)
                    splits = md_splitter.split_text(md_text)
                    splits = recur_splitter.split_documents(splits)
                    for s in splits:
                        s.metadata["source"] = fname
                    chunks.extend(splits)
                except Exception as e:
                    print(f"  ⚠️ {fname} 실패: {e}")

    # Recursive
    elif method == "recursive":
        separators = FINANCE_SEPARATORS if domain == "finance" else HOUSING_SEPARATORS
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=separators
        )
        chunks = text_splitter.split_documents(all_pages)

    # Character
    else:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_documents(all_pages)

    chunks = [c for c in chunks if len(c.page_content.strip()) > 50]
    print(f"✅ {domain} {method} (size={chunk_size}, overlap={chunk_overlap}) → {len(chunks)}개 청크")
    return chunks

# Semantic
def chunking_semantic_by_threshold(all_pages: list, threshold: int, domain: str) -> list:
    """threshold를 파라미터로 받아서 semantic 청킹"""
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai import OpenAIEmbeddings
    from config import EMBEDDING_MODEL, OPENAI_API_KEY

    embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
    splitter = SemanticChunker(
        embeddings=embedding,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=threshold
    )
    chunks = splitter.split_documents(all_pages)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 50]
    print(f"✅ {domain} semantic (threshold={threshold}) → {len(chunks)}개 청크")
    return chunks