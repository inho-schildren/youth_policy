import pdfplumber
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_upstage import UpstageDocumentParseLoader
import os
from dotenv import load_dotenv

load_dotenv()

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")


def load_pdf_plumber(file_path: str) -> Document | None:
    """pdfplumber로 PDF 전체를 하나의 Document로 로드"""
    pages = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text() or ""
                except Exception as e:
                    print(f"  ⚠️ 페이지 {i+1} 텍스트 추출 실패: {e}")
                    text = ""
                pages.append(text)
    except Exception as e:
        print(f"  ❌ PDF 열기 실패: {e}")
        return None

    full_text = "\n".join([p for p in pages if p.strip()])
    return Document(
        page_content=full_text,
        metadata={"source": file_path}
    )


def load_pdf_pypdf(file_path: str) -> Document | None:
    """PyPDFLoader로 PDF 전체를 하나의 Document로 로드"""
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
    except Exception as e:
        print(f"  ❌ PDF 열기 실패: {e}")
        return None

    full_text = "\n".join([p.page_content for p in pages if p.page_content.strip()])
    return Document(
        page_content=full_text,
        metadata={"source": file_path}
    )

def load_pdf_upstage(file_path: str) -> Document | None:
    """Upstage Document Parse로 PDF 전체를 하나의 Document로 로드"""
    try:
        loader = UpstageDocumentParseLoader(
            file_path=file_path,
            split="page", # element
            api_key=UPSTAGE_API_KEY
        )

        pages = loader.load()

    except Exception as e:
        print(f"  ❌ Upstage 로드 실패: {e}")
        return None

    full_text = "\n".join([
        p.page_content for p in pages
        if p.page_content and p.page_content.strip()
    ])

    return Document(
        page_content=full_text,
        metadata={"source": file_path}
    )