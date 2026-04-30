from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
import pdfplumber
import os

def load_pdf_pages_pymupdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    return loader.load()

def load_pdf_pages_plumber(pdf_path):
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                from langchain.schema import Document
                pages.append(Document(
                    page_content=text,
                    metadata={"source": pdf_path}
                ))
    except Exception as e:
        print(f"  ❌ PDF 열기 실패: {e}")
    return pages

def collect_documents(root_dir, loader_type="pymupdf"):
    all_documents = []

    # loader 선택
    load_fn = load_pdf_pages_plumber if loader_type == "plumber" else load_pdf_pages_pymupdf

    for item in sorted(os.listdir(root_dir)):
        item_path = os.path.join(root_dir, item)

        if os.path.isdir(item_path):
            pdf_files = sorted([
                f for f in os.listdir(item_path) if f.endswith(".pdf")
            ])
            if not pdf_files:
                continue

            for pdf_file in pdf_files:
                pdf_path = os.path.join(item_path, pdf_file)
                pages = load_fn(pdf_path)  # ← load_fn 사용
                for page in pages:
                    page.metadata["group_name"] = item
                all_documents.extend(pages)

        elif item.endswith(".pdf"):
            pages = load_fn(item_path)  # ← load_fn 사용
            for page in pages:
                page.metadata["group_name"] = item.replace(".pdf", "")
            all_documents.extend(pages)

    return all_documents