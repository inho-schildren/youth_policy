from langchain_community.document_loaders import PyPDFLoader
import os

def load_pdf_pages(pdf_path: str) -> list:
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def collect_documents(root_dir: str) -> list[dict]:
    documents = []

    for item in sorted(os.listdir(root_dir)):
        item_path = os.path.join(root_dir, item)

        if os.path.isdir(item_path):
            pdf_files = sorted([
                f for f in os.listdir(item_path) if f.endswith(".pdf")
            ])
            if not pdf_files:
                continue

            all_pages = []
            for pdf_file in pdf_files:
                pdf_path = os.path.join(item_path, pdf_file)
                all_pages.extend(load_pdf_pages(pdf_path))

            documents.append({
                "group_name": item,
                "type": "folder",
                "pages": all_pages,
                "combined_text": "\n".join([p.page_content for p in all_pages]),
                "files": pdf_files
            })

        elif item.endswith(".pdf"):
            pages = load_pdf_pages(item_path)
            documents.append({
                "group_name": item.replace(".pdf", ""),
                "type": "single",
                "pages": pages,
                "combined_text": "\n".join([p.page_content for p in pages]),
                "files": [item]
            })

    return documents