# build_base_docs.py
import os

from core.housing_loader import (
    load_pdf_plumber,
    load_pdf_pypdf,
    load_pdf_upstage,
)
from core.metadata import (
    extract_metadata,
    housing_normalize_meta,
    save_metadata,
    save_documents,
)
from config import PDF_FOLDER, MAX_TEXT_LENGTH

BASE_DIR = "./experiments/_base"

# 공통 metadata
HOUSING_META_PATH = os.path.join(BASE_DIR, "housing_meta2.json")

# loader별 docs
HOUSING_DOCS_PATHS = {
    "pdfplumber": os.path.join(BASE_DIR, "housing_docs_pdfplumber.json"),
    "pypdf": os.path.join(BASE_DIR, "housing_docs_pypdf.json"),
    "upstage": os.path.join(BASE_DIR, "housing_docs_upstage.json"),
}

HOUSING_LOADERS = {
    "pdfplumber": load_pdf_plumber,
    "pypdf": load_pdf_pypdf,
    "upstage": load_pdf_upstage,
}


def _pdf_files():
    return [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]


# =========================
# 1. 공통 metadata 생성
# =========================
def build_housing_metadata_once():
    os.makedirs(BASE_DIR, exist_ok=True)

    print("🔄 주거 Metadata 생성 (공통, pdfplumber 기준)")
    meta_list = []
    meta_by_file = {}

    for file in _pdf_files():
        print(f"  📄 metadata: {file}")

        file_path = os.path.join(PDF_FOLDER, file)
        doc = load_pdf_plumber(file_path)

        if doc is None:
            print(f"  ⚠️ metadata 실패 → 스킵: {file}")
            continue

        meta = housing_normalize_meta(
            extract_metadata(doc.page_content[:MAX_TEXT_LENGTH]),
            file
        )

        meta_list.append(meta)
        meta_by_file[file] = meta

    save_metadata(meta_list, HOUSING_META_PATH)

    print(f"✅ metadata 완료: {len(meta_list)}개 → {HOUSING_META_PATH}\n")

    return meta_by_file


# =========================
# 2. loader별 docs 생성
# =========================
def build_housing_docs_by_loader(loader_name, loader_func, meta_by_file):
    os.makedirs(BASE_DIR, exist_ok=True)

    docs = []
    docs_path = HOUSING_DOCS_PATHS[loader_name]

    print(f"🔄 주거 Document 생성 - loader={loader_name}")

    for file in _pdf_files():
        print(f"  📄 [{loader_name}] {file}")

        file_path = os.path.join(PDF_FOLDER, file)
        loaded = loader_func(file_path)

        if loaded is None:
            print(f"  ⚠️ 로드 실패 → 스킵: {file}")
            continue

        # loader 반환 형태 통일
        loaded_docs = loaded if isinstance(loaded, list) else [loaded]

        common_meta = meta_by_file.get(file, {})

        for idx, doc in enumerate(loaded_docs):
            doc.metadata.update(common_meta)
            doc.metadata["loader"] = loader_name
            doc.metadata["source_file"] = file

            # page 정보 보완
            if "page" not in doc.metadata:
                doc.metadata["page"] = idx + 1 if len(loaded_docs) > 1 else None

            docs.append(doc)

    save_documents(docs, docs_path)

    print(f"✅ docs 완료: loader={loader_name}, {len(docs)}개 → {docs_path}\n")


# =========================
# 3. 전체 실행
# =========================
def build_all_housing_docs():
    meta_by_file = build_housing_metadata_once()

    for loader_name, loader_func in HOUSING_LOADERS.items():
        build_housing_docs_by_loader(loader_name, loader_func, meta_by_file)


if __name__ == "__main__":
    build_all_housing_docs()