import pdfplumber

def load_pdf(file_path):
    pages = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text() or ""
                except Exception as e:
                    print(f"  ⚠️ 페이지 {i+1} 텍스트 추출 실패: {e}")
                    text = ""
                pages.append({"page_content": text, "page": i + 1})
    except Exception as e:
        print(f"  ❌ PDF 열기 실패: {e}")
    return pages

def get_text_by_pages(pages, max_pages=30):
    return " ".join([p["page_content"] for p in pages[:max_pages]])

def show_metadata(documents, limit=3):
    seen = set()
    count = 0
    for doc in documents:
        if count >= limit:
            break
        file_name = doc.metadata.get("source")
        if file_name in seen:
            continue
        seen.add(file_name)
        count += 1

        print("=" * 60)
        print(f"📄 파일명: {file_name}")
        print("=" * 60)
        max_key_length = max(len(k) for k in doc.metadata.keys())
        for k, v in doc.metadata.items():
            if k == "page":
                continue
            v_str = str(v)
            if len(v_str) > 100:
                v_str = v_str[:100] + "..."
            print(f"{k:<{max_key_length}} : {v_str}")
        print()