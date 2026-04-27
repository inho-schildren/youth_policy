import os
import json
from openai import OpenAI
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_metadata(text):
    prompt = f"""
너는 주택 정책 문서를 분석하는 AI다.
아래 텍스트를 보고 반드시 JSON만 출력해라. 마크다운 코드블록 없이 순수 JSON만 출력해라.

{{
    "doc_id": "", "title": "", "category": "주택",
    "source": "", "source_url": "", "summary": "", "tags": [],
    "target": [], "age_min": null, "age_max": null,
    "marital_status": "무관", "is_homeless": null,
    "income_condition": "", "income_max_man": null, "asset_max_man": null,
    "region": "", "policy_type": "housing",
    "housing_type": "", "supply_type": "", "contract_type": "",
    "support_type": [], "deposit_support": false, "monthly_rent_support": false,
    "max_deposit": 0, "max_monthly_rent": 0,
    "residence_requirement": "", "priority_condition": [],
    "duration": "", "application_period": ""
}}

텍스트: {text}
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = res.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON 파싱 실패: {e}")
        return {}
    except Exception as e:
        print(f"  ❌ metadata 추출 실패: {e}")
        return {}

def safe_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        return [value]
    return []

def normalize_meta(meta, file_name):
    return {
        "doc_id":                meta.get("doc_id") or file_name,
        "title":                 meta.get("title", ""),
        "category":              "주택",
        "source":                file_name,
        "source_url":            meta.get("source_url", ""),
        "summary":               meta.get("summary", ""),
        "tags":                  safe_list(meta.get("tags")),
        "target":                safe_list(meta.get("target")),
        "age_min":               meta.get("age_min"),
        "age_max":               meta.get("age_max"),
        "marital_status":        meta.get("marital_status", "무관"),
        "is_homeless":           meta.get("is_homeless"),
        "income_condition":      meta.get("income_condition", ""),
        "income_max_man":        meta.get("income_max_man"),
        "asset_max_man":         meta.get("asset_max_man"),
        "region":                meta.get("region", ""),
        "policy_type":           "housing",
        "housing_type":          meta.get("housing_type", ""),
        "supply_type":           meta.get("supply_type", ""),
        "contract_type":         meta.get("contract_type", ""),
        "support_type":          safe_list(meta.get("support_type")),
        "deposit_support":       bool(meta.get("deposit_support", False)),
        "monthly_rent_support":  bool(meta.get("monthly_rent_support", False)),
        "max_deposit":           meta.get("max_deposit") or 0,
        "max_monthly_rent":      meta.get("max_monthly_rent") or 0,
        "residence_requirement": meta.get("residence_requirement", ""),
        "priority_condition":    safe_list(meta.get("priority_condition")),
        "duration":              meta.get("duration", ""),
        "application_period":    meta.get("application_period", "")
    }

def save_metadata(meta_list, save_path="./data/metadata.json"):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)
    print(f"✅ metadata 저장 완료 → {save_path}")

def load_metadata(save_path="./data/metadata.json"):
    with open(save_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_documents(documents, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents],
            f, ensure_ascii=False, indent=2
        )
    print(f"✅ documents 저장 완료 → {save_path}")

def load_documents(save_path):
    with open(save_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]