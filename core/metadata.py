import os
import re
import json
from openai import OpenAI
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from core.finance_loader import collect_documents

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

def housing_normalize_meta(meta, file_name):
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


FINANCE_SCHEMA = """
{
    "doc_id": "고유 ID",
    "title": "상품명",
    "region": "전국/서울/경기 등",
    "source_url": "상품 URL 또는 null",
    "summary": "한줄요약",
    "tags": ["관련 태그 목록"],
    "target": ["청년/신혼부부/일반 중 해당하는 것"],
    "age_min": 숫자 또는 null,
    "age_max": 숫자 또는 null,
    "marital_status": "미혼/기혼/예비신혼부부/무관",
    "requires_no_house": "무주택자여야 신청 가능하면 true, 아니면 false",
    "income_condition": "소득 조건 텍스트 또는 null",
    "income_max_man": 숫자(만원) 또는 null,
    "asset_max_man": 숫자(만원) 또는 null,
    "loan_type": "전세/월세/구입",
    "interest_rate_min": 숫자 또는 null,
    "interest_rate_max": 숫자 또는 null,
    "loan_limit_man": 숫자(만원) 또는 null,
    "loan_period_max": 숫자(년) 또는 null,
    "special_condition": ["생애최초/전세사기피해/신생아 등 또는 빈 리스트"],
    "is_first_purchase": true/false
}
"""

FINANCE_DEFAULTS = {
    "age_min": 0, "age_max": 99,
    "marital_status": "무관", "requires_no_house": None,
    "income_max_man": 99999, "asset_max_man": 99999,
    "region": "전국", "special_condition": [],
    "is_first_purchase": False,
}

def extract_finance_metadata(doc: dict) -> dict:
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    text = doc["combined_text"]
    response = model.invoke([
        SystemMessage(
            content=f"""청년 주거 금융 정책 문서입니다.
            메타데이터를 JSON 형식에 맞춰 반환하세요. JSON 외 텍스트 금지.
            값을 알 수 없거나 문서에 없는 경우 null로 채우세요.
            상품 폴더명: {doc['group_name']}
            스키마: {FINANCE_SCHEMA}"""),
        HumanMessage(content=f"문서 내용:\n{text}")
    ])
    content = re.sub("```json|```", "", response.content).strip()
    result = json.loads(content)
    return {
        "group_name": doc["group_name"],
        "category": "금융",
        "source": doc["files"],
        **FINANCE_DEFAULTS,
        **{k: v for k, v in result.items() if v is not None},
    }


def run_finance_metadata_extraction(raw_data_path: str, save_path: str = "./data/finance_metadata.json") -> list:
    
    documents = collect_documents(raw_data_path)
    all_metadata = []
    for doc in documents:
        print(f"처리 중: {doc['group_name']}")
        meta = extract_finance_metadata(doc)
        all_metadata.append(meta)
    save_metadata(all_metadata, save_path)
    return all_metadata