import json
import pandas as pd
import re
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# =========================
# 환경 변수
# =========================
load_dotenv()

# =========================
# 경로 설정
# =========================
BASE_DIR = Path("experiments")

RESULTS_DIR = BASE_DIR / "results"
SCORES_DIR = BASE_DIR / "scores"

SCORES_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# ✅ 2차 정밀 평가 설정
# =========================
EVAL_TAG = "precise"
MAX_EVAL_ROWS = 30         # 전체 데이터 사용
MAX_CONTEXTS = 5             # TOP_K와 동일하게
MAX_CONTEXT_CHARS = 3000     # context 자르지 않음

# =========================
# RAGAS 평가 모델
# =========================
ragas_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

ragas_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# =========================
# 평가 지표
# =========================
METRICS = [
    context_precision,
    context_recall,
]

# =========================
# 유틸 함수
# =========================
def load_result_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_bad_ground_truth(ground_truth: str) -> bool:
    ground_truth = str(ground_truth).strip()

    if not ground_truth:
        return True

    bad_patterns = [
        "문서에서 확인되지 않습니다",
        "문서에 명시되어 있지 않습니다",
        "문서에는 해당 내용이 없습니다",
        "제공된 문서에는",
        "확인할 수 없습니다",
        "알 수 없습니다",
        "답변할 수 없습니다",
        "정보가 없습니다",
        "포함되어 있지 않습니다",
    ]

    return any(pattern in ground_truth for pattern in bad_patterns)

# =========================
# 🔥 evidence hit 계산
# =========================
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text))

def evidence_hit(contexts, evidence: str) -> bool:
    if not evidence:
        return False

    evidence_norm = normalize_text(evidence)
    merged_contexts = normalize_text("\n".join(contexts))

    return evidence_norm in merged_contexts

# =========================
# Dataset 변환
# =========================
def convert_to_ragas_dataset(results):
    rows = []
    skipped = 0

    for item in results:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        ground_truth = str(item.get("ground_truth", "")).strip()
        contexts = item.get("contexts", [])
        evidence = item.get("evidence", "")

        if isinstance(contexts, str):
            contexts = [contexts]

        contexts = [
            str(c).strip()
            for c in contexts
            if str(c).strip()
        ]

        # ✅ context 개수 제한
        if MAX_CONTEXTS is not None:
            contexts = contexts[:MAX_CONTEXTS]

        # ✅ context 길이 제한 제거
        if MAX_CONTEXT_CHARS is not None:
            contexts = [c[:MAX_CONTEXT_CHARS] for c in contexts]

        # 필수값 체크
        if not question or not answer or not contexts or not ground_truth:
            skipped += 1
            continue

        if is_bad_ground_truth(ground_truth):
            skipped += 1
            continue

        # 🔥 evidence hit 계산
        hit = evidence_hit(contexts, evidence)

        rows.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
            "evidence": evidence,
            "evidence_hit": hit,
        })

    # 평가 개수 제한 없음 (정밀 평가)
    if MAX_EVAL_ROWS is not None:
        rows = rows[:MAX_EVAL_ROWS]

    print(f"✅ RAGAS 변환 완료: {len(rows)}개 / 제외 {skipped}개")

    if len(rows) == 0:
        print("⚠️ 평가 가능한 데이터가 0개라서 스킵")
        return None

    return Dataset.from_list(rows)

# =========================
# detail 경로
# =========================
def get_detail_path(result_path: Path):
    return SCORES_DIR / f"detail_{EVAL_TAG}_{result_path.stem}.csv"

# =========================
# summary 생성
# =========================
def summarize_detail_csv(detail_path: Path, result_path: Path):
    df = pd.read_csv(detail_path)

    metric_cols = [
        "context_precision",
        "context_recall",
    ]

    missing_cols = [col for col in metric_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️ metric 컬럼 없음: {missing_cols}")
        return None

    mean_scores = df[metric_cols].mean(numeric_only=True)

    # 🔥 evidence hit rate 추가
    if "evidence_hit" in df.columns:
        evidence_hit_rate = df["evidence_hit"].mean()
    else:
        evidence_hit_rate = None

    return {
        "experiment": result_path.stem.replace("rag_result_", ""),
        "context_precision": float(mean_scores["context_precision"]),
        "context_recall": float(mean_scores["context_recall"]),
        "evidence_hit_rate": float(evidence_hit_rate) if evidence_hit_rate is not None else None,
    }

# =========================
# 평가 실행
# =========================
def evaluate_one_result_file(result_path: Path):
    detail_path = get_detail_path(result_path)

    if detail_path.exists():
        print(f"⏩ 이미 평가됨 → 스킵: {result_path.name}")
        return summarize_detail_csv(detail_path, result_path)

    print(f"\n🚀 2차 정밀 평가 시작: {result_path.name}")

    results = load_result_json(result_path)
    dataset = convert_to_ragas_dataset(results)

    if dataset is None:
        return None

    print(f"📊 평가 데이터 개수: {len(dataset)}")

    score = evaluate(
        dataset=dataset,
        metrics=METRICS,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    df = score.to_pandas()

    # 🔥 evidence_hit 다시 붙이기
    df["evidence_hit"] = [item["evidence_hit"] for item in dataset]

    df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    summary = summarize_detail_csv(detail_path, result_path)

    print("✅ 완료:", summary)

    return summary

# =========================
# main
# =========================
def main():
    result_files = sorted(RESULTS_DIR.glob("rag_result_*.json"))

    if not result_files:
        print("❌ 결과 파일 없음")
        return

    summaries = []

    for path in result_files:
        summary = evaluate_one_result_file(path)

        if summary is not None:
            summaries.append(summary)

    if not summaries:
        print("❌ summary 생성 실패")
        return

    summary_df = pd.DataFrame(summaries)

    # 🔥 정렬 기준: evidence_hit_rate → recall
    summary_df = summary_df.sort_values(
        by=["evidence_hit_rate", "context_recall"],
        ascending=False
    )

    summary_path = SCORES_DIR / f"ragas_summary_{EVAL_TAG}.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n🏆 2차 정밀 평가 결과")
    print(summary_df)

    print(f"\n💾 저장 완료: {summary_path}")

if __name__ == "__main__":
    main()