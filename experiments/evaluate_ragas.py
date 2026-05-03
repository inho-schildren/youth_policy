import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
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
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]


def load_result_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_to_ragas_dataset(results):
    rows = []
    skipped = 0

    for item in results:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        ground_truth = str(item.get("ground_truth", "")).strip()
        contexts = item.get("contexts", [])

        if isinstance(contexts, str):
            contexts = [contexts]

        contexts = [
            str(c).strip()
            for c in contexts
            if str(c).strip()
        ]

        if not question or not answer or not contexts:
            skipped += 1
            continue

        rows.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

    print(f"✅ RAGAS 변환 완료: {len(rows)}개 / 제외 {skipped}개")

    if len(rows) == 0:
        raise ValueError("평가 가능한 데이터가 0개입니다. result JSON의 question, answer, contexts를 확인하세요.")

    return Dataset.from_list(rows)


def get_detail_path(result_path: Path):
    return SCORES_DIR / f"detail_{result_path.stem}.csv"


def summarize_detail_csv(detail_path: Path, result_path: Path):
    df = pd.read_csv(detail_path)

    metric_cols = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    missing_cols = [col for col in metric_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️ metric 컬럼 없음, summary 제외: {detail_path.name} / {missing_cols}")
        return None

    mean_scores = df[metric_cols].mean(numeric_only=True)

    return {
        "experiment": result_path.stem.replace("rag_result_", ""),
        "faithfulness": float(mean_scores["faithfulness"]),
        "answer_relevancy": float(mean_scores["answer_relevancy"]),
        "context_precision": float(mean_scores["context_precision"]),
        "context_recall": float(mean_scores["context_recall"]),
    }


def evaluate_one_result_file(result_path: Path):
    detail_path = get_detail_path(result_path)

    # =========================
    # 이미 detail CSV 있으면 스킵
    # =========================
    if detail_path.exists():
        print(f"⏩ 이미 평가됨 → 스킵: {result_path.name}")
        return summarize_detail_csv(detail_path, result_path)

    print(f"\n🚀 평가 시작: {result_path.name}")

    results = load_result_json(result_path)
    dataset = convert_to_ragas_dataset(results)

    print(f"📊 평가 데이터 개수: {len(dataset)}")

    score = evaluate(
        dataset=dataset,
        metrics=METRICS,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    df = score.to_pandas()

    df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    summary = summarize_detail_csv(detail_path, result_path)

    print("✅ 완료:", summary)

    return summary


def main():
    result_files = sorted(RESULTS_DIR.glob("rag_result_*.json"))

    if not result_files:
        print(f"❌ 결과 파일 없음: {RESULTS_DIR}")
        return

    summaries = []

    for path in result_files:
        summary = evaluate_one_result_file(path)

        if summary is not None:
            summaries.append(summary)

    if not summaries:
        print("❌ summary로 만들 데이터가 없습니다.")
        return

    summary_df = pd.DataFrame(summaries)

    summary_df = summary_df.sort_values(
        by="faithfulness",
        ascending=False
    )

    summary_path = SCORES_DIR / "ragas_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n🏆 RAGAS 평가 결과")
    print(summary_df)

    print(f"\n💾 저장 완료: {summary_path}")


if __name__ == "__main__":
    main()