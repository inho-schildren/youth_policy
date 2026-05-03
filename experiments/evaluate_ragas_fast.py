import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    # faithfulness,          # ✅ 수정: 1차 평가는 retrieval 중심이라 제외
    # answer_relevancy,      # ✅ 수정: 1차 평가는 답변 품질보다 검색 품질 우선
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
# ✅ 수정: 빠른 평가 옵션
# =========================
EVAL_TAG = "fast"      # detail/summary 파일명 구분용
MAX_EVAL_ROWS = 10               # 빠른 테스트용 평가 개수 제한
MAX_CONTEXTS = 3                 # context 개수 제한
MAX_CONTEXT_CHARS = 1500         # context 하나당 최대 글자 수 제한

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
# ✅ 수정: 1차 평가는 검색 품질 지표만 사용
# =========================
METRICS = [
    context_precision,
    context_recall,
]


def load_result_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# ✅ 수정: 답변 불가 ground_truth 필터링 함수 추가
# =========================
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

        # =========================
        # ✅ 수정: context 개수와 길이 제한
        # - RAGAS 속도 단축
        # - 너무 긴 context로 인한 비용 증가 방지
        # =========================
        contexts = contexts[:MAX_CONTEXTS]
        contexts = [c[:MAX_CONTEXT_CHARS] for c in contexts]

        # =========================
        # ✅ 수정: ground_truth도 필수값으로 검사
        # - context_recall/context_precision은 reference가 중요함
        # =========================
        if not question or not answer or not contexts or not ground_truth:
            skipped += 1
            continue

        # =========================
        # ✅ 수정: "문서 없음" 유형 ground_truth 제외
        # - chunk 성능 비교를 왜곡하는 데이터 제거
        # =========================
        if is_bad_ground_truth(ground_truth):
            skipped += 1
            continue

        rows.append({
            # 기존 RAGAS 구버전 호환 컬럼
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

    # =========================
    # ✅ 수정: 평가 개수 제한
    # - 빠른 1차 평가용
    # =========================
    if MAX_EVAL_ROWS is not None:
        rows = rows[:MAX_EVAL_ROWS]

    print(f"✅ RAGAS 변환 완료: {len(rows)}개 / 제외 {skipped}개")

    if len(rows) == 0:
        raise ValueError(
            "평가 가능한 데이터가 0개입니다. "
            "question, answer, contexts, ground_truth를 확인하세요."
        )

    return Dataset.from_list(rows)


def get_detail_path(result_path: Path):
    # =========================
    # ✅ 수정: 평가 태그를 파일명에 포함
    # - 기존 전체 평가 detail csv와 충돌 방지
    # =========================
    return SCORES_DIR / f"detail_{EVAL_TAG}_{result_path.stem}.csv"


def summarize_detail_csv(detail_path: Path, result_path: Path):
    df = pd.read_csv(detail_path)

    # =========================
    # ✅ 수정: 1차 평가에서 사용하는 metric만 summary 계산
    # =========================
    metric_cols = [
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
        "context_precision": float(mean_scores["context_precision"]),
        "context_recall": float(mean_scores["context_recall"]),
    }


def evaluate_one_result_file(result_path: Path):
    detail_path = get_detail_path(result_path)

    if detail_path.exists():
        print(f"⏩ 이미 평가됨 → 스킵: {result_path.name}")
        return summarize_detail_csv(detail_path, result_path)

    print(f"\n🚀 빠른 1차 평가 시작: {result_path.name}")

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

    # =========================
    # ✅ 수정 선택사항:
    # 실험 파일 개수까지 줄이고 싶으면 아래 주석 해제
    # =========================
    # result_files = result_files[:2]

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

    # =========================
    # ✅ 수정: chunk 1차 평가는 context_recall 우선 정렬
    # - 정답 근거를 잘 가져오는 chunk 방식이 우선
    # =========================
    summary_df = summary_df.sort_values(
        by="context_recall",
        ascending=False
    )

    # =========================
    # ✅ 수정: summary 파일명 분리
    # - 기존 ragas_summary.csv 덮어쓰기 방지
    # =========================
    summary_path = SCORES_DIR / f"ragas_summary_{EVAL_TAG}.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n🏆 RAGAS 빠른 1차 Retrieval 평가 결과")
    print(summary_df)

    print(f"\n💾 저장 완료: {summary_path}")


if __name__ == "__main__":
    main()