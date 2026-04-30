"""LangSmith experiment 실행 + RAGAS 결과 metadata 부착."""
from __future__ import annotations
import os, sys, json, datetime as dt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langsmith import Client
from langsmith.evaluation import evaluate as ls_evaluate

from chain.rag_chain import build_chain
# TODO: 실제 retriever 로드 경로에 맞춰 import
from pipeline import housing_retriever, finance_retriever  # noqa

from sb_test.eval.evaluators import CUSTOM_EVALUATORS
from sb_test.eval.ragas_eval import run_ragas, push_to_langsmith

LS = Client()
DATASET_NAME = os.getenv("LS_DATASET", "youth-policy-eval-v1")
EXPERIMENT_PREFIX = os.getenv("LS_EXP", "rag-v")


# ── chain wrapper: LangSmith가 호출 ─────────────────────────
top3_chain, report_chain = build_chain(housing_retriever, finance_retriever)

def target_fn(inputs: dict) -> dict:
    q = inputs["question"]
    # contexts 회수
    h_docs = housing_retriever.invoke(q)[:5]
    f_docs = finance_retriever.invoke(q)[:5]
    contexts = [d.page_content for d in (h_docs + f_docs)]
    answer = report_chain.invoke({"query": q})
    return {"answer": answer, "contexts": contexts}


def main() -> None:
    exp_name = f"{EXPERIMENT_PREFIX}{dt.datetime.now().strftime('%Y%m%d-%H%M')}"
    result = ls_evaluate(
        target_fn,
        data=DATASET_NAME,
        evaluators=CUSTOM_EVALUATORS,
        experiment_prefix=exp_name,
        max_concurrency=4,
        metadata={"chain": "top3+report", "model": "gpt-4o-mini"},
    )

    # RAGAS용 rows 구성
    rows = []
    for r in result:
        rows.append({
            "run_id":       str(r["run"].id),
            "question":     r["example"].inputs["question"],
            "answer":       r["run"].outputs["answer"],
            "contexts":     r["run"].outputs["contexts"],
            "ground_truth": r["example"].outputs["ground_truth"],
        })

    scores = run_ragas(rows)
    push_to_langsmith(rows, scores)

    out = Path("eval/_runs.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for row, sc in zip(rows, scores):
            f.write(json.dumps({**row, "ragas": sc}, ensure_ascii=False) + "\n")
    print(f"experiment={exp_name} rows={len(rows)} → {out}")


if __name__ == "__main__":
    main()
