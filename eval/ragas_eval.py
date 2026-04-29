"""RAGAS 실행 + LangSmith feedback push."""
from __future__ import annotations
import os, json
from typing import Dict, List
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy,
    context_precision, context_recall, answer_correctness,
)
from langsmith import Client

LS = Client()

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness]


def to_ragas_ds(rows: List[Dict]) -> Dataset:
    """rows: [{question, answer, contexts, ground_truth, run_id}]"""
    return Dataset.from_list([{
        "question":     r["question"],
        "answer":       r["answer"],
        "contexts":     r["contexts"],
        "ground_truth": r["ground_truth"],
    } for r in rows])


def run_ragas(rows: List[Dict]) -> Dict:
    ds = to_ragas_ds(rows)
    result = evaluate(ds, metrics=METRICS)
    return result.to_pandas().to_dict(orient="records")


def push_to_langsmith(rows: List[Dict], scores: List[Dict]) -> None:
    for row, sc in zip(rows, scores):
        run_id = row.get("run_id")
        if not run_id:
            continue
        for k, v in sc.items():
            if k in {"question", "answer", "contexts", "ground_truth"}:
                continue
            try:
                LS.create_feedback(run_id=run_id, key=f"ragas/{k}", score=float(v))
            except Exception as e:
                print(f"feedback push fail {k}: {e}")


if __name__ == "__main__":
    # TODO: run_experiment에서 생성된 rows.jsonl을 받아 처리
    path = os.getenv("RAGAS_INPUT", "eval/_runs.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    scores = run_ragas(rows)
    push_to_langsmith(rows, scores)
    print(json.dumps(scores, ensure_ascii=False, indent=2))
