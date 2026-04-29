"""LangSmith dataset 업로드 + evaluator 등록."""
from __future__ import annotations
import json, os
from langsmith import Client
from evaluators import CUSTOM_EVALUATORS

LS = Client()
DATASET_NAME = os.getenv("LS_DATASET", "youth-policy-eval-v1")


def upload_dataset(path: str = "eval/dataset.jsonl") -> str:
    try:
        ds = LS.create_dataset(dataset_name=DATASET_NAME, description="서울 청년 주거·금융 정책 RAG 평가셋")
    except Exception:
        ds = LS.read_dataset(dataset_name=DATASET_NAME)

    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            examples.append({
                "inputs":  {"question": r["question"], "user_profile": r["user_profile"], "input": r["input"]},
                "outputs": {"ground_truth": r["ground_truth"], "expected_policies": r["expected_policies"]},
            })
    LS.create_examples(
        inputs=[e["inputs"] for e in examples],
        outputs=[e["outputs"] for e in examples],
        dataset_id=ds.id,
    )
    return ds.id


if __name__ == "__main__":
    print(upload_dataset())
