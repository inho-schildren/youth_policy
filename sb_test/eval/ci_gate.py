"""CI gate: 평균 점수가 임계치 미만이면 exit(1)."""
from __future__ import annotations
import json, sys
from pathlib import Path
from statistics import mean

THRESHOLDS = {
    "ragas/faithfulness":      0.85,
    "ragas/answer_relevancy":  0.85,
    "retrieval_relevance":     0.80,
    "policy_fit":              0.90,
    # hallucination_rate = 1 - faithfulness  (≤ 0.05 → faithfulness ≥ 0.95)
    "hallucination_rate_max":  0.05,
}

def load(path: str = "eval/_runs.jsonl") -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]


def main() -> None:
    rows = load()
    fail: list[str] = []

    faith = [r["ragas"].get("faithfulness", 0.0) for r in rows]
    relev = [r["ragas"].get("answer_relevancy", 0.0) for r in rows]
    # custom evaluator 결과는 별도 langsmith pull이 필요. 여기서는 ragas만 게이트.
    # TODO: langsmith feedback 평균 조회 → retrieval_relevance, policy_fit 추가 게이트

    f_avg = mean(faith) if faith else 0.0
    r_avg = mean(relev) if relev else 0.0
    halluc = 1.0 - f_avg

    if f_avg < THRESHOLDS["ragas/faithfulness"]:
        fail.append(f"faithfulness {f_avg:.3f} < {THRESHOLDS['ragas/faithfulness']}")
    if r_avg < THRESHOLDS["ragas/answer_relevancy"]:
        fail.append(f"answer_relevancy {r_avg:.3f} < {THRESHOLDS['ragas/answer_relevancy']}")
    if halluc > THRESHOLDS["hallucination_rate_max"]:
        fail.append(f"hallucination_rate {halluc:.3f} > {THRESHOLDS['hallucination_rate_max']}")

    if fail:
        print("CI GATE FAIL:\n  - " + "\n  - ".join(fail))
        sys.exit(1)
    print(f"CI GATE PASS  faithfulness={f_avg:.3f} relevancy={r_avg:.3f} halluc={halluc:.3f}")


if __name__ == "__main__":
    main()
