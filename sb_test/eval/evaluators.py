"""LangSmith custom evaluators."""
from __future__ import annotations
import json, datetime as dt, re
from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from langsmith.schemas import Example, Run
from langsmith.evaluation import EvaluationResult

JUDGE = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _judge(prompt: str) -> float:
    raw = JUDGE.invoke(prompt).content.strip()
    m = re.search(r"[01](?:\.\d+)?", raw)
    return float(m.group()) if m else 0.0


# ── 1. retrieval_relevance ────────────────────────────────
def retrieval_relevance(run: Run, example: Example) -> EvaluationResult:
    q = example.inputs["question"]
    ctxs: List[str] = run.outputs.get("contexts", []) or []
    if not ctxs:
        return EvaluationResult(key="retrieval_relevance", score=0.0)
    scores = []
    for c in ctxs[:8]:
        s = _judge(
            f"질문에 직접 답하는 정보가 문서에 있는가? 0(무관)/0.5(부분)/1(직접). 숫자만.\n"
            f"질문: {q}\n문서: {c[:800]}"
        )
        scores.append(s)
    return EvaluationResult(key="retrieval_relevance", score=sum(scores)/len(scores))


# ── 2. groundedness ───────────────────────────────────────
def groundedness(run: Run, example: Example) -> EvaluationResult:
    ans = run.outputs.get("answer", "")
    ctxs = "\n---\n".join(run.outputs.get("contexts", [])[:8])
    s = _judge(
        "답변의 사실 주장 중 contexts로 뒷받침되는 비율(0~1, 숫자만):\n"
        f"답변:\n{ans[:2000]}\ncontexts:\n{ctxs[:4000]}"
    )
    return EvaluationResult(key="groundedness", score=s)


# ── 3. policy_fit ─────────────────────────────────────────
def policy_fit(run: Run, example: Example) -> EvaluationResult:
    profile = example.inputs["user_profile"]
    expected = example.outputs.get("expected_policies", [])
    ans = run.outputs.get("answer", "")
    s = _judge(
        "추천된 정책들이 user_profile(연령/소득/무주택/지역/주거형태)과 expected_policies 의도와 맞는 비율(0~1, 숫자만):\n"
        f"profile: {json.dumps(profile, ensure_ascii=False)}\n"
        f"expected: {json.dumps(expected, ensure_ascii=False)}\n"
        f"answer:\n{ans[:2500]}"
    )
    return EvaluationResult(key="policy_fit", score=s)


# ── 4. freshness (rule) ───────────────────────────────────
def freshness(run: Run, example: Example) -> EvaluationResult:
    today = dt.date.today()
    ans = run.outputs.get("answer", "")
    expected = example.outputs.get("expected_policies", [])
    bad = 0
    total = 0
    for p in expected:
        if p["title"] in ans:
            total += 1
            try:
                vu = dt.date.fromisoformat(p.get("valid_until", "9999-12-31"))
                if vu < today:
                    bad += 1
            except ValueError:
                pass
    score = 1.0 if total == 0 else 1.0 - (bad / total)
    return EvaluationResult(key="freshness", score=score)


CUSTOM_EVALUATORS = [retrieval_relevance, groundedness, policy_fit, freshness]
