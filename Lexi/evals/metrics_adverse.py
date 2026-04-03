from __future__ import annotations

import json
from typing import Any


def score_adverse_with_llm(
    llm_callable,
    query: str,
    benchmark_item: dict[str, Any],
    agent_answer: str,
    predicted_adverse_cases: list[str],
) -> dict[str, Any]:
    prompt = {
        "query": query,
        "gold_adverse_cases": benchmark_item.get("gold_adverse_cases", []),
        "expected_risk_themes": benchmark_item.get("expected_risk_themes", []),
        "agent_answer": agent_answer,
        "predicted_adverse_cases": predicted_adverse_cases,
    }

    raw = llm_callable(prompt)
    if isinstance(raw, str):
        data = json.loads(raw)
    else:
        data = raw

    total = sum(
        int(data.get(k, 0))
        for k in [
            "adverse_presence",
            "adverse_accuracy",
            "risk_honesty",
            "distinction_quality",
        ]
    )
    data["total"] = total
    data["normalized"] = total / 8.0
    return data


def adverse_presence_rate(per_query_results: list[dict]) -> float:
    expected = 0
    surfaced = 0
    for row in per_query_results:
        if row.get("gold_adverse_cases"):
            expected += 1
            if row.get("predicted_adverse_cases"):
                surfaced += 1
    return surfaced / expected if expected else 0.0