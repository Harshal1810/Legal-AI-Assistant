from __future__ import annotations

import json
from typing import Any


def score_reasoning_with_llm(
    llm_callable,
    query: str,
    benchmark_item: dict[str, Any],
    agent_answer: str,
    predicted_cases: list[str],
) -> dict[str, Any]:
    prompt = {
        "query": query,
        "benchmark_expectations": {
            "gold_relevant_cases": benchmark_item.get("gold_relevant_cases", []),
            "gold_supporting_cases": benchmark_item.get("gold_supporting_cases", []),
            "gold_adverse_cases": benchmark_item.get("gold_adverse_cases", []),
            "gold_mixed_cases": benchmark_item.get("gold_mixed_cases", []),
            "expected_principles": benchmark_item.get("expected_principles", []),
            "expected_risk_themes": benchmark_item.get("expected_risk_themes", []),
            "notes": benchmark_item.get("notes", ""),
        },
        "agent_answer": agent_answer,
        "predicted_cases": predicted_cases,
    }

    raw = llm_callable(prompt)
    if isinstance(raw, str):
        data = json.loads(raw)
    else:
        data = raw

    total = sum(
        int(data.get(k, 0))
        for k in [
            "factual_alignment",
            "legal_principle_accuracy",
            "applicability_reasoning",
            "grounding",
            "nuance",
        ]
    )
    data["total"] = total
    data["normalized"] = total / 10.0
    return data