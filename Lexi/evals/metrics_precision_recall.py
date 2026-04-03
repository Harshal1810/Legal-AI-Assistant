from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class PrecisionRecallResult:
    precision: float
    recall: float
    must_find_recall: float
    support_precision: float
    support_recall: float
    adverse_recall: float


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _to_set(values: Iterable[str]) -> set[str]:
    return {v for v in values if v}


def score_precision_recall(
    predicted_cases: list[str],
    gold_relevant_cases: list[str],
    gold_must_find_cases: list[str],
    predicted_supporting_cases: list[str] | None = None,
    gold_supporting_cases: list[str] | None = None,
    predicted_adverse_cases: list[str] | None = None,
    gold_adverse_cases: list[str] | None = None,
) -> PrecisionRecallResult:
    pred = _to_set(predicted_cases)
    gold = _to_set(gold_relevant_cases)
    must = _to_set(gold_must_find_cases)

    pred_support = _to_set(predicted_supporting_cases or [])
    gold_support = _to_set(gold_supporting_cases or [])

    pred_adverse = _to_set(predicted_adverse_cases or [])
    gold_adverse = _to_set(gold_adverse_cases or [])

    precision = _safe_div(len(pred & gold), len(pred))
    recall = _safe_div(len(pred & gold), len(gold))
    must_find_recall = _safe_div(len(pred & must), len(must))

    support_precision = _safe_div(len(pred_support & gold_support), len(pred_support))
    support_recall = _safe_div(len(pred_support & gold_support), len(gold_support))

    adverse_recall = _safe_div(len(pred_adverse & gold_adverse), len(gold_adverse))

    return PrecisionRecallResult(
        precision=precision,
        recall=recall,
        must_find_recall=must_find_recall,
        support_precision=support_precision,
        support_recall=support_recall,
        adverse_recall=adverse_recall,
    )