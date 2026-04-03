from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


def build_summary(results: dict) -> dict:
    per_query = results["per_query"]

    def avg(key: str) -> float:
        vals = [row[key] for row in per_query if key in row]
        return mean(vals) if vals else 0.0

    summary = {
        "num_queries": len(per_query),
        "precision_avg": avg("precision"),
        "recall_avg": avg("recall"),
        "must_find_recall_avg": avg("must_find_recall"),
        "support_precision_avg": avg("support_precision"),
        "support_recall_avg": avg("support_recall"),
        "adverse_recall_avg": avg("adverse_recall"),
        "reasoning_score_avg": avg("reasoning_normalized"),
        "adverse_score_avg": avg("adverse_normalized"),
        "adverse_presence_rate": results.get("adverse_presence_rate", 0.0),
    }
    return summary


def write_markdown_report(results: dict, out_path: Path) -> None:
    summary = results["summary"]
    cost_summary = results.get("cost_summary", {})

    lines = []
    lines.append("# Evaluation Summary")
    lines.append("")
    lines.append(f"- Queries evaluated: **{summary['num_queries']}**")
    lines.append(f"- Precision avg: **{summary['precision_avg']:.3f}**")
    lines.append(f"- Recall avg: **{summary['recall_avg']:.3f}**")
    lines.append(f"- Must-find recall avg: **{summary['must_find_recall_avg']:.3f}**")
    lines.append(f"- Supporting precision avg: **{summary['support_precision_avg']:.3f}**")
    lines.append(f"- Supporting recall avg: **{summary['support_recall_avg']:.3f}**")
    lines.append(f"- Adverse recall avg: **{summary['adverse_recall_avg']:.3f}**")
    lines.append(f"- Reasoning quality avg: **{summary['reasoning_score_avg']:.3f}**")
    lines.append(f"- Adverse honesty avg: **{summary['adverse_score_avg']:.3f}**")
    lines.append(f"- Adverse presence rate: **{summary['adverse_presence_rate']:.3f}**")
    if cost_summary:
        lines.append(f"- Total eval cost (USD): **${cost_summary.get('total_cost_usd', 0.0):.6f}**")
        lines.append(f"- Agent cost (USD): **${cost_summary.get('agent_cost_usd', 0.0):.6f}**")
        lines.append(f"- Judge cost (USD): **${cost_summary.get('judge_cost_usd', 0.0):.6f}**")
    lines.append("")
    lines.append("## Per-query results")
    lines.append("")

    for row in results["per_query"]:
        lines.append(f"### {row['id']} — {row['query']}")
        lines.append(f"- Precision: {row['precision']:.3f}")
        lines.append(f"- Recall: {row['recall']:.3f}")
        lines.append(f"- Must-find recall: {row['must_find_recall']:.3f}")
        lines.append(f"- Support precision: {row['support_precision']:.3f}")
        lines.append(f"- Support recall: {row['support_recall']:.3f}")
        lines.append(f"- Adverse recall: {row['adverse_recall']:.3f}")
        lines.append(f"- Reasoning: {row['reasoning_normalized']:.3f}")
        lines.append(f"- Adverse: {row['adverse_normalized']:.3f}")
        lines.append(f"- Predicted cases: {row['predicted_cases']}")
        lines.append(f"- Predicted adverse: {row['predicted_adverse_cases']}")
        cost = row.get("cost") or {}
        if cost:
            lines.append(f"- Cost (USD): ${cost.get('total_cost_usd', 0.0):.6f} (agent=${cost.get('agent_cost_usd', 0.0):.6f}, judge=${cost.get('judge_cost_usd', 0.0):.6f})")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
