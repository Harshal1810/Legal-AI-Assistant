from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.metrics_precision_recall import score_precision_recall
from evals.metrics_reasoning import score_reasoning_with_llm
from evals.metrics_adverse import score_adverse_with_llm, adverse_presence_rate
from evals.report import build_summary, write_markdown_report
from reasoning.agent_runner import run_agent
from llm.judges import judge_llm


ROOT = Path(__file__).resolve().parent
BENCHMARK_PATH = ROOT / "benchmark_lean.json"
RESULTS_DIR = ROOT / "results"


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def main() -> None:
    # Load Lexi .env so OPENAI_API_KEY / GROQ_API_KEY are available for retrieval + judging.
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["groq", "openai"], default="groq")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--benchmark", default=str(BENCHMARK_PATH))
    parser.add_argument("--id", action="append", default=None, help="Run only specific eval id(s). Can be passed multiple times.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N benchmark items (after --id filtering).")
    args = parser.parse_args()

    benchmark = load_benchmark(Path(args.benchmark))
    if args.id:
        want = set(args.id)
        benchmark = [b for b in benchmark if b.get("id") in want]
    if args.limit is not None:
        benchmark = benchmark[: args.limit]
    out_dir = ensure_output_dir()

    per_query = []

    # Use a dedicated judge provider/model if configured, else default to agent provider.
    judge_provider = os.environ.get("EVAL_JUDGE_PROVIDER", args.provider)
    judge_model = os.environ.get("EVAL_JUDGE_MODEL", None)
    judge_api_key = os.environ.get("EVAL_JUDGE_API_KEY", None)

    def judge_callable(payload: dict[str, Any]) -> dict[str, Any]:
        # Prefer dedicated judge key; fall back to the agent key only when providers match.
        api_key = judge_api_key or (args.api_key if judge_provider == args.provider else None)
        return judge_llm(payload, provider=judge_provider, api_key=api_key, model=judge_model)

    # Pricing defaults for eval cost tracking (USD per 1M tokens).
    # Update if your pricing changes.
    DEFAULT_CHAT_PRICE = {
        "gpt-5-mini": {"in": 0.25, "out": 2.00},
    }

    def cost_usd(model: str, in_tokens: int, out_tokens: int) -> float:
        p = DEFAULT_CHAT_PRICE.get(model)
        if not p:
            return 0.0
        return (in_tokens / 1_000_000.0) * p["in"] + (out_tokens / 1_000_000.0) * p["out"]

    for item in benchmark:
        query = item["query"]
        result = run_agent(query=query, provider=args.provider, api_key=args.api_key)

        pr = score_precision_recall(
            predicted_cases=result.get("predicted_cases", []),
            gold_relevant_cases=item.get("gold_relevant_cases", []),
            gold_must_find_cases=item.get("gold_must_find_cases", []),
            predicted_supporting_cases=result.get("predicted_supporting_cases", []),
            gold_supporting_cases=item.get("gold_supporting_cases", []),
            predicted_adverse_cases=result.get("predicted_adverse_cases", []),
            gold_adverse_cases=item.get("gold_adverse_cases", []),
        )

        reasoning = score_reasoning_with_llm(
            llm_callable=judge_callable,
            query=query,
            benchmark_item=item,
            agent_answer=result.get("answer", ""),
            predicted_cases=result.get("predicted_cases", []),
        )

        adverse = score_adverse_with_llm(
            llm_callable=judge_callable,
            query=query,
            benchmark_item=item,
            agent_answer=result.get("answer", ""),
            predicted_adverse_cases=result.get("predicted_adverse_cases", []),
        )

        # Cost tracking (best-effort)
        agent_usage = result.get("agent_usage") or {}
        agent_usage_data = agent_usage.get("usage") or {}
        agent_model = str(agent_usage.get("model") or "")
        agent_in = int(agent_usage_data.get("input_tokens") or 0)
        agent_out = int(agent_usage_data.get("output_tokens") or 0)
        agent_cost = cost_usd(agent_model, agent_in, agent_out)

        rj = reasoning.get("_judge_usage") or {}
        aj = adverse.get("_judge_usage") or {}
        judge_calls = []
        for u in [rj, aj]:
            if u:
                judge_calls.append(u)
        judge_in = sum(int(u.get("input_tokens") or 0) for u in judge_calls)
        judge_out = sum(int(u.get("output_tokens") or 0) for u in judge_calls)
        judge_model_name = str(judge_calls[0].get("model")) if judge_calls else ""
        judge_cost = cost_usd(judge_model_name, judge_in, judge_out) if judge_model_name else 0.0

        row = {
            "id": item["id"],
            "query": query,
            "task_type": item.get("task_type", ""),
            "gold_adverse_cases": item.get("gold_adverse_cases", []),
            "predicted_cases": result.get("predicted_cases", []),
            "predicted_supporting_cases": result.get("predicted_supporting_cases", []),
            "predicted_adverse_cases": result.get("predicted_adverse_cases", []),
            "answer": result.get("answer", ""),
            "trace": result.get("trace", {}),
            "precision": pr.precision,
            "recall": pr.recall,
            "must_find_recall": pr.must_find_recall,
            "support_precision": pr.support_precision,
            "support_recall": pr.support_recall,
            "adverse_recall": pr.adverse_recall,
            "reasoning_raw": reasoning,
            "reasoning_normalized": reasoning["normalized"],
            "adverse_raw": adverse,
            "adverse_normalized": adverse["normalized"],
            "cost": {
                "agent_model": agent_model,
                "agent_input_tokens": agent_in,
                "agent_output_tokens": agent_out,
                "agent_cost_usd": agent_cost,
                "judge_model": judge_model_name,
                "judge_input_tokens": judge_in,
                "judge_output_tokens": judge_out,
                "judge_cost_usd": judge_cost,
                "total_cost_usd": agent_cost + judge_cost,
            },
        }
        per_query.append(row)

        # Lightweight progress print for quick single-run checks
        print(
            f"[{item['id']}] cost=${agent_cost + judge_cost:.6f} "
            f"(agent=${agent_cost:.6f}, judge=${judge_cost:.6f})"
        )

    results = {
        "benchmark_path": str(Path(args.benchmark).resolve()),
        "provider": args.provider,
        "per_query": per_query,
    }
    results["adverse_presence_rate"] = adverse_presence_rate(per_query)
    results["summary"] = build_summary(results)
    results["cost_summary"] = {
        "total_cost_usd": sum((row.get("cost", {}) or {}).get("total_cost_usd", 0.0) for row in per_query),
        "agent_cost_usd": sum((row.get("cost", {}) or {}).get("agent_cost_usd", 0.0) for row in per_query),
        "judge_cost_usd": sum((row.get("cost", {}) or {}).get("judge_cost_usd", 0.0) for row in per_query),
    }

    (out_dir / "eval_run.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown_report(results, out_dir / "eval_summary.md")

    print(f"Saved results to: {out_dir}")


if __name__ == "__main__":
    main()
