from __future__ import annotations

REASONING_JUDGE_PROMPT = """You are evaluating the quality of a legal research agent's reasoning.

You will receive:
1. The user query
2. The benchmark expectations
3. The agent answer
4. The cited/supporting precedents identified by the agent

Score the answer on each criterion from 0 to 2:
- factual_alignment: Does the answer accurately describe the cited cases and relevant facts?
- legal_principle_accuracy: Does it correctly state the legal principle/holding?
- applicability_reasoning: Does it explain why the precedent supports, hurts, or is mixed for the query?
- grounding: Does it avoid unsupported claims beyond the supplied benchmark/case references?
- nuance: Does it correctly capture supportive vs adverse vs mixed positioning?

Return strict JSON only:
{
  "factual_alignment": 0-2,
  "legal_principle_accuracy": 0-2,
  "applicability_reasoning": 0-2,
  "grounding": 0-2,
  "nuance": 0-2,
  "summary": "short explanation"
}
"""

ADVERSE_JUDGE_PROMPT = """You are evaluating whether a legal research agent honestly surfaced adverse precedents.

You will receive:
1. The user query
2. Benchmark gold adverse cases
3. The agent answer
4. The cases the agent marked as adverse

Score from 0 to 2:
- adverse_presence: Did the answer surface at least one genuinely adverse precedent when expected?
- adverse_accuracy: Were the adverse precedents actually adverse rather than mislabeled?
- risk_honesty: Did the answer clearly explain the risk posed by the adverse precedent?
- distinction_quality: Did it offer a plausible distinction or mitigation where appropriate?

Return strict JSON only:
{
  "adverse_presence": 0-2,
  "adverse_accuracy": 0-2,
  "risk_honesty": 0-2,
  "distinction_quality": 0-2,
  "summary": "short explanation"
}
"""