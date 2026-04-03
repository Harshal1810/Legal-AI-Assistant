from __future__ import annotations

"""
LLM-based judges used by `evals/run_evals.py`.

The eval metrics call `judge_llm(prompt_payload)` where prompt_payload is a dict
with either:
- reasoning judge payload (contains "benchmark_expectations"), or
- adverse judge payload (contains "gold_adverse_cases").

This module implements a small adapter that calls Groq/OpenAI via LangChain and
returns strict JSON as a Python dict.
"""

import json
import os
import re
from typing import Any, Dict, Literal, Optional

from langchain_core.prompts import ChatPromptTemplate

from evals.judge_prompts import ADVERSE_JUDGE_PROMPT, REASONING_JUDGE_PROMPT
from llm.provider import LLMFactory


ProviderName = Literal["openai", "groq"]


def _extract_json_object(text: str) -> str:
    """
    Extract the first balanced JSON object from a response.
    """
    if not text:
        raise ValueError("Empty judge response.")

    start = text.find("{")
    if start == -1:
        raise ValueError(f"Judge did not return JSON. Got: {text[:200]}")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()

    raise ValueError(f"Unbalanced JSON in judge response. Got: {text[:200]}")


def _judge_prompt(payload: dict[str, Any]) -> str:
    if "benchmark_expectations" in payload:
        return REASONING_JUDGE_PROMPT
    if "gold_adverse_cases" in payload:
        return ADVERSE_JUDGE_PROMPT
    # Default to reasoning judge
    return REASONING_JUDGE_PROMPT


def judge_llm(
    prompt_payload: dict[str, Any],
    provider: ProviderName = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run an LLM judge and return a parsed JSON dict.

    Defaults:
    - provider: openai
    - model: env EVAL_JUDGE_MODEL if set; else gpt-4.1-mini for openai, llama-3.3-70b-versatile for groq
    - api_key: OPENAI_API_KEY / GROQ_API_KEY from env if not provided
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY") if provider == "openai" else os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(f"Missing API key for judge provider={provider!r}.")

    if model is None:
        model = os.environ.get("EVAL_JUDGE_MODEL")
    if not model:
        model = "gpt-4.1-mini" if provider == "openai" else "llama-3.3-70b-versatile"

    judge_instructions = _judge_prompt(prompt_payload)

    # Keep the judge prompt small and deterministic.
    # LangChain prompt templates treat `{...}` as variables, but our judge prompts
    # include JSON examples. Escape braces in the system prompt only.
    system = _escape_braces(judge_instructions.strip()) + "\n\nReturn strict JSON only."
    human = "INPUT PAYLOAD (json):\n{payload_json}"

    llm = LLMFactory.chat_model(provider=provider, model=model, api_key=api_key, temperature=0.0)
    chain = ChatPromptTemplate.from_messages([("system", system), ("human", human)]) | llm

    payload_json = json.dumps(prompt_payload, ensure_ascii=False, indent=2)[:12000]
    msg = chain.invoke({"payload_json": payload_json})
    content = getattr(msg, "content", "") or ""

    json_str = _extract_json_object(content)
    # Strip common trailing commas and non-JSON noise if any (best-effort).
    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*]", "]", json_str)
    data = json.loads(json_str)
    usage = _extract_usage(msg)
    if usage:
        data["_judge_usage"] = {
            "provider": provider,
            "model": model,
            **usage,
        }
    return data


def _extract_usage(msg) -> Optional[Dict[str, int]]:
    if msg is None:
        return None

    usage_md = getattr(msg, "usage_metadata", None) or {}
    if usage_md:
        in_tok = int(usage_md.get("input_tokens", 0) or 0)
        out_tok = int(usage_md.get("output_tokens", 0) or 0)
        tot_tok = int(usage_md.get("total_tokens", in_tok + out_tok) or 0)
        return {"input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": tot_tok}

    resp_md = getattr(msg, "response_metadata", None) or {}
    token_usage = resp_md.get("token_usage") or resp_md.get("usage") or {}
    if token_usage:
        in_tok = int(token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0) or 0)
        out_tok = int(token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0) or 0)
        tot_tok = int(token_usage.get("total_tokens", in_tok + out_tok) or 0)
        return {"input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": tot_tok}

    return None


def _escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")
