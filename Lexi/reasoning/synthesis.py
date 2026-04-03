from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate

from llm.provider import LLMFactory

SYSTEM = """You are a legal research assistant. Use only the supplied case packets and retrieved evidence.
Be explicit about supportive, adverse, or mixed value.
Do not invent case facts.
If evidence is thin, say so."""

QA_PROMPT = """Answer the user query using the provided case packets.

User query:
{query}

Case packets:
{case_packets}

Return:
1. Direct answer
2. Key authorities relied on
3. Important caveats
"""

RESEARCH_PROMPT = """Produce a precedent research memo from the supplied case packets.

User query:
{query}

Case packets:
{case_packets}

Return sections:
1. Supporting precedents
2. Adverse precedents
3. Mixed / distinguishable precedents
4. Strategy recommendation
5. Risks and uncertainty
6. Compensation observations (only if relevant)
"""


def generate_answer(provider: str, model: str, api_key: str, query: str, case_packets: List[Dict[str, Any]], mode: str) -> str:
    llm = LLMFactory.chat_model(provider=provider, model=model, api_key=api_key, temperature=0.1)
    prompt = RESEARCH_PROMPT if mode == 'deep_research' else QA_PROMPT
    chain = ChatPromptTemplate.from_messages([('system', SYSTEM), ('human', prompt)]) | llm
    rendered_packets = json.dumps(case_packets, ensure_ascii=False, indent=2)
    msg = chain.invoke({'query': query, 'case_packets': rendered_packets})
    usage = _extract_usage(msg)
    return {'answer': msg.content, 'usage': usage, 'model': model, 'provider': provider}


def _extract_usage(msg) -> Optional[Dict[str, int]]:
    """
    Extract {input_tokens, output_tokens, total_tokens} from a LangChain AIMessage when available.
    """
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
