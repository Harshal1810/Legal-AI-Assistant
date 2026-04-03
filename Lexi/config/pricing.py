from __future__ import annotations

"""
Central place for token price assumptions used by the Streamlit UI cost breakdown.

All prices are USD per 1M tokens unless stated otherwise.
Update these to match your OpenAI billing page.
"""

# Embeddings are billed per token (single rate).
EMBEDDING_USD_PER_1M: dict[str, float] = {
    "text-embedding-3-small": 0.02,
}

# Chat/completions style models are often billed with different input/output rates.
CHAT_USD_PER_1M: dict[str, dict[str, float]] = {
    # Examples — update if needed
    "gpt-5-mini": {"in": 0.25, "out": 2.0},
    "gpt-5.2": {"in": 1.75, "out": 14.0},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
    "gpt-4.1-nano": {"in": 0.10, "out": 0.40},
}

