from __future__ import annotations

"""
Evaluation/CLI adapter for running the Lexi pipeline without Streamlit.

The eval harness expects:
{
  "answer": str,
  "predicted_cases": list[str],
  "predicted_supporting_cases": list[str],
  "predicted_adverse_cases": list[str],
  "trace": dict
}
"""

import os
from dataclasses import asdict
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional

from config.settings import PATHS
from reasoning.deep_research_pipeline import DeepResearchPipeline
from reasoning.qa_pipeline import QAPipeline
from reasoning.router import QueryRouter
from retrieval.bm25_index import BM25Index
from retrieval.corpus_loader import load_corpus
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.vector_index import VectorIndex


ProviderName = Literal["openai", "groq"]


@lru_cache(maxsize=1)
def _runtime_objects() -> tuple[dict[str, Any], HybridRetriever]:
    docs = load_corpus(PATHS.metadata_dir, PATHS.parsed_text_dir)
    bm25 = BM25Index.load(PATHS.bm25_index_path)
    vector = VectorIndex.load(PATHS.vector_index_path)
    retriever = HybridRetriever(docs=docs, bm25_index=bm25, vector_index=vector)
    doc_map = {d.doc_id: d for d in docs}
    return doc_map, retriever


def run_agent(query: str, provider: ProviderName, api_key: Optional[str]) -> dict[str, Any]:
    """
    Run retrieval + generation and return eval-friendly outputs.

    Notes:
    - Query embeddings are always computed with OpenAI (vector index model), so
      we need an OpenAI key for retrieval. We use OPENAI_API_KEY from env, and
      fall back to the provided `api_key` if provider==openai.
    - Generation uses `provider` + `api_key` (or env var for that provider if None).
    """
    doc_map, retriever = _runtime_objects()

    embedding_key = os.environ.get("OPENAI_API_KEY") or (api_key if provider == "openai" else None)
    if not embedding_key:
        raise RuntimeError("OPENAI_API_KEY is required for query embeddings during retrieval.")

    gen_key = api_key
    if not gen_key:
        gen_key = os.environ.get("OPENAI_API_KEY") if provider == "openai" else os.environ.get("GROQ_API_KEY")
    if not gen_key:
        raise RuntimeError(f"API key missing for provider={provider!r}. Pass --api-key or set env var.")

    router = QueryRouter.route(query)
    retrieval_trace = retriever.retrieve(query=query, embedding_api_key=embedding_key)

    # Default models (match Streamlit defaults)
    default_model = "gpt-4.1-mini" if provider == "openai" else "llama-3.3-70b-versatile"
    model = os.environ.get("LEXI_INFERENCE_MODEL") or default_model

    if router["mode"] == "deep_research":
        result = DeepResearchPipeline.run(
            query=query,
            retrieval_trace=retrieval_trace,
            doc_map=doc_map,
            provider=provider,
            model=model,
            api_key=gen_key,
        )
    else:
        result = QAPipeline.run(
            query=query,
            retrieval_trace=retrieval_trace,
            doc_map=doc_map,
            provider=provider,
            model=model,
            api_key=gen_key,
        )

    packets = result.get("case_packets") or []
    predicted_cases: List[str] = [p.get("doc_id", "") for p in packets if p.get("doc_id")]
    predicted_supporting_cases: List[str] = [p.get("doc_id", "") for p in packets if p.get("stance") == "supportive"]
    predicted_adverse_cases: List[str] = [p.get("doc_id", "") for p in packets if p.get("stance") == "adverse"]

    trace = {
        "router": router,
        "retrieval": retrieval_trace,
        "packets": packets,
        "generation": result.get("generation") or {},
    }

    generation = result.get("generation") or {}
    agent_usage = {
        "provider": generation.get("provider", provider),
        "model": generation.get("model", model),
        "usage": generation.get("usage"),
    }

    return {
        "answer": result.get("answer", ""),
        "predicted_cases": predicted_cases,
        "predicted_supporting_cases": predicted_supporting_cases,
        "predicted_adverse_cases": predicted_adverse_cases,
        "trace": trace,
        "agent_usage": agent_usage,
    }
