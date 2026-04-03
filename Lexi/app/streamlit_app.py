from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ui_components import show_retrieval_trace
from config.pricing import CHAT_USD_PER_1M, EMBEDDING_USD_PER_1M
from config.settings import PATHS
from reasoning.deep_research_pipeline import DeepResearchPipeline
from reasoning.qa_pipeline import QAPipeline
from reasoning.router import QueryRouter
from retrieval.bm25_index import BM25Index
from retrieval.corpus_loader import load_corpus
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.vector_index import VectorIndex

load_dotenv()

st.set_page_config(page_title='Lexi Research Agent', layout='wide')
st.title('Lexi Research Agent')
st.caption('Hybrid RAG over extracted judgment metadata + parsed text, with visible intermediate steps.')

@st.cache_resource
def get_runtime_objects():
    docs = load_corpus(PATHS.metadata_dir, PATHS.parsed_text_dir)
    bm25 = BM25Index.load(PATHS.bm25_index_path)
    vector = VectorIndex.load(PATHS.vector_index_path)
    retriever = HybridRetriever(docs=docs, bm25_index=bm25, vector_index=vector)
    doc_map = {d.doc_id: d for d in docs}
    return doc_map, retriever

provider = "openai"
openai_model = st.sidebar.text_input("OpenAI model", value="gpt-4.1-mini")
openai_api_key = st.sidebar.text_input("OpenAI API key", type="password")
st.sidebar.markdown(
    "This app uses a single OpenAI key for both retrieval (query embeddings) and generation. "
    "You can also set `OPENAI_API_KEY` via Streamlit Secrets / environment."
)

query = st.text_area('Ask a question', height=120, placeholder='e.g. Find supporting and adverse precedents on insurer liability where the commercial vehicle driver lacked a valid licence or endorsement.')
run = st.button('Run analysis', type='primary')

def _cost_chat(model: str, in_tokens: int, out_tokens: int, price_in: float, price_out: float) -> float:
    return (in_tokens / 1_000_000.0) * price_in + (out_tokens / 1_000_000.0) * price_out

def _cost_embed(tokens: int, price_per_1m: float) -> float:
    return (tokens / 1_000_000.0) * price_per_1m

if run:
    doc_map, retriever = get_runtime_objects()
    api_key = (
        openai_api_key
        or st.secrets.get("OPENAI_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    )
    if not api_key:
        st.error(
            "OpenAI API key is missing. Provide it in the sidebar, or set `OPENAI_API_KEY` via Streamlit secrets/env."
        )
        st.stop()

    router = QueryRouter.route(query)
    retrieval_trace = retriever.retrieve(query=query, embedding_api_key=api_key)
    model = openai_model

    if router['mode'] == 'deep_research':
        result = DeepResearchPipeline.run(query=query, retrieval_trace=retrieval_trace, doc_map=doc_map, provider=provider, model=model, api_key=api_key)
    else:
        result = QAPipeline.run(query=query, retrieval_trace=retrieval_trace, doc_map=doc_map, provider=provider, model=model, api_key=api_key)

    st.subheader('Final answer')
    st.write(result['answer'])

    # ── Cost breakdown ────────────────────────────────────────────────
    st.subheader('Cost breakdown')

    embedding_usage = retrieval_trace.get('embedding_usage') or {}
    emb_model = str(embedding_usage.get('model') or '')
    emb_tokens = int(embedding_usage.get('tokens') or 0)

    gen = (result.get('generation') or {})
    gen_usage = gen.get('usage') or {}
    gen_in = int(gen_usage.get('input_tokens') or 0)
    gen_out = int(gen_usage.get('output_tokens') or 0)
    gen_total = int(gen_usage.get('total_tokens') or (gen_in + gen_out))

    # Allow override prices in UI, defaulting from config maps when present.
    with st.expander('Pricing (edit if needed)', expanded=False):
        default_emb_price = EMBEDDING_USD_PER_1M.get(emb_model) if emb_model else None
        emb_price = st.number_input(
            'Embedding price (USD per 1M tokens)',
            min_value=0.0,
            value=float(default_emb_price) if default_emb_price is not None else 0.0,
            step=0.001,
        )

        default_chat = CHAT_USD_PER_1M.get(model)
        chat_price_in = st.number_input(
            'Chat input price (USD per 1M tokens)',
            min_value=0.0,
            value=float(default_chat.get('in')) if default_chat else 0.0,
            step=0.01,
        )
        chat_price_out = st.number_input(
            'Chat output price (USD per 1M tokens)',
            min_value=0.0,
            value=float(default_chat.get('out')) if default_chat else 0.0,
            step=0.01,
        )

    emb_cost = _cost_embed(emb_tokens, emb_price) if emb_tokens and emb_price else 0.0
    gen_cost = _cost_chat(model, gen_in, gen_out, chat_price_in, chat_price_out) if (chat_price_in or chat_price_out) else 0.0
    total_cost = emb_cost + gen_cost

    st.write(
        {
            "embedding_model": emb_model,
            "embedding_tokens": emb_tokens,
            "embedding_cost_usd": round(emb_cost, 6),
            "generation_provider": provider,
            "generation_model": model,
            "prompt_tokens": gen_in,
            "completion_tokens": gen_out,
            "total_tokens": gen_total,
            "generation_cost_usd": round(gen_cost, 6),
            "total_cost_usd": round(total_cost, 6),
        }
    )

    show_retrieval_trace({'router': router, 'retrieval': retrieval_trace, 'packets': result['case_packets']})
