from __future__ import annotations

import streamlit as st


def show_retrieval_trace(trace: dict) -> None:
    st.subheader('Intermediate steps')
    with st.expander('1. Query routing', expanded=True):
        st.json(trace['router'])
    with st.expander('2. Metadata filtering', expanded=True):
        st.json(trace['retrieval']['filter_trace'])
    with st.expander('3. BM25 hits', expanded=False):
        st.write(trace['retrieval']['bm25_hits'])
    with st.expander('4. Vector hits', expanded=False):
        st.write(trace['retrieval']['vector_hits'])
    with st.expander('5. Hybrid ranking', expanded=True):
        st.write(trace['retrieval']['hybrid_ranked'])
    with st.expander('6. Case packets sent to generation', expanded=False):
        st.write(trace['packets'])
