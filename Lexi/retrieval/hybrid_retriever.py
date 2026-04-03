from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from config.settings import RETRIEVAL
from retrieval.bm25_index import BM25Index
from retrieval.corpus_loader import CorpusDocument
from retrieval.metadata_filters import apply_metadata_filters
from retrieval.vector_index import VectorIndex


class HybridRetriever:
    def __init__(self, docs: List[CorpusDocument], bm25_index: BM25Index, vector_index: VectorIndex):
        self.docs = docs
        self.doc_map = {d.doc_id: d for d in docs}
        self.bm25_index = bm25_index
        self.vector_index = vector_index

    def retrieve(self, query: str, embedding_api_key: str) -> Dict[str, Any]:
        filtered_docs, filter_trace = apply_metadata_filters(self.docs, query)
        filtered_ids = {d.doc_id for d in filtered_docs}

        bm25_hits = [h for h in self.bm25_index.search(query, top_k=RETRIEVAL.top_k_bm25 * 3) if h['doc_id'] in filtered_ids][: RETRIEVAL.top_k_bm25]
        vector_hits_all, embedding_tokens = self.vector_index.search_with_usage(
            query, api_key=embedding_api_key, top_k=RETRIEVAL.top_k_vector * 3
        )
        vector_hits = [h for h in vector_hits_all if h['doc_id'] in filtered_ids][: RETRIEVAL.top_k_vector]

        aggregate = defaultdict(lambda: {'doc_id': '', 'case_name': '', 'citation': '', 'scores': {}, 'best_text': '', 'best_section': ''})
        for hit in bm25_hits:
            row = aggregate[hit['doc_id']]
            row['doc_id'] = hit['doc_id']
            row['case_name'] = hit.get('case_name', '')
            row['citation'] = hit.get('citation', '')
            row['scores']['bm25'] = hit['score']
            if not row['best_text']:
                row['best_text'] = hit['text']
                row['best_section'] = hit['section']
        for hit in vector_hits:
            row = aggregate[hit['doc_id']]
            row['doc_id'] = hit['doc_id']
            row['case_name'] = hit.get('case_name', '')
            row['citation'] = hit.get('citation', '')
            row['scores']['vector'] = hit['score']
            if not row['best_text']:
                row['best_text'] = hit['text']
                row['best_section'] = hit['section']

        ranked = []
        max_bm25 = max((h['score'] for h in bm25_hits), default=1.0) or 1.0
        max_vector = max((h['score'] for h in vector_hits), default=1.0) or 1.0
        for row in aggregate.values():
            bm25 = row['scores'].get('bm25', 0.0) / max_bm25
            vector = row['scores'].get('vector', 0.0) / max_vector
            meta_bonus = self._metadata_bonus(self.doc_map[row['doc_id']].metadata, query)
            final_score = (RETRIEVAL.bm25_weight * bm25) + (RETRIEVAL.dense_weight * vector) + (RETRIEVAL.metadata_weight * meta_bonus)
            ranked.append({**row, 'normalized_scores': {'bm25': bm25, 'vector': vector, 'metadata_bonus': meta_bonus}, 'hybrid_score': final_score})

        ranked.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return {
            'filter_trace': filter_trace,
            'bm25_hits': bm25_hits,
            'vector_hits': vector_hits,
            'hybrid_ranked': ranked[: RETRIEVAL.top_k_hybrid],
            'embedding_usage': {
                'model': self.vector_index.model,
                'tokens': embedding_tokens,
            },
        }

    @staticmethod
    def _metadata_bonus(metadata: Dict[str, Any], query: str) -> float:
        q = query.lower()
        score = 0.0
        if metadata.get('insurance_contested_liability') and any(k in q for k in ['insurance', 'insurer', 'liability', 'policy']):
            score += 0.3
        if metadata.get('pay_and_recover_applied') and 'recover' in q:
            score += 0.3
        if metadata.get('involves_commercial_vehicle') and any(k in q for k in ['commercial', 'truck', 'tanker', 'goods carriage']):
            score += 0.2
        if metadata.get('involves_unlicensed_driver') and any(k in q for k in ['license', 'licence', 'endorsement', 'unlicensed']):
            score += 0.2
        return min(score, 1.0)
