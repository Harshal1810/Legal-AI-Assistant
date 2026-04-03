from __future__ import annotations

from typing import Any, Dict

from config.settings import GENERATION
from reasoning.case_packet_builder import CasePacketBuilder
from reasoning.synthesis import generate_answer


class QAPipeline:
    @staticmethod
    def run(query: str, retrieval_trace: Dict[str, Any], doc_map: Dict[str, Any], provider: str, model: str, api_key: str) -> Dict[str, Any]:
        top_rows = retrieval_trace['hybrid_ranked'][: GENERATION.qa_max_cases]
        packets = [CasePacketBuilder.build(doc_map[row['doc_id']], row, query) for row in top_rows]
        result = generate_answer(provider=provider, model=model, api_key=api_key, query=query, case_packets=packets, mode='qa')
        return {'answer': result['answer'], 'generation': result, 'case_packets': packets}
