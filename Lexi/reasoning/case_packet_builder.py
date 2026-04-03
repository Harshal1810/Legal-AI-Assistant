from __future__ import annotations

from typing import Any, Dict

from config.settings import GENERATION
from retrieval.corpus_loader import CorpusDocument


class CasePacketBuilder:
    @staticmethod
    def build(doc: CorpusDocument, retrieval_row: Dict[str, Any], query: str) -> Dict[str, Any]:
        meta = doc.metadata
        evidence_spans = []
        for section in ['facts', 'ratio_decidendi', 'final_order', 'arguments_respondent']:
            text = (meta.get(section) or '').strip()
            if text:
                evidence_spans.append({'type': section, 'text': text[:1200]})
            if len(evidence_spans) >= GENERATION.max_evidence_spans_per_case:
                break

        stance = 'mixed'
        q = query.lower()
        if meta.get('pay_and_recover_applied') and any(k in q for k in ['insurer', 'insurance', 'liability', 'license', 'licence']):
            stance = 'supportive'
        if meta.get('outcome_for_claimant') == 'lost':
            stance = 'adverse'

        return {
            'doc_id': doc.doc_id,
            'case_name': meta.get('case_name', ''),
            'citation': meta.get('citation', ''),
            'court': meta.get('court', ''),
            'judgment_date': meta.get('judgment_date', ''),
            'issue_tags': [k for k in ['involves_motor_accident', 'involves_commercial_vehicle', 'involves_unlicensed_driver', 'insurance_contested_liability', 'pay_and_recover_applied', 'involves_death', 'involves_injury'] if meta.get(k)],
            'outcome_for_claimant': meta.get('outcome_for_claimant', 'unclear'),
            'facts': meta.get('facts', ''),
            'ratio_decidendi': meta.get('ratio_decidendi', ''),
            'legal_principles': meta.get('legal_principles', []),
            'final_order': meta.get('final_order', ''),
            'summary': meta.get('summary', ''),
            'compensation_amount': meta.get('compensation_amount', None),
            'retrieval': retrieval_row,
            'stance': stance,
            'evidence_spans': evidence_spans,
        }
