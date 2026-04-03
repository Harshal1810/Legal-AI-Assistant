from __future__ import annotations

from typing import Any, Dict, List

from retrieval.corpus_loader import CorpusDocument


BOOLEAN_KEYWORDS = {
    'involves_commercial_vehicle': ['commercial vehicle', 'truck', 'tanker', 'bus', 'goods carriage'],
    'involves_unlicensed_driver': ['unlicensed', 'invalid licence', 'invalid license', 'driving licence', 'driving license', 'endorsement'],
    'insurance_contested_liability': ['insurer liability', 'insurance company', 'policy void', 'breach of policy', 'liability'],
    'pay_and_recover_applied': ['pay and recover', 'recover from owner', 'recover from insured'],
    'involves_death': ['death', 'deceased', 'fatal'],
    'involves_injury': ['injury', 'injured'],
}


def infer_metadata_filters(query: str) -> Dict[str, bool]:
    q = query.lower()
    out: Dict[str, bool] = {}
    for field, patterns in BOOLEAN_KEYWORDS.items():
        if any(p in q for p in patterns):
            out[field] = True
    return out


def apply_metadata_filters(docs: List[CorpusDocument], query: str) -> tuple[List[CorpusDocument], Dict[str, Any]]:
    filters = infer_metadata_filters(query)
    if not filters:
        return docs, {'applied_filters': {}, 'kept_docs': len(docs)}

    filtered = [doc for doc in docs if all(bool(doc.metadata.get(field, False)) == expected for field, expected in filters.items())]
    chosen = filtered or docs
    return chosen, {
        'applied_filters': filters,
        'kept_docs': len(chosen),
        'strict_match_count': len(filtered),
    }
