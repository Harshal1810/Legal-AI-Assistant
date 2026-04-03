from __future__ import annotations

from typing import Dict


DEEP_RESEARCH_TRIGGERS = [
    'supporting precedent', 'adverse precedent', 'strategy', 'risk', 'compensation range',
    'distinguish', 'counter', 'precedent', 'recommendation', 'deep research'
]


class QueryRouter:
    @staticmethod
    def route(query: str) -> Dict[str, str]:
        q = query.lower()
        mode = 'deep_research' if any(t in q for t in DEEP_RESEARCH_TRIGGERS) else 'qa'
        needs_web = any(t in q for t in ['outside corpus', 'outside context', 'what does', 'explain law', 'meaning of'])
        return {
            'mode': mode,
            'needs_web': 'yes' if needs_web else 'no',
            'reason': 'Detected precedent/strategy style query.' if mode == 'deep_research' else 'Detected direct Q&A style query.',
        }
