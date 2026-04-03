from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from rank_bm25 import BM25Okapi

from retrieval.corpus_loader import CorpusDocument
from retrieval.chunker import build_chunks


@dataclass
class BM25Record:
    chunk_id: str
    doc_id: str
    case_name: str
    citation: str
    section: str
    text: str


class BM25Index:
    def __init__(self, records: List[BM25Record]):
        self.records = records
        self.tokens = [self._tokenize(r.text) for r in records]
        self.model = BM25Okapi(self.tokens) if self.tokens else None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()

    @classmethod
    def from_corpus(cls, docs: List[CorpusDocument]) -> 'BM25Index':
        records: List[BM25Record] = []
        for doc in docs:
            for chunk in build_chunks(doc):
                records.append(BM25Record(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    case_name=chunk.metadata.get('case_name', ''),
                    citation=chunk.metadata.get('citation', ''),
                    section=chunk.section,
                    text=chunk.text,
                ))
        return cls(records)

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        if not self.model or not self.records:
            return []
        scores = self.model.get_scores(self._tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {'source': 'bm25', 'score': float(score), **asdict(self.records[idx])}
            for idx, score in ranked if score > 0
        ]

    def save(self, path: Path) -> None:
        payload = [asdict(r) for r in self.records]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path: Path) -> 'BM25Index':
        payload = json.loads(path.read_text(encoding='utf-8'))
        return cls([BM25Record(**r) for r in payload])
