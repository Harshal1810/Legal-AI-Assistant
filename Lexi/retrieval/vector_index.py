from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from openai import OpenAI

from retrieval.corpus_loader import CorpusDocument
from retrieval.chunker import build_chunks


@dataclass
class VectorRecord:
    chunk_id: str
    doc_id: str
    case_name: str
    citation: str
    section: str
    text: str
    vector: List[float]


class VectorIndex:
    def __init__(self, records: List[VectorRecord], model: str):
        self.records = records
        self.model = model
        self.matrix = np.array([r.vector for r in records], dtype=np.float32) if records else np.zeros((0, 0), dtype=np.float32)
        if self.matrix.size:
            norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.matrix = self.matrix / norms

    @staticmethod
    def _client(api_key: str) -> OpenAI:
        return OpenAI(api_key=api_key)

    @classmethod
    def build(cls, docs: List[CorpusDocument], api_key: str, model: str) -> 'VectorIndex':
        client = cls._client(api_key)
        chunks = []
        texts = []
        for doc in docs:
            built = build_chunks(doc)
            chunks.extend(built)
            texts.extend([c.text for c in built])

        vectors: List[List[float]] = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = client.embeddings.create(model=model, input=batch)
            vectors.extend([d.embedding for d in resp.data])

        records = [
            VectorRecord(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                case_name=chunk.metadata.get('case_name', ''),
                citation=chunk.metadata.get('citation', ''),
                section=chunk.section,
                text=chunk.text,
                vector=vec,
            ) for chunk, vec in zip(chunks, vectors)
        ]
        return cls(records, model=model)

    def embed_query(self, query: str, api_key: str) -> np.ndarray:
        client = self._client(api_key)
        resp = client.embeddings.create(model=self.model, input=[query])
        vector = np.array(resp.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vector) or 1.0
        return vector / norm

    @staticmethod
    def _embedding_usage_tokens(resp) -> int:
        """
        Best-effort extraction of embedding token usage from OpenAI SDK response.
        """
        usage = getattr(resp, "usage", None)
        if usage is None:
            return 0
        # embeddings usage typically exposes total_tokens
        for key in ("total_tokens", "prompt_tokens", "input_tokens"):
            val = getattr(usage, key, None)
            if isinstance(val, int):
                return val
        return 0

    def embed_query_with_usage(self, query: str, api_key: str) -> tuple[np.ndarray, int]:
        client = self._client(api_key)
        resp = client.embeddings.create(model=self.model, input=[query])
        vector = np.array(resp.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vector) or 1.0
        return (vector / norm), self._embedding_usage_tokens(resp)

    def search(self, query: str, api_key: str, top_k: int = 10) -> List[Dict]:
        if not self.records:
            return []
        q = self.embed_query(query=query, api_key=api_key)
        scores = self.matrix @ q
        ranked = np.argsort(-scores)[:top_k]
        return [
            {'source': 'vector', 'score': float(scores[idx]), **asdict(self.records[idx])}
            for idx in ranked
        ]

    def search_with_usage(self, query: str, api_key: str, top_k: int = 10) -> tuple[List[Dict], int]:
        """
        Same as search(), but also returns embedding token usage for the query embedding call.
        """
        if not self.records:
            return [], 0
        q, usage_tokens = self.embed_query_with_usage(query=query, api_key=api_key)
        scores = self.matrix @ q
        ranked = np.argsort(-scores)[:top_k]
        hits = [
            {'source': 'vector', 'score': float(scores[idx]), **asdict(self.records[idx])}
            for idx in ranked
        ]
        return hits, usage_tokens

    def save(self, path: Path) -> None:
        payload = {'model': self.model, 'records': [asdict(r) for r in self.records]}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')

    @classmethod
    def load(cls, path: Path) -> 'VectorIndex':
        payload = json.loads(path.read_text(encoding='utf-8'))
        return cls([VectorRecord(**r) for r in payload['records']], model=payload['model'])
