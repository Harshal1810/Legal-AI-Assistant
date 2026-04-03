from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from retrieval.corpus_loader import CorpusDocument


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section: str
    text: str
    metadata: Dict[str, Any]


def _split_text(text: str, target_chars: int = 1800, overlap: int = 250) -> List[str]:
    text = (text or '').strip()
    if not text:
        return []
    if len(text) <= target_chars:
        return [text]
    out: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + target_chars, len(text))
        out.append(text[start:end])
        if end >= len(text):
            break
        start += max(1, target_chars - overlap)
    return out


def build_chunks(doc: CorpusDocument) -> List[Chunk]:
    meta = doc.metadata
    chunks: List[Chunk] = []
    section_map = {
        'summary': meta.get('embedding_summary') or meta.get('summary') or '',
        'facts': meta.get('facts') or '',
        'ratio': meta.get('ratio_decidendi') or '',
        'order': meta.get('final_order') or '',
        'arguments_claimant': meta.get('arguments_claimant') or '',
        'arguments_respondent': meta.get('arguments_respondent') or '',
    }

    for section, text in section_map.items():
        for idx, piece in enumerate(_split_text(text, target_chars=1600, overlap=150), start=1):
            chunks.append(Chunk(
                chunk_id=f'{doc.doc_id}:{section}:{idx}',
                doc_id=doc.doc_id,
                section=section,
                text=piece,
                metadata={
                    'case_name': meta.get('case_name', ''),
                    'citation': meta.get('citation', ''),
                    'court': meta.get('court', ''),
                    'year': meta.get('year', 0),
                },
            ))

    for idx, piece in enumerate(_split_text(doc.parsed_text, target_chars=2200, overlap=250), start=1):
        chunks.append(Chunk(
            chunk_id=f'{doc.doc_id}:full:{idx}',
            doc_id=doc.doc_id,
            section='full',
            text=piece,
            metadata={
                'case_name': meta.get('case_name', ''),
                'citation': meta.get('citation', ''),
                'court': meta.get('court', ''),
                'year': meta.get('year', 0),
            },
        ))
    return chunks
