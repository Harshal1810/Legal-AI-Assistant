from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import PATHS
from retrieval.chunker import build_chunks
from retrieval.corpus_loader import load_corpus


def main() -> None:
    docs = load_corpus(PATHS.metadata_dir, PATHS.parsed_text_dir)
    PATHS.chunks_dir.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        chunks = build_chunks(doc)
        payload = [{'chunk_id': c.chunk_id, 'doc_id': c.doc_id, 'section': c.section, 'text': c.text, 'metadata': c.metadata} for c in chunks]
        (PATHS.chunks_dir / f'{doc.doc_id}.chunks.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote chunk artifacts to {PATHS.chunks_dir}')


if __name__ == '__main__':
    main()
