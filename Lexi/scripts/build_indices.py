from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import PATHS, RETRIEVAL
from retrieval.bm25_index import BM25Index
from retrieval.corpus_loader import load_corpus
from retrieval.vector_index import VectorIndex


def main() -> None:
    load_dotenv()
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY not found in environment. Required for embedding builds.')

    docs = load_corpus(PATHS.metadata_dir, PATHS.parsed_text_dir)
    bm25 = BM25Index.from_corpus(docs)
    bm25.save(PATHS.bm25_index_path)

    vector = VectorIndex.build(docs=docs, api_key=api_key, model=RETRIEVAL.embedding_model)
    vector.save(PATHS.vector_index_path)
    print(f'Built BM25 index -> {PATHS.bm25_index_path}')
    print(f'Built vector index -> {PATHS.vector_index_path}')


if __name__ == '__main__':
    main()
