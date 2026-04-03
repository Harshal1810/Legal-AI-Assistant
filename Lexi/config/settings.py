from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field


class AppPaths(BaseModel):
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1])

    @property
    def data_dir(self) -> Path:
        return self.base_dir / 'data'

    @property
    def raw_pdf_dir(self) -> Path:
        return self.data_dir / 'raw_pdfs'

    @property
    def metadata_dir(self) -> Path:
        return self.data_dir / 'metadata'

    @property
    def parsed_text_dir(self) -> Path:
        return self.data_dir / 'parsed_text'

    @property
    def chunks_dir(self) -> Path:
        return self.data_dir / 'chunks'

    @property
    def vector_index_path(self) -> Path:
        return self.data_dir / 'indices' / 'vector_index.json'

    @property
    def bm25_index_path(self) -> Path:
        return self.data_dir / 'indices' / 'bm25_index.json'


class RetrievalSettings(BaseModel):
    embedding_model: str = 'text-embedding-3-small'
    top_k_bm25: int = 12
    top_k_vector: int = 12
    top_k_hybrid: int = 12
    top_k_rerank: int = 8
    dense_weight: float = 0.45
    bm25_weight: float = 0.35
    metadata_weight: float = 0.20


class GenerationSettings(BaseModel):
    qa_max_cases: int = 4
    research_max_cases: int = 6
    max_evidence_spans_per_case: int = 4
    temperature: float = 0.1


PATHS = AppPaths()
RETRIEVAL = RetrievalSettings()
GENERATION = GenerationSettings()
