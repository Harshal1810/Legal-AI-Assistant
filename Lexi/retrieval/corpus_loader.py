from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class CorpusDocument:
    doc_id: str
    metadata: Dict[str, Any]
    parsed_text: str

    @property
    def case_name(self) -> str:
        return self.metadata.get('case_name', self.doc_id)


def load_corpus(metadata_dir: Path, parsed_text_dir: Path) -> List[CorpusDocument]:
    docs: List[CorpusDocument] = []
    # Prefer doc_*.json to avoid ingesting helper files like prev_fetched.json.
    paths = sorted(metadata_dir.glob('doc_*.json')) or sorted(metadata_dir.glob('*.json'))
    for path in paths:
        raw = path.read_bytes()
        text = None
        for enc in ('utf-8', 'utf-8-sig', 'cp1252'):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            # last resort
            text = raw.decode('utf-8', errors='replace')
        meta = json.loads(text)
        doc_id = meta.get('doc_id') or path.stem
        parsed_path = parsed_text_dir / f'{doc_id}.txt'
        parsed_text = parsed_path.read_text(encoding='utf-8') if parsed_path.exists() else ''
        docs.append(CorpusDocument(doc_id=doc_id, metadata=meta, parsed_text=parsed_text))
    return docs
