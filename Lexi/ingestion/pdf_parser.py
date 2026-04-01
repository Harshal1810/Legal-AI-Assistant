"""
pdf_parser.py — Extract and lightly structure text from Indian court judgment PDFs.

We use pdfplumber (better layout preservation than PyPDF2) and apply
heuristic section detection to help the LLM extraction prompt focus
on the right parts of the document.
"""

from __future__ import annotations
import re
import pdfplumber
from pathlib import Path
from dataclasses import dataclass, field


# ── Section marker patterns (covers common Indian judgment structures) ─
SECTION_PATTERNS: dict[str, list[str]] = {
    "header": [
        r"IN THE (?:HIGH|SUPREME) COURT",
        r"BEFORE.*?JUSTICE",
        r"CORAM\s*:",
        r"HON'BLE",
    ],
    "facts": [
        r"BRIEF FACTS",
        r"FACTS OF THE CASE",
        r"BACKGROUND",
        r"brief facts.*?are as follows",
        r"facts.*?in.*?narrow compass",
        r"factual background",
    ],
    "issues": [
        r"ISSUES? FRAMED",
        r"QUESTIONS? FOR (?:CONSIDERATION|DECISION)",
        r"following.*?issues.*?were framed",
        r"point.*?for determination",
    ],
    "arguments": [
        r"(?:learned )?(?:counsel|advocate).*?(?:submit|contend|argu)",
        r"ARGUMENTS? ADVANCED",
        r"SUBMISSIONS? MADE",
        r"it is (?:submitted|contended|argued)",
        r"on (?:the )?other hand",
    ],
    "ratio": [
        r"I HAVE (?:CAREFULLY )?CONSIDERED",
        r"WE (?:ARE )?(?:OF THE )?(?:VIEW|OPINION)",
        r"HAVING (?:REGARD|CONSIDERED)",
        r"it is (?:well )?settled",
        r"the (?:legal )?position is",
        r"ratio.*?decidendi",
        r"the court (?:is of the view|holds|finds)",
    ],
    "order": [
        r"IN THE RESULT",
        r"FOR THE (?:ABOVE )?REASONS",
        r"IN VIEW OF THE ABOVE",
        r"ORDER\s*$",
        r"JUDGMENT\s*$",
        r"(?:appeal|petition|application) is (?:hereby )?(?:allowed|dismissed|disposed)",
    ],
}

# Compile all patterns for efficiency
_COMPILED: dict[str, list[re.Pattern]] = {
    section: [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns]
    for section, patterns in SECTION_PATTERNS.items()
}


@dataclass
class ParsedJudgment:
    doc_id: str
    full_text: str
    page_count: int
    detected_sections: dict[str, str] = field(default_factory=dict)
    char_count: int = 0
    token_estimate: int = 0  # rough: chars / 4

    def truncated_for_llm(self, max_chars: int = 24_000) -> str:
        """
        Return text safe for LLM context window.
        Prioritises header + detected sections over raw truncation.
        24000 chars ≈ 6000 tokens — fits Claude Haiku / GPT-4o-mini easily.
        """
        if len(self.full_text) <= max_chars:
            return self.full_text

        # Smart truncation: keep header, detected sections, and tail
        parts: list[str] = []
        budget = max_chars

        # Always keep first 3000 chars (header, parties, bench)
        header_chunk = self.full_text[:3000]
        parts.append(header_chunk)
        budget -= len(header_chunk)

        # Keep last 2000 chars (usually the order)
        tail_chunk = self.full_text[-2000:]
        budget -= len(tail_chunk)

        # Fill middle with detected sections in priority order
        priority = ["ratio", "facts", "arguments", "issues"]
        for sec in priority:
            if sec in self.detected_sections and budget > 500:
                text = self.detected_sections[sec][:min(budget, 4000)]
                parts.append(f"\n\n[SECTION: {sec.upper()}]\n{text}")
                budget -= len(text)

        parts.append(f"\n\n[FINAL SECTION]\n{tail_chunk}")
        return "\n".join(parts)


def parse_pdf(pdf_path: str | Path) -> ParsedJudgment:
    """
    Extract full text from a court judgment PDF and detect sections.
    Returns a ParsedJudgment ready for the LLM extraction step.
    """
    path = Path(pdf_path)
    doc_id = path.stem  # e.g. "doc_001"

    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if text:
                # Normalise whitespace but preserve paragraph breaks
                text = re.sub(r"[ \t]+", " ", text)
                text = re.sub(r"\n{3,}", "\n\n", text)
                pages.append(text.strip())

    full_text = "\n\n".join(pages)

    # Clean common PDF artifacts in Indian legal docs
    full_text = _clean_indian_judgment_text(full_text)

    # Detect section boundaries
    detected = _detect_sections(full_text)

    return ParsedJudgment(
        doc_id=doc_id,
        full_text=full_text,
        page_count=len(pages),
        detected_sections=detected,
        char_count=len(full_text),
        token_estimate=len(full_text) // 4,
    )


def _clean_indian_judgment_text(text: str) -> str:
    """Remove common noise patterns from Indian court judgment PDFs."""
    # Remove Indian Kanoon footer lines
    text = re.sub(r"Indian Kanoon\s*-\s*http://indiankanoon\.org/doc/\S+", "", text)
    # Remove page number lines like "1 of 26"
    text = re.sub(r"\n\s*\d+\s+of\s+\d+\s*\n", "\n", text)
    # Remove "Downloaded on" lines
    text = re.sub(r":::.*?Downloaded on.*?:::", "", text)
    # Remove "Digitally signed by" blocks
    text = re.sub(r"Digitally\s+signed by.*?(?=\n[A-Z])", "", text, flags=re.DOTALL)
    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _detect_sections(text: str) -> dict[str, str]:
    """
    Find approximate section boundaries using regex markers.
    Returns a dict mapping section name → section text.
    Not perfect — used only to assist the LLM prompt, not as ground truth.
    """
    # Find all marker positions
    hits: list[tuple[int, str]] = []  # (char_position, section_name)

    for section, patterns in _COMPILED.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                hits.append((match.start(), section))

    if not hits:
        return {}

    # Sort by position and deduplicate (keep first hit per section)
    hits.sort(key=lambda x: x[0])
    seen_sections: set[str] = set()
    ordered: list[tuple[int, str]] = []
    for pos, sec in hits:
        if sec not in seen_sections:
            seen_sections.add(sec)
            ordered.append((pos, sec))

    # Slice text between consecutive markers
    detected: dict[str, str] = {}
    for i, (pos, sec) in enumerate(ordered):
        end = ordered[i + 1][0] if i + 1 < len(ordered) else len(text)
        # Cap each section at 5000 chars to avoid runaway sections
        detected[sec] = text[pos:end][:5000]

    return detected