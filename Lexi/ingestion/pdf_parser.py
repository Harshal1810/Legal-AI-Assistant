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

    def truncated_for_llm(self, max_chars: int | None = 24_000) -> str:
        text, _ = self._build_truncated_for_llm(max_chars=max_chars)
        return text

    def truncated_for_llm_with_report(self, max_chars: int | None = 24_000) -> tuple[str, dict]:
        """
        Return (truncated_text, report) where report describes how much of each
        section was included and rough token estimates.
        """
        return self._build_truncated_for_llm(max_chars=max_chars)

    def _build_truncated_for_llm(self, max_chars: int | None = 24_000) -> tuple[str, dict]:
        """
        Return text safe for LLM context window.
        Prioritises header + detected sections over raw truncation.
        24000 chars ≈ 6000 tokens — fits Claude Haiku / GPT-4o-mini easily.
        """
        report: dict = {
            "max_chars": max_chars,
            "full_text_chars": len(self.full_text or ""),
            "full_text_tokens_est": len(self.full_text or "") // 4,
            "header_chars": 0,
            "tail_chars": 0,
            "included_sections": [],  # list of dicts
            "result_chars": 0,
            "result_tokens_est": 0,
            "no_truncation": False,
        }

        if max_chars is None:
            report["no_truncation"] = True
            report["result_chars"] = len(self.full_text or "")
            report["result_tokens_est"] = len(self.full_text or "") // 4
            return self.full_text, report

        if len(self.full_text) <= max_chars:
            report["result_chars"] = len(self.full_text)
            report["result_tokens_est"] = len(self.full_text) // 4
            return self.full_text, report

        # Smart truncation: keep header, detected sections, and tail
        parts: list[str] = []
        budget = max_chars

        # Always keep first 3000 chars (header, parties, bench)
        header_chunk = self.full_text[:3000]
        parts.append(header_chunk)
        budget -= len(header_chunk)
        report["header_chars"] = len(header_chunk)

        # Keep last 2000 chars (usually the order)
        tail_chunk = self.full_text[-2000:]
        budget -= len(tail_chunk)
        report["tail_chars"] = len(tail_chunk)

        # Fill middle with detected sections in priority order.
        # We prefer ORDER/RATIO because they often contain: final compensation figures,
        # citations, holdings, and the operative directions.
        per_section_cap = {
            "order": 8000,
            "ratio": 8000,
            "facts": 4500,
            "arguments": 3000,
            "issues": 2000,
        }
        priority = ["order", "ratio", "facts", "arguments", "issues"]
        for sec in priority:
            if sec in self.detected_sections and budget > 500:
                cap = per_section_cap.get(sec, 4000)
                take = min(budget, cap)
                available = self.detected_sections[sec] or ""
                included = available[:take]
                parts.append(f"\n\n[SECTION: {sec.upper()}]\n{included}")
                budget -= len(included)

                report["included_sections"].append(
                    {
                        "section": sec,
                        "available_chars": len(available),
                        "available_tokens_est": len(available) // 4,
                        "included_chars": len(included),
                        "included_tokens_est": len(included) // 4,
                        "cap_chars": cap,
                    }
                )

        parts.append(f"\n\n[FINAL SECTION]\n{tail_chunk}")
        out = "\n".join(parts)
        report["result_chars"] = len(out)
        report["result_tokens_est"] = len(out) // 4
        return out, report


def parse_pdf(pdf_path: str | Path, section_cap_chars: int | None = 15000) -> ParsedJudgment:
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
    detected = _detect_sections(full_text, cap_chars=section_cap_chars)

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


def _detect_sections(text: str, cap_chars: int | None = 15000) -> dict[str, str]:
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
        section_text = text[pos:end]
        if cap_chars is not None:
            section_text = section_text[:cap_chars]
        detected[sec] = section_text

    return detected
