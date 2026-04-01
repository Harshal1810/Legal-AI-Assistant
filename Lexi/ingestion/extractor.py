"""
extractor.py — LangChain-based LLM extraction of JudgmentMetadata.

Architecture:
  1. Primary chain  : ChatPromptTemplate | ChatGroq.with_structured_output(JudgmentMetadata)
                      Uses json_mode — Groq parses the LLM's JSON output and validates it
                      against the Pydantic schema. No manual JSON parsing needed.
  2. Repair chain   : if primary fails (malformed output), a second prompt asks the LLM
                      to fix its own broken JSON — returns plain text, we parse manually.
  3. Tenacity retry : wraps the whole call with exponential backoff for Groq rate limits.

NOTE on json_mode vs function_calling:
  Groq supports both modes for with_structured_output. We use json_mode because:
  - Works reliably across all Groq models (Llama, Mixtral, Gemma)
  - Slightly faster than function_calling on Groq's infrastructure
  - Requires the system prompt to explicitly instruct JSON output (handled below)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from groq import RateLimitError, APIStatusError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from pydantic import ValidationError

from ingestion.schema import JudgmentMetadata
from ingestion.pdf_parser import ParsedJudgment

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a specialist legal data extraction assistant working with \
Indian court judgments. Your task is to extract structured information accurately and \
completely from the judgment text provided.

Rules:
- Extract information ONLY from the provided text. Do not hallucinate facts.
- For boolean fields: return true only if clearly evidenced in the text.
- For the ratio_decidendi: this is the most important field. Quote the court's exact \
  language where it states the legal principle. Be thorough — 4 to 6 sentences.
- For legal_principles: be specific to this case. Avoid generic entries like \
  "negligence" — instead write "insurer liable despite unlicensed driver under pay and recover".
- If a field cannot be determined from the text, use the specified default value.
- The doc_id field must be exactly: {doc_id}

IMPORTANT: You must respond with a single valid JSON object only. \
No markdown fences, no explanation, no text before or after the JSON."""

EXTRACTION_PROMPT = """Extract all structured information from the following Indian court judgment.

{section_hints}

FULL JUDGMENT TEXT:
{judgment_text}"""

SECTION_HINTS_TEMPLATE = """The following sections were detected in this document \
(use them to locate the relevant parts — the full text is also provided below):

DETECTED FACTS SECTION:
{facts}

DETECTED RATIO/HOLDING SECTION:
{ratio}

DETECTED ORDER SECTION:
{order}
"""

# Repair prompt — used when primary extraction partially fails
REPAIR_SYSTEM = """You are a JSON repair assistant. You will be given a broken or \
incomplete JSON object and the schema it should conform to. Fix it and return ONLY \
valid JSON that matches the schema exactly. Do not add markdown fences."""

REPAIR_PROMPT = """The following JSON was extracted from a court judgment but is \
malformed or incomplete. Fix it to match the required schema.

BROKEN JSON:
{broken_json}

SCHEMA FIELD REQUIREMENTS:
{schema_info}

Return only the fixed, complete JSON object."""


# ─────────────────────────────────────────────────────────────────────
# Extractor class
# ─────────────────────────────────────────────────────────────────────

class JudgmentExtractor:
    """
    Extracts JudgmentMetadata from a ParsedJudgment using LangChain + Groq.

    Uses with_structured_output(method="json_mode") for type-safe extraction.
    Falls back to a repair chain if the primary extraction produces a partial result.
    """

    # Groq model recommendations for this task:
    #   llama-3.3-70b-versatile  — best quality, handles long legal text well
    #   llama-3.1-8b-instant     — 3x faster, good enough for simple docs
    #   mixtral-8x7b-32768       — 32k context window (useful for very long judgments)
    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        self.model_name = model
        self.max_retries = max_retries

        # ── Primary LLM — Groq with JSON structured output ──
        # json_mode instructs Groq to guarantee valid JSON output.
        # with_structured_output then validates it against JudgmentMetadata.
        base_llm = ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=4096,
            # api_key read from GROQ_API_KEY env var automatically
        )

        self.structured_llm = base_llm.with_structured_output(
            JudgmentMetadata,
            method="json_mode",   # use json_mode — most reliable across Groq models
            include_raw=True,     # returns {"raw": msg, "parsed": obj, "parsing_error": str}
        )

        # ── Primary extraction chain ──
        self.primary_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", EXTRACTION_PROMPT),
        ])
        self.primary_chain = self.primary_prompt | self.structured_llm

        # ── Repair LLM — plain text output to fix broken JSON ──
        # Use the fast 8b model for repair — it's just JSON fixing, not reasoning.
        repair_llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=4096,
        )
        self.repair_prompt = ChatPromptTemplate.from_messages([
            ("system", REPAIR_SYSTEM),
            ("human", REPAIR_PROMPT),
        ])
        self.repair_chain = self.repair_prompt | repair_llm | StrOutputParser()

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def extract(self, parsed: ParsedJudgment) -> JudgmentMetadata:
        """
        Extract JudgmentMetadata from a ParsedJudgment.
        Retries on transient errors, attempts repair on parsing failures.

        Returns a fully validated JudgmentMetadata with bm25_text populated.
        Raises RuntimeError if all attempts fail.
        """
        section_hints = self._build_section_hints(parsed)
        judgment_text = parsed.truncated_for_llm(max_chars=24_000)

        input_vars = {
            "doc_id": parsed.doc_id,
            "judgment_text": judgment_text,
            "section_hints": section_hints,
        }

        logger.info(f"[{parsed.doc_id}] Extracting — {parsed.token_estimate} est. tokens")
        start = time.perf_counter()

        metadata = self._extract_with_retry(input_vars, parsed.doc_id)

        # Post-process: build BM25 text, ensure doc_id is correct
        metadata.doc_id = parsed.doc_id
        metadata.build_bm25_text()

        elapsed = time.perf_counter() - start
        logger.info(f"[{parsed.doc_id}] Done in {elapsed:.1f}s — {metadata.case_name}")
        return metadata

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        retry=retry_if_exception_type((RateLimitError, APIStatusError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _extract_with_retry(
        self, input_vars: dict, doc_id: str
    ) -> JudgmentMetadata:
        """Wrapped with tenacity for Groq rate-limit resilience."""

        result = self.primary_chain.invoke(input_vars)

        # with_structured_output(include_raw=True) returns a dict:
        # {"raw": AIMessage, "parsed": JudgmentMetadata | None, "parsing_error": str | None}
        parsed_obj: Optional[JudgmentMetadata] = result.get("parsed")
        parsing_error = result.get("parsing_error")

        if parsed_obj is not None:
            return parsed_obj

        # Primary failed — attempt repair
        logger.warning(
            f"[{doc_id}] Primary extraction failed: {parsing_error}. "
            "Attempting repair chain."
        )
        raw_message = result.get("raw")
        raw_text = raw_message.content if raw_message else ""

        return self._repair(raw_text, doc_id)

    def _repair(self, broken_output: str, doc_id: str) -> JudgmentMetadata:
        """
        Ask the LLM to fix its own broken output.
        Used as a fallback when tool-calling produces invalid JSON.
        """
        # Build a concise schema summary for the repair prompt
        schema_info = "\n".join(
            f"- {name}: {field.description}"
            for name, field in JudgmentMetadata.model_fields.items()
            if name != "bm25_text"  # skip computed field
        )

        fixed_json_str = self.repair_chain.invoke({
            "broken_json": broken_output[:8000],  # cap to avoid huge repair prompts
            "schema_info": schema_info[:3000],
        })

        # Strip markdown fences if model added them despite instruction
        fixed_json_str = fixed_json_str.strip()
        if fixed_json_str.startswith("```"):
            fixed_json_str = re.sub(r"^```(?:json)?\n?", "", fixed_json_str)
            fixed_json_str = re.sub(r"\n?```$", "", fixed_json_str)

        try:
            data = json.loads(fixed_json_str)
            data["doc_id"] = doc_id  # always enforce correct doc_id
            return JudgmentMetadata.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            raise RuntimeError(
                f"[{doc_id}] Repair chain also failed: {e}\n"
                f"Repair output was: {fixed_json_str[:500]}"
            ) from e

    def _build_section_hints(self, parsed: ParsedJudgment) -> str:
        """
        Build the section hints block injected into the prompt.
        If section detection found relevant parts, surface them prominently.
        """
        if not parsed.detected_sections:
            return "(No sections were pre-detected — use the full text below.)"

        ds = parsed.detected_sections
        return SECTION_HINTS_TEMPLATE.format(
            facts=ds.get("facts", "(not detected)")[:1500],
            ratio=ds.get("ratio", "(not detected)")[:2000],
            order=ds.get("order", "(not detected)")[:800],
        )