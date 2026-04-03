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
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
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
# Usage / cost tracking
# ─────────────────────────────────────────────────────────────────────

@dataclass
class LLMUsage:
    phase: Literal["primary", "fallback", "repair"]
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


def _extract_usage(raw_message) -> Optional[LLMUsage]:
    """
    Best-effort extraction of token usage from a LangChain AIMessage.
    Works across providers when they populate `usage_metadata`.
    """
    if raw_message is None:
        return None

    usage_md = getattr(raw_message, "usage_metadata", None) or {}
    if usage_md:
        in_tok = int(usage_md.get("input_tokens", 0) or 0)
        out_tok = int(usage_md.get("output_tokens", 0) or 0)
        tot_tok = int(usage_md.get("total_tokens", in_tok + out_tok) or 0)
        return LLMUsage(
            phase="primary",
            provider="",
            model="",
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=tot_tok,
        )

    # Fallback: some providers store usage under response_metadata
    resp_md = getattr(raw_message, "response_metadata", None) or {}
    token_usage = resp_md.get("token_usage") or resp_md.get("usage") or {}
    if token_usage:
        in_tok = int(token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0) or 0)
        out_tok = int(token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0) or 0)
        tot_tok = int(token_usage.get("total_tokens", in_tok + out_tok) or 0)
        return LLMUsage(
            phase="primary",
            provider="",
            model="",
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=tot_tok,
        )

    return None


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
- The embedding_summary field is used for retrieval/embeddings. Make it a complete, query-dense \
  summary of the whole judgment: include parties, court, year, citation/case number, accident \
  date/place, vehicles, injuries/deaths, key statutes/rules, core legal issues, outcome, and \
  compensation amount(s) where available.
- For compensation_amount and compensation_breakdown: use the FINAL operative figures in this \
  judgment (appellate/High Court modifications), not just the Tribunal/MACT figures mentioned in \
  background. If the judgment contains multiple connected claims, include each as a separate item \
  in compensation_breakdown.
- For cases_cited: include ALL precedent cases the court relied upon/applied in its reasoning \
  (not just mentioned), including Supreme Court authorities on compensation calculation if present.
- If a field cannot be determined from the text, use the specified default value.
- The doc_id field must be exactly: {doc_id}

Output requirements:
- Return a single valid JSON object only (no markdown fences, no explanation).
- Use ONLY the schema fields (do not add keys like "case_no", "factual_matrix", etc.).
- Include every field from the schema, even if you must use a default/empty value.
- Use these fallback conventions when the text is unclear:
  - Unknown text fields: "" (empty string)
  - Unknown booleans: false
  - Unknown lists: []
  - Unknown outcome_for_claimant: "unclear"
  - Unknown year: 0

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

KEY MONEY/COMPENSATION LINES (auto-extracted, may include both tribunal and final figures):
{money_lines}

KEY CASE CITATION LINES (auto-extracted):
{case_lines}
"""

# Repair prompt — used when primary extraction partially fails
REPAIR_SYSTEM = """You are a JSON repair assistant. You will be given a broken or \
incomplete JSON object and the schema it should conform to.

Rules:
- Return ONLY a single valid JSON object. No markdown fences. No explanations.
- Use ONLY keys from the schema. Do not invent new keys.
- Include every field required by the schema (use defaults when unknown).
- Fallback conventions:
  - Unknown text fields: "" (empty string)
  - Unknown booleans: false
  - Unknown lists: []
  - Unknown outcome_for_claimant: "unclear"
  - Unknown year: 0
"""

REPAIR_PROMPT = """The following JSON was extracted from a court judgment but is \
malformed, incomplete, or contains extra keys. Fix it to match the required schema.

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
    DEFAULT_PROVIDER: Literal["groq", "openai"] = "groq"

    def __init__(
        self,
        provider: Literal["groq", "openai"] = DEFAULT_PROVIDER,
        model: str = DEFAULT_MODEL,
        fallback_model: Optional[str] = None,
        repair_model: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        max_chars: int | None = 24_000,
        max_output_tokens: Optional[int] = None,
    ):
        self.provider = provider
        self.model_name = model
        self.fallback_model_name = fallback_model
        self.max_retries = max_retries
        self.max_chars = max_chars
        self.max_output_tokens = max_output_tokens
        self.usage: list[LLMUsage] = []

        llm, structured_method = self._build_llm(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        # ── Primary LLM — Groq with JSON structured output ──
        # json_mode instructs Groq to guarantee valid JSON output.
        # with_structured_output then validates it against JudgmentMetadata.
        self.structured_llm = llm.with_structured_output(
            JudgmentMetadata,
            method=structured_method,
            include_raw=True,     # returns {"raw": msg, "parsed": obj, "parsing_error": str}
        )

        # ── Primary extraction chain ──
        self.primary_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", EXTRACTION_PROMPT),
        ])
        self.primary_chain = self.primary_prompt | self.structured_llm

        # ── Optional fallback extraction chain (same prompt, stronger/different model) ──
        self.fallback_chain = None
        if self.fallback_model_name:
            fb_llm, fb_method = self._build_llm(
                model=self.fallback_model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            fb_structured = fb_llm.with_structured_output(
                JudgmentMetadata,
                method=fb_method,
                include_raw=True,
            )
            self.fallback_chain = self.primary_prompt | fb_structured

        # ── Repair LLM — plain text output to fix broken JSON ──
        # Keep the repair prompt small to avoid provider TPM/context limits.
        self.repair_model_name = repair_model or self._default_repair_model()
        self.repair_llm, _ = self._build_llm(
            model=self.repair_model_name,
            temperature=0.0,
            max_output_tokens=max_output_tokens,
        )
        self.repair_prompt = ChatPromptTemplate.from_messages([
            ("system", REPAIR_SYSTEM),
            ("human", REPAIR_PROMPT),
        ])

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
        judgment_text = parsed.truncated_for_llm(max_chars=self.max_chars)

        input_vars = {
            "doc_id": parsed.doc_id,
            "judgment_text": judgment_text,
            "section_hints": section_hints,
        }

        logger.info(f"[{parsed.doc_id}] Extracting — {parsed.token_estimate} est. tokens")
        start = time.perf_counter()

        self.usage = []
        metadata = self._extract_with_retry(input_vars, parsed.doc_id, phase="primary")

        if self.fallback_chain and self._needs_fallback(metadata):
            logger.warning(
                f"[{parsed.doc_id}] Output looks incomplete; retrying with fallback model "
                f"{self.fallback_model_name!r}."
            )
            metadata = self._extract_with_retry(input_vars, parsed.doc_id, phase="fallback")

        # Post-process: build BM25 text, ensure doc_id is correct
        metadata.doc_id = parsed.doc_id
        metadata.build_bm25_text()

        elapsed = time.perf_counter() - start
        logger.info(f"[{parsed.doc_id}] Done in {elapsed:.1f}s — {metadata.case_name}")
        return metadata

    def compute_cost_usd(self, price_in_per_m: float, price_out_per_m: float) -> float:
        """
        Compute total USD cost from recorded usage using prices per 1M tokens.
        Note: this only includes calls made inside `extract()` (primary + repair if used).
        """
        total = 0.0
        for u in self.usage:
            total += (u.input_tokens / 1_000_000.0) * price_in_per_m
            total += (u.output_tokens / 1_000_000.0) * price_out_per_m
        return total

    def _needs_fallback(self, m: JudgmentMetadata) -> bool:
        """
        Heuristic completeness check. Defaults in the schema prevent hard failures,
        so we use this to decide when to re-run extraction with a stronger model.
        """
        if not m.case_name or not m.court or m.year == 0:
            return True

        # These fields are the "meat" for RAG/retrieval.
        if len((m.facts or "").strip()) < 200:
            return True
        if len((m.ratio_decidendi or "").strip()) < 400:
            return True
        if len((m.final_order or "").strip()) < 120:
            return True
        if len((m.embedding_summary or "").strip()) < 350:
            return True

        # Avoid empty citations/principles/sections when the doc clearly has them.
        if not m.legal_principles:
            return True
        if not m.sections_cited:
            return True

        return False

    def _default_repair_model(self) -> str:
        if self.provider == "openai":
            # Reasonable default; override via `repair_model=` if desired.
            return "gpt-4.1-mini"
        return "llama-3.1-8b-instant"

    def _build_llm(self, model: str, temperature: float, max_output_tokens: Optional[int]):
        """
        Returns (chat_model, structured_method).
        - Groq: use json_mode (most reliable on Groq for structured output)
        - OpenAI: prefer json_schema
        """
        if self.provider == "openai":
            try:
                from langchain_openai import ChatOpenAI  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "OpenAI provider selected but dependency is missing. "
                    "Install: pip install langchain-openai openai"
                ) from e

            kwargs = {
                "model": model,
                "temperature": temperature,
                # api_key read from OPENAI_API_KEY env var automatically
            }
            if max_output_tokens is not None:
                kwargs["max_tokens"] = max_output_tokens

            return (ChatOpenAI(**kwargs), "json_schema")

        kwargs = {
            "model": model,
            "temperature": temperature,
            # api_key read from GROQ_API_KEY env var automatically
        }
        if max_output_tokens is not None:
            kwargs["max_tokens"] = max_output_tokens

        return (ChatGroq(**kwargs), "json_mode")

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
        self, input_vars: dict, doc_id: str, phase: Literal["primary", "fallback"] = "primary"
    ) -> JudgmentMetadata:
        """Wrapped with tenacity for Groq rate-limit resilience."""

        chain = self.primary_chain if phase == "primary" else self.fallback_chain
        if chain is None:
            raise RuntimeError("Fallback requested but no fallback chain is configured.")

        model_name = self.model_name if phase == "primary" else (self.fallback_model_name or "")
        result = chain.invoke(input_vars)

        # with_structured_output(include_raw=True) returns a dict:
        # {"raw": AIMessage, "parsed": JudgmentMetadata | None, "parsing_error": str | None}
        parsed_obj: Optional[JudgmentMetadata] = result.get("parsed")
        parsing_error = result.get("parsing_error")
        raw_message = result.get("raw")

        usage = _extract_usage(raw_message)
        if usage is not None:
            usage.phase = phase
            usage.provider = self.provider
            usage.model = model_name
            self.usage.append(usage)

        if parsed_obj is not None:
            return parsed_obj

        # Primary failed — attempt repair
        logger.warning(
            f"[{doc_id}] Primary extraction failed: {parsing_error}. "
            "Attempting repair chain."
        )
        raw_text = raw_message.content if raw_message else ""

        return self._repair(raw_text, doc_id)

    def _repair(self, broken_output: str, doc_id: str) -> JudgmentMetadata:
        """
        Ask the LLM to fix its own broken output.
        Used as a fallback when tool-calling produces invalid JSON.
        """
        broken_json = _extract_json_object(broken_output) or broken_output

        # Concise schema summary (keep small to avoid TPM/context errors).
        schema_info = _schema_summary(max_chars=2200)

        messages = self.repair_prompt.format_messages(
            broken_json=broken_json[:3500],
            schema_info=schema_info,
        )
        repair_msg = self.repair_llm.invoke(messages)
        fixed_json_str = repair_msg.content if repair_msg else ""

        usage = _extract_usage(repair_msg)
        if usage is not None:
            usage.phase = "repair"
            usage.provider = self.provider
            usage.model = self.repair_model_name
            self.usage.append(usage)

        # Strip markdown fences if model added them despite instruction
        fixed_json_str = fixed_json_str.strip()
        if fixed_json_str.startswith("```"):
            fixed_json_str = re.sub(r"^```(?:json)?\n?", "", fixed_json_str)
            fixed_json_str = re.sub(r"\n?```$", "", fixed_json_str)

        try:
            fixed_json_candidate = _extract_json_object(fixed_json_str) or fixed_json_str
            data = json.loads(fixed_json_candidate)
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
        money_lines = _extract_money_lines(parsed.full_text, max_chars=1100)
        case_lines = _extract_case_citation_lines(parsed.full_text, max_chars=1100)
        return SECTION_HINTS_TEMPLATE.format(
            facts=ds.get("facts", "(not detected)")[:1500],
            ratio=ds.get("ratio", "(not detected)")[:2200],
            order=ds.get("order", "(not detected)")[:2200],
            money_lines=money_lines,
            case_lines=case_lines,
        )


def _extract_json_object(text: str) -> Optional[str]:
    """
    Best-effort extraction of a JSON object from a messy model response.
    Returns the first balanced {...} object found, else None.
    """
    if not text:
        return None

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()

    return None


def _schema_summary(max_chars: int = 2000) -> str:
    """
    Compact schema requirements for the repair prompt.
    Keeps size small to avoid provider request-size / TPM issues.
    """
    lines: list[str] = []
    for name, field in JudgmentMetadata.model_fields.items():
        if name == "bm25_text":
            continue

        annotation = getattr(field, "annotation", None)
        type_name = getattr(annotation, "__name__", None) or str(annotation)
        desc = (field.description or "").strip().replace("\n", " ")
        lines.append(f"- {name} ({type_name}): {desc}")

    out = "\n".join(lines)
    if len(out) <= max_chars:
        return out
    return out[:max_chars] + f"\n… (truncated, total {len(out)} chars)"


def _extract_money_lines(text: str, max_chars: int = 1200) -> str:
    """
    Extract a compact set of lines likely to contain compensation figures/interest.
    Keeps output small (used only as a hint block).
    """
    if not text:
        return "(none found)"

    # Capture lines with Rs./₹ and common compensation words.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    picked: list[str] = []
    money_re = re.compile(r"(Rs\.|₹)\s*[\d,]+|interest\s*@?\s*\d+%|\bcompensation\b", re.IGNORECASE)
    for ln in lines:
        if money_re.search(ln):
            # Avoid extremely long lines
            picked.append(ln[:240])
        if len(picked) >= 35:
            break

    if not picked:
        return "(none found)"

    out = "\n".join(f"- {ln}" for ln in picked)
    if len(out) <= max_chars:
        return out
    return out[:max_chars] + f"\n… (truncated, total {len(out)} chars)"


def _extract_case_citation_lines(text: str, max_chars: int = 1200) -> str:
    """
    Extract a compact set of lines likely to contain case citations ("vs"/"versus").
    """
    if not text:
        return "(none found)"

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    picked: list[str] = []

    case_re = re.compile(
        r"\b(vs\.?|versus|v\.)\b|SCC|AIR\s*\(|ACJ|RCR\s*\(|FAO-\d+|CWP-\d+|CRM-\w+",
        re.IGNORECASE,
    )

    for ln in lines:
        if case_re.search(ln) and len(ln) >= 18:
            picked.append(ln[:260])
        if len(picked) >= 40:
            break

    if not picked:
        return "(none found)"

    out = "\n".join(f"- {ln}" for ln in picked)
    if len(out) <= max_chars:
        return out
    return out[:max_chars] + f"\n… (truncated, total {len(out)} chars)"
