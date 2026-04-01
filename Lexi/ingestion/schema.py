"""
schema.py — Pydantic models for judgment extraction.

Every field has a description that is injected into the LLM prompt
by LangChain's with_structured_output. Keep descriptions precise —
the LLM reads them as instructions.
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


class JudgmentMetadata(BaseModel):
    """Structured representation of one Indian court judgment."""

    # ── Identity ──────────────────────────────────────────────────────
    doc_id: str = Field(
        description="Document filename without extension, e.g. 'doc_001'"
    )
    case_name: str = Field(
        description=(
            "Full case name exactly as it appears in the judgment header, "
            "e.g. 'United India Insurance Co. Ltd. vs Neelam Devi And Others'"
        )
    )
    court: str = Field(
        description=(
            "Full name of the court that delivered this judgment, "
            "e.g. 'Supreme Court of India', 'High Court of Punjab and Haryana at Chandigarh', "
            "'Motor Accident Claims Tribunal, Jind'"
        )
    )
    year: int = Field(
        description="Year the judgment was delivered as a 4-digit integer, e.g. 2023"
    )
    judgment_date: Optional[str] = Field(
        default=None,
        description="Full date as written in the judgment, e.g. '06 November 2023'. Null if not found."
    )
    citation: Optional[str] = Field(
        default=None,
        description=(
            "Neutral citation or case number if present, "
            "e.g. '2023:PHHC:141930' or 'FAO-1113-2018 (O&M)'. Null if not found."
        )
    )

    # ── Boolean flags for fast metadata filtering ─────────────────────
    involves_motor_accident: bool = Field(
        description="True if the case involves a motor vehicle accident claim"
    )
    involves_commercial_vehicle: bool = Field(
        description=(
            "True if the accident involved a commercial vehicle such as a truck, "
            "bus, tanker, goods carriage, or any vehicle used for commercial purposes"
        )
    )
    involves_unlicensed_driver: bool = Field(
        description=(
            "True if the driver of the offending vehicle did not hold a valid and "
            "effective driving license at the time of the accident, including expired licenses"
        )
    )
    insurance_contested_liability: bool = Field(
        description=(
            "True if the insurance company contested its liability to pay compensation, "
            "e.g. argued policy was void or sought to recover from the insured"
        )
    )
    pay_and_recover_applied: bool = Field(
        description=(
            "True if the court applied the 'pay and recover' or 'pay first, recover later' "
            "doctrine — directing the insurer to pay the claimant and then recover from the owner/driver"
        )
    )
    involves_death: bool = Field(
        description="True if the accident resulted in the death of one or more persons"
    )
    involves_injury: bool = Field(
        description="True if the accident resulted in injuries (not death) to one or more persons"
    )
    compensation_awarded: bool = Field(
        description="True if the court awarded or upheld monetary compensation to the claimant(s)"
    )

    # ── Outcome ───────────────────────────────────────────────────────
    outcome_for_claimant: Literal["won", "lost", "partial", "remanded", "unclear"] = Field(
        description=(
            "Final outcome from the claimant's perspective. "
            "'won' = full relief granted. "
            "'lost' = claim dismissed. "
            "'partial' = some but not all relief granted, or compensation modified. "
            "'remanded' = matter sent back to lower court. "
            "'unclear' = cannot determine from the text."
        )
    )
    compensation_amount: Optional[str] = Field(
        default=None,
        description=(
            "Total compensation amount awarded as a string with currency symbol, "
            "e.g. 'Rs. 40,60,400/-'. Include all components if itemised. Null if not awarded."
        )
    )

    # ── Substantive legal content ──────────────────────────────────────
    facts: str = Field(
        description=(
            "3–5 sentence summary of the key background facts: who was involved, "
            "what happened, what injury/loss occurred, and what claim was filed. "
            "Be specific about the type of vehicle, nature of accident, and parties."
        )
    )
    arguments_claimant: str = Field(
        description=(
            "2–4 sentence summary of the main legal and factual arguments made "
            "by the claimant or petitioner. Focus on the core legal positions, "
            "not procedural history."
        )
    )
    arguments_respondent: str = Field(
        description=(
            "2–4 sentence summary of the main legal and factual arguments made "
            "by the respondent (usually the insurance company or vehicle owner). "
            "Focus on the defences raised."
        )
    )
    ratio_decidendi: str = Field(
        description=(
            "The actual legal holding of the court — the principle of law on which "
            "the decision rests. This is the single most important field. "
            "Write 4–6 sentences covering: (1) the legal question decided, "
            "(2) the rule or principle the court applied, "
            "(3) how it applied to these facts, "
            "(4) any important qualifications or conditions. "
            "Quote key phrases from the judgment where they capture the rule precisely."
        )
    )
    final_order: str = Field(
        description=(
            "1–2 sentences stating exactly what was ordered: e.g. appeal allowed/dismissed, "
            "compensation amount confirmed/modified, liability on insurer/owner, "
            "any pay-and-recover direction."
        )
    )
    summary: str = Field(
        description=(
            "3–4 sentence plain-English summary of the entire case suitable for a "
            "lawyer doing quick triage. Should capture: type of case, key legal issue, "
            "court's decision, and practical significance. "
            "Do NOT use legal jargon — write as if explaining to a client."
        )
    )

    # ── Legal principles and citations ────────────────────────────────
    legal_principles: list[str] = Field(
        description=(
            "List of specific legal principles, doctrines, or rules applied or established "
            "by this judgment. Each entry should be a short phrase, e.g.: "
            "'pay and recover doctrine', 'beneficial legislation interpretation', "
            "'insurer cannot escape liability for unlicensed driver', "
            "'future prospects multiplier for compensation'. "
            "Include 3–8 principles. Be specific, not generic."
        )
    )
    sections_cited: list[str] = Field(
        description=(
            "List of statutory sections, rules, or provisions cited in the judgment. "
            "Use the exact citation format, e.g.: "
            "'Section 149(2) Motor Vehicles Act 1988', "
            "'Section 166 MV Act', 'Rule 9 Central Motor Vehicles Rules 1989'. "
            "Include all that are materially relevant."
        )
    )
    cases_cited: list[str] = Field(
        default_factory=list,
        description=(
            "List of precedent cases cited by the court in its reasoning. "
            "Format: 'Case Name, (Year) Volume Reporter Page', "
            "e.g. 'National Insurance Co. Ltd. vs Swaran Singh, (2004) 3 SCC 297'. "
            "Include only cases that the court actually relied on, not just mentioned."
        )
    )

    # ── BM25 search text (computed field, set after extraction) ───────
    bm25_text: str = Field(
        default="",
        description=(
            "Concatenation of all searchable text fields for BM25 indexing. "
            "Set programmatically after extraction — do not populate this field."
        )
    )

    def build_bm25_text(self) -> "JudgmentMetadata":
        """
        Build the BM25 search corpus by concatenating all human-readable
        fields. Called after extraction. Returns self for chaining.
        """
        parts = [
            self.case_name,
            self.court,
            str(self.year),
            self.facts,
            self.arguments_claimant,
            self.arguments_respondent,
            self.ratio_decidendi,
            self.final_order,
            self.summary,
            " ".join(self.legal_principles),
            " ".join(self.sections_cited),
            " ".join(self.cases_cited),
        ]
        self.bm25_text = " ".join(filter(None, parts))
        return self

    def to_section_texts(self) -> dict[str, str]:
        """
        Returns the four embeddable sections — each fits within the
        512-token limit of bge-base-en-v1.5.
        Used to build per-section embeddings for Stage 2 retrieval.
        """
        return {
            "summary": self.summary,
            "facts": self.facts,
            "ratio": self.ratio_decidendi,
            "order": f"{self.final_order}. Principles: {', '.join(self.legal_principles)}",
        }