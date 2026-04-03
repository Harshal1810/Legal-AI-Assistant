"""
test_extraction.py — Run the full extraction pipeline on the available PDFs
and print a human-readable summary. Use this to verify extraction quality
before running the full batch.

Usage:
    export GROQ_API_KEY=your-key
    python test_extraction.py                          # test doc_001 only
    python test_extraction.py --doc doc_053            # specific doc
    python test_extraction.py --all                    # all available docs
    python test_extraction.py --all --save             # save output to storage/
    python test_extraction.py --debug                  # print parsed sections + prompt preview
    python test_extraction.py --debug --save-artifacts # also save parsed text + prompt to storage/

Cost examples (OpenAI):
    export OPENAI_API_KEY=sk_...
    python test_extraction.py --provider openai --openai-model gpt-5-mini --fallback-model gpt-5.2 --repair-model gpt-4.1-nano --report-usage --prices-file prices.openai.example.json

If you see an error like "length limit was reached" (truncated JSON), set an output cap:
    python test_extraction.py --provider openai --openai-model gpt-5-mini --fallback-model gpt-5.2 --repair-model gpt-4.1-nano --max-output-tokens 9000 --report-usage --prices-file prices.openai.example.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from ingestion.pdf_parser import parse_pdf
from ingestion.extractor import JudgmentExtractor
from ingestion.schema import JudgmentMetadata


def print_metadata(m: JudgmentMetadata) -> None:
    """Pretty-print extracted metadata for human review."""
    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  {m.doc_id.upper()}  |  {m.case_name}")
    print(f"  {m.court}  |  {m.year}")
    if m.citation:
        print(f"  Citation: {m.citation}")
    print(sep)

    # Flags table
    flags = {
        "Motor accident":          m.involves_motor_accident,
        "Commercial vehicle":      m.involves_commercial_vehicle,
        "Unlicensed driver":       m.involves_unlicensed_driver,
        "Insurance contested":     m.insurance_contested_liability,
        "Pay & recover applied":   m.pay_and_recover_applied,
        "Involves death":          m.involves_death,
        "Compensation awarded":    m.compensation_awarded,
    }
    flag_line = "  " + "  ".join(
        f"{'[Y]' if v else '[N]'} {k}" for k, v in flags.items()
    )
    print(flag_line)
    print(f"  Outcome: {m.outcome_for_claimant.upper()}"
          + (f"  |  Amount: {m.compensation_amount}" if m.compensation_amount else ""))

    print(f"\n  FACTS\n  {m.facts}\n")
    print(f"  RATIO DECIDENDI\n  {m.ratio_decidendi}\n")
    print(f"  FINAL ORDER\n  {m.final_order}\n")
    print(f"  SUMMARY\n  {m.summary}\n")
    if m.embedding_summary:
        print(f"  EMBEDDING SUMMARY\n  {m.embedding_summary}\n")
    if m.compensation_breakdown:
        print("  COMPENSATION BREAKDOWN")
        for item in m.compensation_breakdown:
            bits = [item.claimant_or_victim]
            if item.claim_type and item.claim_type != "unknown":
                bits.append(f"type={item.claim_type}")
            if item.tribunal_amount:
                bits.append(f"tribunal={item.tribunal_amount}")
            if item.final_amount:
                bits.append(f"final={item.final_amount}")
            if item.interest:
                bits.append(f"interest={item.interest}")
            if item.notes:
                bits.append(f"notes={item.notes}")
            print("    • " + " | ".join(bits))
    print(f"  LEGAL PRINCIPLES")
    for p in m.legal_principles:
        print(f"    • {p}")
    print(f"\n  SECTIONS CITED")
    for s in m.sections_cited:
        print(f"    • {s}")
    if m.cases_cited:
        print(f"\n  CASES CITED")
        for c in m.cases_cited:
            print(f"    • {c}")
    print(f"\n  BM25 text length: {len(m.bm25_text)} chars")
    print(sep)


def _preview(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n… (truncated, total {len(text)} chars)"


def _print_debug(parsed, extractor: JudgmentExtractor, model: str, preview_chars: int) -> None:
    print("\nDEBUG — pdfplumber parse + prompt preview")
    print(f"  Model: {model}")
    print(f"  Doc: {parsed.doc_id}")
    print(f"  Pages: {parsed.page_count}")
    print(f"  Full text chars: {parsed.char_count}")
    print(f"  Token estimate: {parsed.token_estimate}")
    print(f"  Detected sections: {list(parsed.detected_sections.keys())}")
    print("  NOTE: detected section snippets are capped in the parser (see `ingestion/pdf_parser.py`).")

    print("\n--- FULL TEXT (preview) ---")
    print(_preview(parsed.full_text, preview_chars))

    if parsed.detected_sections:
        for name, sec_text in parsed.detected_sections.items():
            print(f"\n--- SECTION: {name.upper()} (preview) ---")
            print(_preview(sec_text, preview_chars))
            sec_tokens = estimate_tokens(sec_text or "")
            capped_hint = " (likely capped)" if len(sec_text or "") >= 15000 else ""
            print(f"[STATS] {name}: chars={len(sec_text or '')} tokens~={sec_tokens}{capped_hint}")

    # Show truncation/injection stats for what goes into the LLM prompt
    truncated_text, trunc_report = parsed.truncated_for_llm_with_report(max_chars=getattr(extractor, "max_chars", 24_000))
    print("\n--- TRUNCATION REPORT (what is sent as judgment_text) ---")
    print(f"max_chars={trunc_report.get('max_chars')}  result_chars={trunc_report.get('result_chars')}  result_tokens~={trunc_report.get('result_tokens_est')}")
    print(f"full_chars={trunc_report.get('full_text_chars')}  full_tokens~={trunc_report.get('full_text_tokens_est')}")
    print(f"header_chars={trunc_report.get('header_chars')}  tail_chars={trunc_report.get('tail_chars')}")
    included = trunc_report.get("included_sections") or []
    if included:
        for item in included:
            print(
                "[INJECT] "
                f"{item['section']}: included_chars={item['included_chars']} tokens~={item['included_tokens_est']} "
                f"(available_chars={item['available_chars']} tokens~={item['available_tokens_est']} cap={item['cap_chars']})"
            )
    else:
        print("(No detected sections injected; likely budget too small or no sections detected.)")

    # Render the exact messages that will be sent to the model
    section_hints = extractor._build_section_hints(parsed)
    judgment_text = truncated_text
    input_vars = {
        "doc_id": parsed.doc_id,
        "judgment_text": judgment_text,
        "section_hints": section_hints,
    }
    messages = extractor.primary_prompt.format_messages(**input_vars)

    print("\n--- PROMPT MESSAGES (preview) ---")
    for idx, msg in enumerate(messages, start=1):
        role = getattr(msg, "type", None) or msg.__class__.__name__
        content = getattr(msg, "content", str(msg))
        print(f"\n[{idx}] {role} (chars={len(content)})")
        print(_preview(content, preview_chars))


def _save_artifacts(
    parsed,
    extractor: JudgmentExtractor,
    out_root: Path,
    preview_chars: Optional[int] = None,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    # Save parsed text + sections
    parsed_dir = out_root / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    (parsed_dir / f"{parsed.doc_id}.txt").write_text(parsed.full_text or "", encoding="utf-8")
    (parsed_dir / f"{parsed.doc_id}.sections.json").write_text(
        json.dumps(parsed.detected_sections or {}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Save prompt (system+human) as rendered text
    prompts_dir = out_root / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    section_hints = extractor._build_section_hints(parsed)
    judgment_text = parsed.truncated_for_llm(max_chars=getattr(extractor, "max_chars", 24_000))
    input_vars = {
        "doc_id": parsed.doc_id,
        "judgment_text": judgment_text,
        "section_hints": section_hints,
    }
    messages = extractor.primary_prompt.format_messages(**input_vars)

    prompt_path = prompts_dir / f"{parsed.doc_id}.primary.txt"
    with prompt_path.open("w", encoding="utf-8") as f:
        for i, msg in enumerate(messages, start=1):
            role = getattr(msg, "type", None) or msg.__class__.__name__
            content = getattr(msg, "content", str(msg))
            f.write(f"[{i}] {role}\n")
            if preview_chars is None:
                f.write(content)
            else:
                f.write(_preview(content, preview_chars))
            f.write("\n\n" + ("-" * 80) + "\n\n")

    print(f"  Saved parse + prompt → {out_root.resolve()}")


def estimate_tokens(text: str) -> int:
    """
    Lightweight token estimate without external deps.
    Rule of thumb: ~4 chars/token for English-ish text.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_message_tokens(messages) -> int:
    total = 0
    for msg in messages:
        content = getattr(msg, "content", "") or ""
        total += estimate_tokens(content)
    return total


def estimate_cost_usd(input_tokens: int, output_tokens: int, price_in: float, price_out: float) -> float:
    """
    Prices are USD per 1M tokens.
    """
    return (input_tokens / 1_000_000.0) * price_in + (output_tokens / 1_000_000.0) * price_out


def load_prices(prices_json: Optional[str], prices_file: Optional[str]) -> dict:
    """
    Returns a mapping:
      { "<model>": {"in": <usd_per_1m_input>, "out": <usd_per_1m_output>} }
    """
    if prices_json:
        return json.loads(prices_json)
    if prices_file:
        path = Path(prices_file)
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def compute_exact_cost_from_usage(usage, prices: dict, default_in: Optional[float], default_out: Optional[float]) -> Optional[float]:
    total = 0.0
    for u in usage:
        p = prices.get(u.model) or {}
        price_in = p.get("in", default_in)
        price_out = p.get("out", default_out)
        if price_in is None or price_out is None:
            return None
        total += estimate_cost_usd(u.input_tokens, u.output_tokens, float(price_in), float(price_out))
    return total


def compute_cost_breakdown(usage, prices: dict, default_in: Optional[float], default_out: Optional[float]) -> Optional[dict]:
    by_phase: dict[str, float] = {}
    by_model: dict[str, float] = {}
    total = 0.0

    for u in usage:
        p = prices.get(u.model) or {}
        price_in = p.get("in", default_in)
        price_out = p.get("out", default_out)
        if price_in is None or price_out is None:
            return None
        cost = estimate_cost_usd(u.input_tokens, u.output_tokens, float(price_in), float(price_out))
        total += cost
        by_phase[u.phase] = by_phase.get(u.phase, 0.0) + cost
        by_model[u.model] = by_model.get(u.model, 0.0) + cost

    return {"total": total, "by_phase": by_phase, "by_model": by_model}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["groq", "openai"], default="groq",
                        help="LLM provider (default: groq)")
    parser.add_argument("--doc", default="doc_001",
                        help="Single doc_id to test (default: doc_001)")
    parser.add_argument("--all", action="store_true",
                        help="Process all PDFs in data/judgments/")
    parser.add_argument("--save", action="store_true",
                        help="Save JSON output to storage/metadata/")
    parser.add_argument("--save-artifacts", action="store_true",
                        help="Save parsed text + prompt to storage/ (debug artifacts)")
    parser.add_argument("--debug", action="store_true",
                        help="Print parsed sections + prompt preview")
    parser.add_argument("--debug-chars", type=int, default=1200,
                        help="Max chars to print per debug block (default: 1200)")
    parser.add_argument("--model", default="llama-3.3-70b-versatile",
                        help="Groq model to use (ignored if --provider openai)")
    parser.add_argument("--openai-model", default="gpt-5.2",
                        help="OpenAI model to use (when --provider openai)")
    parser.add_argument("--fallback-model", default=None,
                        help="Optional fallback model to re-run extraction if output looks incomplete")
    parser.add_argument("--repair-model", default=None,
                        help="Override repair model (provider-specific)")
    parser.add_argument("--max-chars", type=int, default=None,
                        help="Optional cap on chars of judgment text sent to the LLM (default: no cap)")
    parser.add_argument("--section-cap-chars", type=int, default=None,
                        help="Optional cap on chars stored per detected section (default: no cap)")
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="Optional max output tokens cap (default: no explicit cap)")

    parser.add_argument("--estimate-cost", action="store_true",
                        help="Estimate prompt token usage and cost before calling the LLM")
    parser.add_argument("--price-in", type=float, default=None,
                        help="USD per 1M input tokens (for cost estimation)")
    parser.add_argument("--price-out", type=float, default=None,
                        help="USD per 1M output tokens (for cost estimation)")
    parser.add_argument("--prices-json", default=None,
                        help="JSON string mapping model -> {in, out} prices (USD per 1M tokens)")
    parser.add_argument("--prices-file", default=None,
                        help="Path to JSON file mapping model -> {in, out} prices (USD per 1M tokens)")
    parser.add_argument("--assume-output-tokens", type=int, default=1200,
                        help="Assumed output tokens for cost estimation (default: 1200)")
    parser.add_argument("--report-usage", action="store_true",
                        help="Print actual token usage from the API response (and exact cost if prices provided)")
    args = parser.parse_args()

    if args.provider == "groq":
        if not os.environ.get("GROQ_API_KEY"):
            print("ERROR: GROQ_API_KEY not set.")
            print("Run: export GROQ_API_KEY=gsk_...")
            sys.exit(1)
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY not set.")
            print("Run: export OPENAI_API_KEY=sk_...")
            sys.exit(1)

    base_dir = Path(__file__).resolve().parent
    pdf_dir = base_dir / "lexi_research_take_home_assessment_docs"
    out_dir = base_dir / "storage" / "metadata"
    artifacts_root = base_dir / "storage"

    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)

    chosen_model = args.openai_model if args.provider == "openai" else args.model
    extractor = JudgmentExtractor(
        provider=args.provider,
        model=chosen_model,
        fallback_model=args.fallback_model,
        repair_model=args.repair_model,
        max_chars=args.max_chars,
        max_output_tokens=args.max_output_tokens,
    )

    if args.all:
        pdfs = sorted(pdf_dir.glob("*.pdf"))
    else:
        pdfs = [pdf_dir / f"{args.doc}.pdf"]

    print(f"Testing extraction with model: {chosen_model}")
    print(f"Processing {len(pdfs)} document(s)...\n")

    for pdf_path in pdfs:
        if not pdf_path.exists():
            print(f"NOT FOUND: {pdf_path}")
            continue

        try:
            parsed = parse_pdf(pdf_path, section_cap_chars=args.section_cap_chars)
            print(f"Parsed {pdf_path.stem}: {parsed.page_count} pages, "
                  f"~{parsed.token_estimate} tokens, "
                  f"sections={list(parsed.detected_sections.keys())}")

            if args.debug:
                _print_debug(parsed, extractor, chosen_model, args.debug_chars)

            if args.save_artifacts:
                _save_artifacts(parsed, extractor, artifacts_root)

            if args.estimate_cost:
                section_hints = extractor._build_section_hints(parsed)
                judgment_text = parsed.truncated_for_llm(max_chars=args.max_chars)
                input_vars = {
                    "doc_id": parsed.doc_id,
                    "judgment_text": judgment_text,
                    "section_hints": section_hints,
                }
                messages = extractor.primary_prompt.format_messages(**input_vars)
                in_tok = estimate_message_tokens(messages)
                out_tok = args.assume_output_tokens
                print(f"\nESTIMATE — tokens (rough)")
                print(f"  Input tokens:  {in_tok}")
                print(f"  Output tokens: {out_tok} (assumed)")
                if args.price_in is not None and args.price_out is not None:
                    cost = estimate_cost_usd(in_tok, out_tok, args.price_in, args.price_out)
                    print(f"  Cost: ${cost:.6f} (at ${args.price_in}/1M in, ${args.price_out}/1M out)")
                else:
                    print("  Cost: (provide --price-in and --price-out to compute USD estimate)")

            metadata = extractor.extract(parsed)
            print_metadata(metadata)

            if args.report_usage:
                if getattr(extractor, "usage", None):
                    print("\nUSAGE — API reported")
                    for u in extractor.usage:
                        print(f"  {u.phase}: in={u.input_tokens} out={u.output_tokens} total={u.total_tokens} ({u.provider}/{u.model})")
                    prices = load_prices(args.prices_json, args.prices_file)
                    breakdown = compute_cost_breakdown(extractor.usage, prices, args.price_in, args.price_out)
                    if breakdown is not None:
                        print(f"  Exact cost (total): ${breakdown['total']:.6f}")
                        by_phase = breakdown["by_phase"]
                        if by_phase:
                            phase_str = ", ".join(f"{k}=${v:.6f}" for k, v in by_phase.items())
                            print(f"  Breakdown (phase): {phase_str}")
                        by_model = breakdown["by_model"]
                        if by_model and len(by_model) > 1:
                            model_str = ", ".join(f"{k}=${v:.6f}" for k, v in by_model.items())
                            print(f"  Breakdown (model): {model_str}")
                    else:
                        print("  Exact cost: (provide --price-in/--price-out or --prices-json/--prices-file)")
                else:
                    print("\nUSAGE — API reported")
                    print("  (No usage metadata found on the response objects.)")

            if args.save:
                out_path = out_dir / f"{parsed.doc_id}.json"
                with open(out_path, "w") as f:
                    f.write(metadata.model_dump_json(indent=2))
                print(f"  Saved → {out_path.resolve()}")

        except Exception as e:
            print(f"ERROR processing {pdf_path.stem}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
