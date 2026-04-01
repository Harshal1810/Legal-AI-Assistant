"""
test_extraction.py — Run the full extraction pipeline on the available PDFs
and print a human-readable summary. Use this to verify extraction quality
before running the full batch.

Usage:
    export ANTHROPIC_API_KEY=your-key
    python test_extraction.py                          # test doc_001 only
    python test_extraction.py --doc doc_053            # specific doc
    python test_extraction.py --all                    # all available docs
    python test_extraction.py --all --save             # save output to storage/
"""

import argparse
import json
import os
import sys
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", default="doc_001",
                        help="Single doc_id to test (default: doc_001)")
    parser.add_argument("--all", action="store_true",
                        help="Process all PDFs in data/judgments/")
    parser.add_argument("--save", action="store_true",
                        help="Save JSON output to storage/metadata/")
    parser.add_argument("--model", default="llama-3.3-70b-versatile",
                        help="Groq model to use")
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set.")
        print("Run: export GROQ_API_KEY=gsk_...")
        sys.exit(1)

    pdf_dir = Path("lexi_research_take_home_assessment_docs")
    out_dir = Path("storage/metadata")

    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)

    extractor = JudgmentExtractor(model=args.model)

    if args.all:
        pdfs = sorted(pdf_dir.glob("*.pdf"))
    else:
        pdfs = [pdf_dir / f"{args.doc}.pdf"]

    print(f"Testing extraction with model: {args.model}")
    print(f"Processing {len(pdfs)} document(s)...\n")

    for pdf_path in pdfs:
        if not pdf_path.exists():
            print(f"NOT FOUND: {pdf_path}")
            continue

        try:
            parsed = parse_pdf(pdf_path)
            print(f"Parsed {pdf_path.stem}: {parsed.page_count} pages, "
                  f"~{parsed.token_estimate} tokens, "
                  f"sections={list(parsed.detected_sections.keys())}")

            metadata = extractor.extract(parsed)
            print_metadata(metadata)

            if args.save:
                out_path = out_dir / f"{parsed.doc_id}.json"
                with open(out_path, "w") as f:
                    f.write(metadata.model_dump_json(indent=2))
                print(f"  Saved → {out_path}")

        except Exception as e:
            print(f"ERROR processing {pdf_path.stem}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()