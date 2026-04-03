"""
batch_extract.py — Batch-run the extraction pipeline over all PDFs and save JSON.

Design goals:
- Save ONLY the final JSON per document.
- Print nothing except per-doc progress + cost/tokens.
- Write a costs report to an .xlsx file (Excel).

Examples:
  # OpenAI, cheapest primary + fallback + repair, save JSON + costs.xlsx
  export OPENAI_API_KEY=sk_...
  python batch_extract.py --provider openai --openai-model gpt-5-mini --fallback-model gpt-5.2 --repair-model gpt-4.1-nano --prices-file prices.openai.example.json

  # Groq
  export GROQ_API_KEY=gsk_...
  python batch_extract.py --provider groq --model llama-3.3-70b-versatile
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingestion.extractor import JudgmentExtractor
from ingestion.pdf_parser import parse_pdf


def _load_prices(prices_json: Optional[str], prices_file: Optional[str]) -> dict:
    if prices_json:
        return json.loads(prices_json)
    if prices_file:
        return json.loads(Path(prices_file).read_text(encoding="utf-8"))
    return {}


def _estimate_cost_usd(input_tokens: int, output_tokens: int, price_in: float, price_out: float) -> float:
    return (input_tokens / 1_000_000.0) * price_in + (output_tokens / 1_000_000.0) * price_out


def _cost_from_usage(usage, prices: dict, default_in: Optional[float], default_out: Optional[float]) -> Optional[float]:
    total = 0.0
    for u in usage:
        model_prices = prices.get(u.model) or {}
        price_in = model_prices.get("in", default_in)
        price_out = model_prices.get("out", default_out)
        if price_in is None or price_out is None:
            return None
        total += _estimate_cost_usd(u.input_tokens, u.output_tokens, float(price_in), float(price_out))
    return total


def _write_xlsx(rows: list[dict], out_path: Path) -> None:
    try:
        from openpyxl import Workbook
    except Exception as e:
        raise RuntimeError(
            "Missing dependency for XLSX export. Install: pip install openpyxl"
        ) from e

    wb = Workbook()
    ws = wb.active
    ws.title = "costs"

    headers: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in headers:
                headers.append(k)

    ws.append(headers)
    for row in rows:
        ws.append([row.get(h) for h in headers])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["groq", "openai"], default="openai")

    parser.add_argument("--model", default="llama-3.3-70b-versatile",
                        help="Groq primary model (used when --provider groq)")
    parser.add_argument("--openai-model", default="gpt-5-mini",
                        help="OpenAI primary model (used when --provider openai)")
    parser.add_argument("--fallback-model", default=None,
                        help="Optional fallback model for incomplete outputs")
    parser.add_argument("--repair-model", default=None,
                        help="Optional repair model override")

    parser.add_argument("--pdf-dir", default=None,
                        help="PDF directory (default: auto-detect within this repo)")
    parser.add_argument("--out-dir", default=None,
                        help="Where to save JSON (default: storage/metadata)")
    parser.add_argument("--parsed-dir", default=None,
                        help="Where to save parsed text/sections (default: storage/parsed)")
    parser.add_argument("--only-parsed", action="store_true",
                        help="Only save parsed .txt and .sections.json (skip LLM + skip metadata JSON)")
    parser.add_argument("--costs-xlsx", default=None,
                        help="Where to save costs report (default: storage/costs.xlsx)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip docs whose JSON already exists in out-dir")

    # Truncation controls (None means no cap)
    parser.add_argument("--max-chars", type=int, default=None,
                        help="Optional cap on chars of judgment text sent to the LLM (default: no cap)")
    parser.add_argument("--section-cap-chars", type=int, default=None,
                        help="Optional cap on chars stored per detected section (default: no cap)")
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="Optional max output tokens cap (default: no explicit cap)")

    # Pricing controls (USD per 1M tokens)
    parser.add_argument("--price-in", type=float, default=None)
    parser.add_argument("--price-out", type=float, default=None)
    parser.add_argument("--prices-json", default=None)
    parser.add_argument("--prices-file", default=None)

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    # Keep terminal output clean (only our progress lines).
    logging.getLogger().handlers.clear()
    logging.basicConfig(level=logging.ERROR)

    if args.provider == "groq":
        if not os.environ.get("GROQ_API_KEY"):
            print("ERROR: GROQ_API_KEY not set.")
            return 2
        primary_model = args.model
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY not set.")
            return 2
        primary_model = args.openai_model

    pdf_dir_candidates = [
        base_dir / "lexi_research_take_home_assessment_docs",
        base_dir / "lexi_research_take_home_assignment_docs",
    ]
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else next((p for p in pdf_dir_candidates if p.exists()), None)
    if pdf_dir is None or not pdf_dir.exists():
        print("ERROR: PDF directory not found. Use --pdf-dir.")
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else (base_dir / "storage" / "metadata")
    parsed_dir = Path(args.parsed_dir) if args.parsed_dir else (base_dir / "storage" / "parsed")
    costs_xlsx = Path(args.costs_xlsx) if args.costs_xlsx else (base_dir / "storage" / "costs.xlsx")
    if not args.only_parsed:
        out_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    prices = _load_prices(args.prices_json, args.prices_file)

    extractor = None
    if not args.only_parsed:
        extractor = JudgmentExtractor(
            provider=args.provider,
            model=primary_model,
            fallback_model=args.fallback_model,
            repair_model=args.repair_model,
            max_chars=args.max_chars,
            max_output_tokens=args.max_output_tokens,
        )

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    rows: list[dict] = []

    total = len(pdfs)
    for idx, pdf_path in enumerate(pdfs, start=1):
        doc_id = pdf_path.stem
        json_path = out_dir / f"{doc_id}.json"
        txt_path = parsed_dir / f"{doc_id}.txt"
        sections_path = parsed_dir / f"{doc_id}.sections.json"
        if args.skip_existing:
            if args.only_parsed and txt_path.exists() and sections_path.exists():
                print(f"[{idx}/{total}] {doc_id}: skipped (parsed exists)")
                continue
            if (not args.only_parsed) and json_path.exists():
                print(f"[{idx}/{total}] {doc_id}: skipped (exists)")
                continue

        status = "ok"
        error = ""
        cost = None
        usage_total_in = 0
        usage_total_out = 0
        usage_total = 0
        used_fallback = False
        used_repair = False

        try:
            parsed = parse_pdf(pdf_path, section_cap_chars=args.section_cap_chars)

            # Save parsed artifacts (full extracted text + detected section snippets)
            txt_path.write_text(parsed.full_text or "", encoding="utf-8")
            sections_path.write_text(
                json.dumps(parsed.detected_sections or {}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            if not args.only_parsed:
                assert extractor is not None
                metadata = extractor.extract(parsed)
                json_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")

                for u in extractor.usage:
                    usage_total_in += int(u.input_tokens)
                    usage_total_out += int(u.output_tokens)
                    usage_total += int(u.total_tokens)
                    if u.phase == "fallback":
                        used_fallback = True
                    if u.phase == "repair":
                        used_repair = True

                cost = _cost_from_usage(extractor.usage, prices, args.price_in, args.price_out)

        except Exception as e:
            status = "error"
            error = str(e)

        # Progress line: doc + cost/tokens only
        if args.only_parsed:
            print(f"[{idx}/{total}] {doc_id}: parsed ({status})")
        else:
            if cost is not None:
                print(f"[{idx}/{total}] {doc_id}: ${cost:.6f}")
            else:
                print(f"[{idx}/{total}] {doc_id}: tokens_in={usage_total_in} tokens_out={usage_total_out} total={usage_total} ({status})")

        row = {
            "doc_id": doc_id,
            "status": status,
            "error": error,
            "provider": args.provider,
            "primary_model": primary_model,
            "fallback_model": args.fallback_model or "",
            "repair_model": args.repair_model or (extractor.repair_model_name if extractor else ""),
            "used_fallback": used_fallback,
            "used_repair": used_repair,
            "input_tokens": usage_total_in,
            "output_tokens": usage_total_out,
            "total_tokens": usage_total,
            "cost_usd": float(cost) if cost is not None else "",
        }
        rows.append(row)

    if not args.only_parsed:
        _write_xlsx(rows, costs_xlsx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
