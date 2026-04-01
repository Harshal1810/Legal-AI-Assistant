# ingestion/parse_legal_pdf.py
import pdfplumber
import re

def parse_judgment(pdf_path: str) -> dict:
    sections = {
        "header": "",      # case name, court, date, bench
        "facts": "",       # background facts
        "issues": "",      # legal issues framed
        "arguments": "",   # arguments of both sides
        "ratio": "",       # ratio decidendi — the binding part
        "order": ""        # final order
    }
    
    # Section detection via keyword heuristics
    SECTION_MARKERS = {
        "facts": ["facts", "brief facts", "background"],
        "issues": ["issues", "framed", "questions"],
        "arguments": ["argued", "submitted", "contended", "counsel"],
        "ratio": ["held", "we are of the view", "ratio", "principle"],
        "order": ["order", "disposed", "allowed", "dismissed"]
    }
    
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    
    # ... section splitting logic
    return {"text": full_text, "sections": sections}

print(parse_judgment("lexi_research_take_home_assessment_docs/doc_001.pdf"))