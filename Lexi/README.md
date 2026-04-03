# Lexi Research Agent

Lexi Research Agent is a legal research assistant built for the Lexi Backend Engineer take-home assessment. It works over a corpus of Indian motor accident judgments and supports both:

- **Corpus Q&A**  
  Example: *Which judgments involve commercial vehicles and contested insurer liability?*

- **Deep precedent research**  
  Example: *Find supporting and adverse precedents where the insurer denies liability because the driver lacked a valid licence or endorsement.*

The system is designed to be **inspectable** rather than a black box. The Streamlit UI shows intermediate workflow steps such as:

- query routing
- metadata filtering
- BM25 retrieval hits
- dense retrieval hits
- hybrid ranking
- case packets sent into final answer generation

---

## What the app does

The current system includes:

- hybrid retrieval over a judgment corpus
- metadata-aware search over extracted judgment JSONs
- lexical BM25 retrieval over parsed text
- dense vector retrieval
- query routing between:
  - **Q&A mode**
  - **deep research mode**
- provider-selectable generation:
  - **OpenAI**
  - **Groq**
- benchmark-driven evaluation framework for:
  - precision
  - recall
  - reasoning quality
  - adverse precedent identification

---

## Quickstart

### 0) Install dependencies

```bash
pip install -r requirements.txt
```

### 1) Set keys

Create `Lexi/.env` (or export env vars) with at least:

- `OPENAI_API_KEY=...` (required for dense retrieval query embeddings and index build)
- `GROQ_API_KEY=...` (optional, only if you want Groq for generation/judging)

**Do not commit `.env` to GitHub.** This repo ignores `.env` files via `.gitignore`.

### Deployment note (Streamlit)
If you deploy the Streamlit app, you can provide secrets via:
- Streamlit sidebar inputs (inference + embeddings keys), or
- Streamlit Secrets / environment variables (recommended for production).

### 2) Generate corpus artifacts (PDF → metadata + parsed text)

Write artifacts directly into the runtime locations used by the app/indexers:

```bash
python batch_extract.py --provider openai --openai-model gpt-5-mini --out-dir data/metadata --parsed-dir data/parsed_text
```

This produces:
- `data/metadata/doc_*.json`
- `data/parsed_text/doc_*.txt`
- `data/parsed_text/doc_*.sections.json`

If you already have these artifacts committed/generated, you can skip steps 2–4 and go straight to running the app/evals.

### 3) Build indices (BM25 + vector)

```bash
python scripts/build_indices.py
```

Outputs:
- `data/indices/bm25_index.json`
- `data/indices/vector_index.json`

### 4) (Optional) Build chunk artifacts

```bash
python scripts/build_chunks.py
```

Outputs:
- `data/chunks/doc_*.chunks.json`

### 5) Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

### 6) Run evaluation benchmark (and write report)

Runs the full benchmark and writes:
- `evals/results/<timestamp>/eval_run.json`
- `evals/results/<timestamp>/eval_summary.md`

```bash
python evals/run_evals.py --provider openai
```

## High-level architecture

The pipeline works in this order:

1. **Load corpus artifacts**
   - extracted metadata JSONs
   - parsed full-text `.txt` files

2. **Retrieve relevant material**
   - metadata filters
   - BM25 lexical search
   - dense vector search

3. **Route the query**
   - direct Q&A
   - deep precedent research

4. **Build case packets**
   - structured metadata
   - supporting text spans
   - relevance traces

5. **Generate final answer**
   - using OpenAI or Groq

6. **Show intermediate steps in UI**
   - so the evaluator can inspect how the answer was formed

---

## Repository overview

```text
.
├── app/                     # Streamlit UI
├── config/                  # settings and prompt templates
├── data/
│   ├── raw_pdfs/            # original PDFs (optional at runtime if parsed artifacts exist)
│   ├── metadata/            # extracted metadata JSON files
│   ├── parsed_text/         # cleaned full judgment text
│   ├── chunks/              # optional chunk artifacts
│   └── indices/             # generated BM25 / vector indices
├── storage/                 # optional: extraction/debug artifacts (outside runtime PATHS)
├── evals/
│   ├── benchmark_lean.json
│   ├── run_evals.py
│   ├── results/             # saved eval outputs
│   └── ...
├── ingestion/               # parser, extractor, schema
├── llm/                     # provider abstractions / judge helpers
├── reasoning/               # router, pipelines, case packet builder
├── retrieval/               # metadata filters, BM25, vector, hybrid retrieval
├── scripts/                 # build scripts
├── batch_extract.py          # batch PDF → metadata/parsed artifacts
├── tests/
├── ADR.md                   # architecture decision record
└── README.md
```

**Note:** `OPENAI_API_KEY` is required even when using Groq for generation, because retrieval currently uses OpenAI embeddings.

## Artifacts: `data/` vs `storage/`

The Streamlit app and indexing scripts use `config/settings.py` paths (under `data/`):
- `data/metadata/` (metadata JSON)
- `data/parsed_text/` (parsed `.txt`)

Some extraction helpers may write to `storage/` by default. If you want to generate artifacts directly into the runtime locations for indexing/UI, run:

```bash
python batch_extract.py --provider openai --openai-model gpt-5-mini --out-dir data/metadata --parsed-dir data/parsed_text
```

## Why `OPENAI_API_KEY` is required

`OPENAI_API_KEY` is required because the current retrieval pipeline uses **OpenAI embeddings** for dense search.

This key is used in two places:

1. **Index building**
   - when running `python scripts/build_indices.py`
   - the system embeds the corpus documents / sections and stores the vector index

2. **Runtime retrieval**
   - when a user asks a question in the app
   - the system embeds the query and compares it against the stored document vectors

### Important
Even if you choose **Groq** for final answer generation, retrieval still depends on OpenAI embeddings.

That means:

- **OpenAI key is mandatory**
- **Groq key is optional**
- **Groq key alone is not enough**

### Current provider split

- **Retrieval / embeddings** → OpenAI
- **Final answer generation** → OpenAI or Groq

### Example

If you want to use:
- **OpenAI generation** → provide `OPENAI_API_KEY`
- **Groq generation** → provide both `OPENAI_API_KEY` and `GROQ_API_KEY`

### Future improvement
A future version can replace OpenAI embeddings with a local/open embedding model so that Groq-only usage becomes possible.
