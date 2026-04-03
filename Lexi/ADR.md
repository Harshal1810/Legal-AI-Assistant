# Architecture Decision Record (ADR)

## Title
Legal Precedent Research Agent for Indian Motor Accident Judgments

## Context
This project implements a legal research agent over a corpus of 56 Indian court judgments. The assignment requires a hosted application that can answer both general corpus questions and deeper precedent-research tasks, while exposing intermediate reasoning steps. The system must not be a hard-coded pipeline for one case brief; it should handle flexible prompts such as factual corpus queries, precedent research, and strategy-style legal analysis. The submission must also include an ADR and an evaluation framework covering precision, recall, reasoning quality, and adverse precedent identification. 

The practical constraints were:
- Python-based implementation. 
- Small corpus size (56 judgments), which favors lightweight local indexing over heavy infrastructure.
- Need for transparent retrieval and reasoning traces in the UI. 
- Cost sensitivity during development and evaluation.

## Decision Summary
I chose a **metadata-aware hybrid RAG architecture** with two execution modes:
1. **Fast Q&A mode** for direct corpus questions.
2. **Deep research mode** for precedent analysis, including supporting, adverse, and mixed authorities.

The system uses:
- PDF-derived parsed text plus extracted metadata JSON as the corpus substrate.
- BM25/lexical retrieval over aggregated searchable text.
- Dense retrieval over compact case-level representations.
- A hybrid ranker that combines lexical, semantic, and metadata-filter signals.
- A reasoning layer that builds **case packets** before final generation.
- A Streamlit UI that surfaces router decisions, retrieval traces, ranking outputs, and final synthesis.

## Why this architecture

### 1. Retrieval-first design fits the assignment better than a heavy multi-agent design
The core problem is not open-ended autonomous planning; it is faithful precedent retrieval and balanced legal synthesis over a bounded judgment corpus. For that reason, I chose a compact retrieval-and-reasoning architecture instead of a heavier multi-agent orchestration framework. This keeps the system inspectable and easier to defend in interview discussion.

### 2. Metadata is a first-class retrieval signal
The extraction pipeline produced metadata fields such as:
- vehicle type
- licence issue flags
- insurer-liability contest flags
- pay-and-recover
- compensation details
- ratio / legal principles / sections cited / cases cited

These fields materially improve filtering, ranking, and answer synthesis. Rather than treating metadata as a side artifact, I used it as a first-class retrieval layer alongside text.

### 3. Retrieval unit and generation unit are intentionally different
For legal research, the best retrieval unit is often a chunk or section. The best generation unit is a **case packet**: structured metadata + selected evidence spans. This reduces the risk that final synthesis misses a holding, caveat, or remedy detail that appears in a different part of the judgment.

### 4. Separate Q&A and deep research modes improve flexibility without brittle hard-coding
The system routes simple corpus questions to a lightweight answer path, while deeper precedent-research prompts trigger richer retrieval and synthesis. This directly addresses the assignment’s requirement for flexibility across both general and deep legal research prompts.

## System Architecture

### Corpus inputs
- **Parsed text files**: cleaned full text extracted from PDFs.
- **Metadata JSON**: structured representation of each judgment.

### Retrieval layers
1. **Metadata filters**
   - Used for exact narrowing when the query clearly references issue flags such as commercial vehicle, insurer contest, invalid licence, death, injury, etc.
2. **BM25 / lexical retrieval**
   - Used for exact legal terms, statutory references, and citation-heavy queries.
3. **Dense retrieval**
   - Used for fact-pattern similarity and concept-level matching.
4. **Hybrid aggregation**
   - Combines lexical and semantic hits, with metadata-based boosts.

### Reasoning layers
1. **Router**
   - Chooses between Q&A and deep research.
2. **Case packet builder**
   - Packages each shortlisted judgment as metadata + evidence snippets.
3. **Synthesis**
   - Produces either:
     - a direct answer, or
     - a precedent research memorandum with supporting, adverse, and mixed authorities plus strategy.

### UI design
The Streamlit application exposes:
- route decision
- metadata filtering trace
- BM25 hits
- vector hits
- hybrid ranking
- selected case packets
- final answer

This was intentional because the brief specifically asks for visible intermediate reasoning steps rather than a black-box final answer. 
## Retrieval Strategy
I chose a **hybrid retrieval** approach because legal search fails in different ways:
- Dense search may miss exact doctrinal hooks or section-level wording.
- Lexical search may miss semantically similar fact patterns.

The hybrid approach is especially useful for this corpus because many judgments turn on subtle distinctions, such as:
- no licence vs expired licence vs missing endorsement
- insurer exoneration vs pay-and-recover
- unauthorized passenger vs third-party claimant
- technical breach vs fundamental/causative breach

## Chunking Approach
I did not rely purely on fixed-size chunks. Instead, I used a mixed strategy:
- judgment-level metadata summaries for coarse retrieval
- section-level or reasoning-oriented text units for evidence grounding
- final answer generation over case packets rather than raw chunk dumps

This design is more faithful to legal judgments, where facts, ratio, and final order often appear in different parts of the document.

## Model Choices
### Generation
During evaluation, generation and reasoning were run with **gpt-5-mini**. This was a cost-conscious development choice rather than a claim that it is the best possible model. The eval results show that even with this smaller model, reasoning quality was relatively strong compared with retrieval quality.

A stronger frontier reasoning model such as **o3**, **o4-mini**, or **gpt-5.2** would likely improve:
- case-role classification
- adverse-case surfacing
- strategy synthesis
- nuance in mixed precedents

However, the development constraint was cost. During testing, OpenAI credit limits became a practical bottleneck, so the system design was adjusted to keep retrieval cheap and generation modular.

### Embeddings
The intended long-term architecture keeps generation provider and embedding provider decoupled. This makes it possible to:
- use Groq or OpenAI for generation depending on the user’s supplied key
- use a cheaper or local embedding backend for retrieval

This is the correct separation for a cost-sensitive deployment.

## Tradeoffs Made

### Tradeoff 1: Simplicity over maximal orchestration
I chose a smaller, inspectable architecture rather than a more elaborate multi-agent orchestration system. This reduces engineering complexity and makes the system easier to explain.

### Tradeoff 2: Metadata-aware retrieval over raw full-document prompting
Instead of stuffing full judgments into the generation context, I relied on structured metadata and selected evidence spans. This improves controllability and cost efficiency, at the cost of requiring more retrieval engineering.

### Tradeoff 3: Small-scale local artifacts over cloud vector infrastructure
Because the corpus is only 56 documents, local storage and index files are operationally simpler than a managed vector database. This is sufficient for the take-home scope.

### Tradeoff 4: Cost-conscious model selection during evaluation
Using gpt-5-mini kept evaluation affordable, but likely depressed the final metrics relative to what a stronger reasoning model could achieve.

## How the agent decides between simple Q&A and deep research
The router uses query intent cues rather than case-specific hard-coded branches.

### Q&A mode
Triggered by prompts such as:
- “Which judgments involve commercial vehicles?”
- “Which cases mention contributory negligence?”
- “What did the court hold about future prospects?”

### Deep research mode
Triggered by prompts asking for:
- supporting precedents
- adverse precedents
- mixed / distinguishable authorities
- litigation strategy
- risk analysis
- compensation range / realistic outcome

This is deliberately general so the system can handle more than the provided Lakshmi Devi case brief. 

## If the corpus were 5,000 documents instead of 56
I would make the following changes:

1. **Persistent vector infrastructure**
   - Move dense retrieval to a real vector database or persistent ANN index.
2. **More formal document normalization**
   - Stronger metadata normalization and taxonomy enforcement.
3. **Incremental indexing pipeline**
   - Avoid rebuilding from scratch.
4. **Two-stage retrieval at scale**
   - fast coarse retrieval → cross-encoder reranking.
5. **Caching and query analytics**
   - cache frequent query embeddings and popular case packets.
6. **Corpus governance**
   - versioned metadata artifacts, better benchmark coverage, and drift checks.

## What I would change with another week
The highest-value improvements would be:

1. **Explicit case-role classifier**
   - classify each retrieved case as supportive, adverse, mixed, or irrelevant before synthesis.
2. **Better reranker**
   - add a stronger learned or prompt-based reranking stage tuned for legal relevance and stance.
3. **Adverse-first retrieval pass**
   - deliberately search for contrary authority instead of assuming one retrieval pass is enough.
4. **Improved evaluation instrumentation**
   - separate retrieval recall@k from final-answer recall more explicitly.
5. **Provider-independent local embeddings**
   - fully remove retrieval dependence on paid APIs.
6. **More benchmark coverage**
   - expand beyond the lean benchmark and add harder mixed-authority tasks.

## Evaluation-informed reflection
The current evaluation results show a clear pattern:
- reasoning quality is relatively strong
- retrieval is moderate
- adverse precedent identification is the major weakness

This suggests the architecture is directionally correct but needs a stronger classification/reranking layer between retrieval and synthesis. 

## Final Justification
The chosen design is appropriate for the assignment because it is:
- flexible across Q&A and deep research tasks
- faithful to the bounded legal corpus
- transparent in its intermediate reasoning steps
- practical to host and explain
- extensible for larger corpora and stronger models

Most importantly, it makes the system’s current strengths and weaknesses visible instead of hiding them behind a polished but opaque final answer.
