# Evaluation Analysis Report

## Overview
This report summarizes the automated evaluation framework and results for the legal precedent research agent built over the 56-document corpus of Indian court judgments.

The assignment required at least one automated evaluation for each of the following dimensions:
1. Precision
2. Recall
3. Reasoning Quality
4. Adverse Identification

To satisfy that requirement, I built a **lean benchmark of 12 tasks** covering both corpus Q&A and deep precedent-research prompts. Each benchmark item includes gold labels for:
- relevant cases
- must-find cases
- supporting cases
- adverse cases
- mixed cases
- expected principles and risk themes

The evaluation runner stores both raw outputs and summarized results so the system can be inspected end-to-end rather than only through aggregate metrics.

## Benchmark Design
The benchmark intentionally mixes:
- insurer liability / invalid licence issues
- pay-and-recover
- endorsement defects
- gratuitous / unauthorized passengers
- contributory negligence
- compensation methodology
- future prospects / multiplier
- child death compensation
- disability / serious injury compensation
- vehicle-use / tractor / trolley policy-breach issues
- one factual corpus-Q&A task

This design was chosen because the assignment explicitly says the system will be tested on both general document questions and deeper research tasks, not only on the provided client brief.

## Evaluation Methodology

### 1. Precision
Measured as document-level precision over the final cited precedents:

- **Precision** = relevant predicted cases / predicted cases

This measures whether the system over-retrieves and presents irrelevant authorities as meaningful precedents.

### 2. Recall
Measured as document-level recall against the benchmark gold set:

- **Recall** = relevant predicted cases found / total gold relevant cases
- **Must-find recall** = must-find cases found / total must-find cases

Must-find recall was included because some authorities are much more central than others.

### 3. Reasoning Quality
Measured by a rubric-based automated judge scoring:
- factual alignment
- legal principle accuracy
- applicability reasoning
- grounding
- nuance

Scores were normalized to a 0–1 range.

### 4. Adverse Identification
Measured through:
- adverse recall
- adverse presence rate
- an automated judge scoring whether surfaced adverse cases were explained honestly and whether the answer acknowledged legal risk

This dimension was separated deliberately because legal research systems are dangerous if they only surface favorable authorities.

## Aggregate Results
Across 12 benchmark queries, the system produced the following aggregate metrics:

- **Precision avg:** 0.497
- **Recall avg:** 0.514
- **Must-find recall avg:** 0.653
- **Supporting precision avg:** 0.099
- **Supporting recall avg:** 0.133
- **Adverse recall avg:** 0.000
- **Reasoning quality avg:** 0.808
- **Adverse honesty avg:** 0.844
- **Adverse presence rate:** 0.000

Cost during this run:
- **Total eval cost:** $0.178219
- **Agent cost:** $0.119168
- **Judge cost:** $0.059051

## What the results show

### 1. The system reasons better than it retrieves
The strongest result is the **reasoning quality score of 0.808**, which is materially higher than both precision and recall. This suggests that once the system retrieves relevant material, it can often explain the precedent reasonably well. 

This is an encouraging result because it means the synthesis layer is not the primary bottleneck.

### 2. Retrieval quality is only moderate
Precision and recall both sit around 0.5, which means:
- some relevant authorities are being found
- many are being missed
- some irrelevant or weakly relevant authorities are still being included

This is consistent with an early hybrid-RAG system that has a reasonable retrieval substrate but lacks a stronger legal reranker.

### 3. Adverse precedent identification is the major failure mode
The clearest weakness is:
- **Adverse recall = 0.000**
- **Adverse presence rate = 0.000** 

This means that the system is not reliably surfacing adverse authorities as a distinct category, even when benchmark tasks explicitly require it.

That is also visible in per-query results. For example, in the first benchmark task the system produced a long precedent memo and discussed mixed authorities in prose, but the structured output still recorded:
- `predicted_supporting_cases`: multiple cases
- `predicted_adverse_cases`: empty list

So the system is currently better at producing a coherent legal memorandum than at structurally representing adversarial precedent balance.

## Why these results happened

### A. Retrieval is topic-aware but not stance-aware
The system can retrieve topic-relevant cases, but it does not yet reliably distinguish between:
- supportive
- adverse
- mixed
- irrelevant

This especially hurts:
- supporting precision / recall
- adverse recall

The benchmark shows this clearly through very low support metrics and zero adverse recall. 

### B. Final synthesis is stronger than case-role classification
The final answers often sound legally plausible, but the structured precedent buckets are weak. This indicates that the synthesis model is compensating for an upstream classification gap instead of receiving well-labeled case packets.

### C. The current reranking layer is not strong enough
Hybrid retrieval alone is not enough for legal precedent search. A retrieved case may be topically relevant while still being the wrong polarity or only weakly analogous. The current system needs a more explicit legal reranking/classification stage.

### D. Development model choice was cost-constrained
The evaluation run used **gpt-5-mini** for reasoning. This was a practical cost decision during development rather than a claim that this is the optimal reasoning model. Smaller models are more likely to:
- collapse mixed cases into supportive ones
- miss nuanced distinctions
- fail to surface contrary authority unless explicitly forced

Given the current results, I expect stronger reasoning models such as **o3**, **o4-mini**, or **gpt-5.2** to improve:
- stance classification
- adverse-case surfacing
- nuance in mixed-authority analysis
- strategy synthesis

However, OpenAI credit constraints made it impractical to use more expensive frontier models throughout iterative testing.

## Per-query observations
A few benchmark tasks illustrate the current system profile well:

### eval_001
- strong overall memo
- high precision for overall relevance (0.800)
- reasonable recall (0.571)
- **adverse recall still 0.000** 

Interpretation: topic retrieval works reasonably well, but adverse precedent extraction is failing.

### eval_004
- hazardous-goods endorsement task
- must-find recall = 1.000
- reasoning = 1.000 

Interpretation: when the issue cluster is narrow and well represented in metadata, the pipeline performs much better.

### eval_009 and eval_010
- recall = 1.000 on child-death compensation and serious injury compensation tasks
- support metrics still weak 

Interpretation: the system can retrieve topic-relevant compensation cases but is not sharply classifying their legal role.

### eval_011
- corpus-Q&A on commercial vehicles and contested insurer liability
- precision = 1.000
- recall = 0.211 

Interpretation: metadata filtering is strong when it hits, but not broad enough yet for exhaustive recall.

## What I would fix first

### 1. Add an explicit case-role classifier before final synthesis
Every shortlisted case should be classified into one of:
- supportive
- adverse
- mixed
- irrelevant

This should be done *before* writing the final answer.

This is the highest-value improvement because it directly targets the biggest benchmark weakness.

### 2. Add a stronger reranker
The reranker should score not just general relevance, but:
- factual similarity
- doctrinal similarity
- claimant-side support value
- insurer-side adverse value
- mixed / distinguishable status

### 3. Add an “adverse search” pass
For deep research tasks, the agent should deliberately run a second retrieval pass optimized for contrary authority rather than assuming the main retrieval pass will surface it naturally.

### 4. Separate retrieval recall from final-answer recall more explicitly
The next evaluation iteration should log:
- retrieval recall@10 / @20
- reranked recall
- final answer recall

That will make it easier to diagnose whether misses happen in retrieval, reranking, or synthesis.

### 5. Strengthen taxonomy around licence defects and policy breaches
The corpus includes legally important distinctions such as:
- no licence
- fake licence
- expired licence
- wrong class of licence
- missing hazardous-goods endorsement
- unauthorized passenger / vehicle-use breach

More granular metadata would improve both retrieval and downstream reasoning.

## What I would improve with more time
If I had another week, the evaluation framework would be extended by:
- adding a larger benchmark beyond the lean 12-task set
- running A/B comparisons across different generation models
- evaluating retrieval and synthesis separately at more stages
- adding a calibrated adverse-precedent benchmark focused only on contrary authority
- introducing manual review on a small subset of borderline cases to refine gold labels

## Model / cost note
This evaluation should be interpreted in light of the model choice used during development. Because cost was a real constraint, I used **gpt-5-mini** for reasoning rather than more expensive frontier reasoning models. I believe the reported metrics, especially:
- adverse identification
- support/adverse case-role labeling
- nuanced mixed-case treatment
would improve with stronger models such as **o3**, **o4-mini**, or **gpt-5.2**.

That said, the purpose of this evaluation is not to optimize for the best possible paid model; it is to reveal whether the **architecture and retrieval pipeline** are directionally sound. On that question, the benchmark is informative:
- reasoning is already promising
- retrieval is workable but incomplete
- adverse surfacing is the most urgent architectural gap

## Final assessment
The current system is a credible first version, but not yet a balanced legal research assistant.

Its main strength is:
- coherent reasoning over retrieved authorities

Its main weaknesses are:
- only moderate retrieval coverage
- weak support/adverse role discrimination
- failure to explicitly surface adverse precedents

This is useful and actionable because the benchmark does not merely show that the system underperforms in some areas; it isolates **where** it underperforms and therefore what should be improved next. 
