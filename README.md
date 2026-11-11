# The Relay

A resilient, multi-step engineering Q&A assistant built over foundational research papers, designed to withstand strict rate limits, bursting traffic, and graceful degradation under load.

## Overview

The assistant answers engineers' questions strictly grounded in three research papers:
1. *"Attention Is All You Need"* (arXiv:1706.03762v7)
2. *"Moshi: a speech-text foundation model for real-time dialogue"* (arXiv:2410.00037v1)
3. *"Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"* (arXiv:2507.07955v2)

It runs across two model backends:
- **Real Groq API**: Live inference on Groq's free tier with strict rate limits (RPM, TPM, RPD, in-flight concurrency).
- **Stand-in Model Service**: A local, fully controllable surrogate calibrated against real Groq measurements, simulating streaming tokens, queueing delays, load-dependent degradation, 429 rate limit bodies, transient 503s, and daily quota exhaustion.

## Project Architecture

```
The Relay/
├── the_relay/
│   ├── __init__.py
│   ├── config.py             # Configuration and environment management
│   ├── models.py             # Pydantic schemas for completions, tokens, limits, traces
│   ├── profiler/             # Phase 1: Real Groq profiling and limit discovery
│   │   ├── __init__.py
│   │   └── groq_profiler.py
│   ├── standin/              # Phase 1: Calibrated stand-in model service
│   │   ├── __init__.py
│   │   ├── state.py
│   │   └── service.py
│   ├── retrieval/            # Phase 2: Multi-configuration RAG engine
│   └── pipeline/             # Phase 3 & 4: 4-step pipeline, retry & queue policies
├── data/
│   ├── papers/               # Downloaded arXiv paper versions
│   └── test_suite.json       # 15 curated benchmark questions + traps
├── tests/
│   ├── test_profiler.py
│   └── test_standin.py
├── results/
│   ├── results.csv           # Raw benchmark execution data
│   └── tables/               # Formatted markdown tables
└── requirements.txt
```

## Implementation Roadmap

- [x] **Project Foundation:** Scaffolding, repository setup, configuration schemas.
- [x] **Phase 1: Real Profiling & Stand-in Service**
  - [x] Task 1.1: Empirical Groq profiler (latency spread, headers, recovery times, Table 1).
  - [x] Task 1.2: Stand-in model service (token streaming, load slowdown, failure injection, calibration).
- [x] **Phase 2: Ingestion & Multi-Config Retrieval Engine**
  - [x] Task 2.1: Fetch and parse exact arXiv versions (1706.03762v7, 2410.00037v1, 2507.07955v2).
  - [x] Task 2.2: Curate 15-question benchmark suite and analyze naive retriever pitfalls.
  - [x] Task 2.3: Multi-configuration RAG retrieval engine with dynamic chunking and re-ranking.
  - [x] Task 2.4: Evaluate retrieval quality vs cost and export Table 3.
- [x] **Phase 3: 4-Step Resilient Pipeline with Tool Call & Branch/Loop**
  - [x] Task 3.1: Pipeline data models and execution telemetry (`the_relay/pipeline/models.py`).
  - [x] Task 3.2: 4-step pipeline engine with tool invocation and branch/loop logic (`the_relay/pipeline/core.py`).
  - [x] Task 3.3: Self-verification citation guardrails and unanswerable refusal handling.
  - [x] Task 3.4: Granular latency dissection (waiting vs working vs retrying).
- [x] **Phase 4: Concurrency, Rate Limiting & Queueing Policies**
  - [x] Task 4.1: Adaptive rate limiter with sliding-window RPM/TPM and in-flight concurrency gates.
  - [x] Task 4.2: Retry manager with exponential backoff, AWS Full Jitter, and global retry budget.
  - [x] Task 4.3: Resilient dispatcher with queue backpressure shedding and daily budget exhaustion handling.
  - [x] Task 4.4: Thundering herd and concurrency stress test validation.
- [x] **Phase 5: Load Testing & Experiment Execution**
  - [x] Task 5.1: Benchmark scenario harness (`the_relay/benchmarks/scenarios.py`).
  - [x] Task 5.2: Execute all 5 mandatory load shapes (Steady, Burst, Sustained Overload, Limit Halved, Budget Exhausted).
  - [x] Task 5.3: Capture raw results into `results/results.csv` and export Table 2a and Table 2b.
- [x] **Phase 6: Data Synthesis & `results.csv` Verification**
  - [x] Task 6.1: Automated recomputation and validation against results.csv (`scripts/verify_results.py`).
  - [x] Task 6.2: Verified 100% mathematical consistency across all tables and schemas.
- [x] **Phase 7: 2-Page Executive Technical Report**
  - [x] Task 7.1: Author technical report strictly adhering to required headings (`REPORT.md`).
  - [x] Task 7.2: Vaani product evaluation and architectural feedback.
  - [x] Task 7.3: Complete uncleaned conversation history export (`data/conversation_history.md`).

## Phase 1 Deliverables Summary

### Task 1.1: Empirical Groq Profiler
- **Module:** `the_relay/profiler/groq_profiler.py`
- **Output:** Populates empirical rate-limits and header schemas for **Table 1** (RPM: 30, TPM: 6000, RPD: 1000, TPD: 100,000, Concurrency cap: 1-3 in flight).
- **Features:** SSE streaming parser, accurate TTFT measurement, rate-limit header tracking (`x-ratelimit-*`, `retry-after`), and Table 1 Markdown export.

### Task 1.2: Calibrated Stand-in Model Service
- **Module:** `the_relay/standin/service.py` & `the_relay/standin/state.py`
- **Calibration:**
  - Base TTFT: $\mu = 220\text{ ms}, \sigma = 40\text{ ms}$.
  - Streaming generation: ~70 tokens/sec (14 ms inter-token delay).
  - Concurrency slowdown: TTFT penalization factor $1 + 0.45 \times (C - 1)^{1.3}$ for $C$ active in-flight requests.
  - Rate limiting: Sliding 60-second window enforcing 30 RPM / 6,000 TPM with HTTP 429 payload and `retry-after`.
  - Quota exhaustion: Immediate short-circuit simulation when daily token budget hits zero (3:00 PM scenario).
  - Failure injection: Configurable 503 transient server errors and slow successes (>4,000 ms TTFT).
- **Client:** `the_relay/client.py` provides unified dispatch across Real Groq and the Stand-in model.

## Phase 2 Deliverables Summary

### Task 2.1: arXiv Corpus Ingestion
- **Module:** `the_relay/retrieval/ingest.py`
- **Exact Papers Ingested:**
  - *Attention Is All You Need* (`1706.03762v7`)
  - *Moshi: a speech-text foundation model for real-time dialogue* (`2410.00037v1`)
  - *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* (`2507.07955v2`)
- **Storage:** Structured page-level JSON cache under `data/papers/`.

### Task 2.2: 15-Question Benchmark Suite & Naive Retriever Vulnerability Analysis
- **Dataset:** `data/test_suite.json` (8 grounded technical questions, 3 adversarial traps, 4 unanswerable questions requiring explicit refusal).
- **Three Ways This Library Misleads a Naive Retriever:**
  1. *Modality Collision on "Tokens" and "Codebooks":* Conflates text BPE tokens (Vaswani) with discrete audio RVQ codebooks (Moshi Mimi).
  2. *Semantic Ambiguity on "Chunking":* Conflates fixed 80ms streaming audio buffer chunks (Moshi) with learned boundary predictor gating (Dynamic Chunking).
  3. *Attention Multiplexing Ambiguity:* Conflates standard encoder-decoder cross-attention (Vaswani) with real-time full-duplex conversational dual-stream modeling (Moshi).

### Task 2.3 & 2.4: Multi-Configuration RAG Retrieval Engine & Table 3
- **Engine Module:** `the_relay/retrieval/engine.py` (parameterizable piece sizes, overlap, TF-IDF + keyword scoring, positional re-ranking, and degraded mode when quota is near zero).
- **Evaluator Module:** `the_relay/retrieval/evaluator.py` & `results/table3.md`
- **Empirical Table 3 Results:**

| Setting (piece size, pieces fetched, re-rank yes or no) | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|
| 200 words, 3 pieces, re-rank no | 15 | 66.7% | 6.7% | 26.7% | 28 | 190 / 222 | 554 / 696 |
| 500 words, 5 pieces, re-rank yes | 15 | 73.3% | 0.0% | 26.7% | 29 | 181 / 202 | 498 / 683 |
| 1000 words, 2 pieces, re-rank no | 15 | 73.3% | 0.0% | 26.7% | 29 | 200 / 222 | 518 / 703 |

**Selected Setting:** Setting 2 (500 words, 5 pieces, re-rank yes) is chosen as primary because it eliminates retrieval truncation errors (0.0% wrong) with high citation density at negligible latency delta (+17ms TTFT vs Setting 1). When rate limits approach (<20% quota remaining), the engine switches dynamically to degraded mode (clamping to 2 pieces, 0 re-rank overhead).

## Phase 3 Deliverables Summary

### 4-Step Resilient Pipeline Architecture
- **Module:** `the_relay/pipeline/core.py` & `the_relay/pipeline/models.py`
- **Execution Flow:**
  1. **Step 1: Query Deconstruction & Intent Classifier:** Parses query intent, detects topic domains across the 3 papers, checks for known out-of-domain entities (ViT, RLHF, GPU wattages, Mamba), and prepares optimized search arguments.
  2. **Step 2 (Tool Call): Passage Retrieval (`retrieve_passages`):** Formally invokes the retrieval tool with query arguments and captures candidate passages with author, year, and page-level citation provenance.
  3. **Step 3 (Branch / Loop): Sufficiency & Ambiguity Evaluation:**
     - *Loop Trigger:* Evaluates cross-paper terminology ambiguities (e.g., generic "chunking" or "codebook" without paper scope). Automatically refines the query with contextual disambiguation terms and re-executes Step 2 (bounded by `max_loops=2`).
     - *Branch Trigger:* If textual evidence is missing or entity is out-of-domain, branches to deterministic refusal (`PipelineBranch.REFUSAL`). Otherwise, advances to `PipelineBranch.GROUNDED_ANSWER`.
  4. **Step 4: Grounded Drafting & Self-Verification Guardrail:**
     - For refusals: Outputs strict refusal without hallucination (*"The provided papers do not say anything about this topic..."*).
     - For answers: Synthesizes grounded explanation with parenthetical citations `(Author et al. (Year), Page X)`. Automatically checks that claims are accompanied by citations matching retrieved passages.
- **Latency Dissection Telemetry:** Records fine-grained timing for every request:
  - `waiting_ms` (queueing delay)
  - `working_ms` (pipeline computation & model inference)
  - `retrying_ms` (backoff delay)
  - `first_token_ms` (TTFT relative to arrival)
  - `total_ms` (end-to-end turnaround time)
  - `delay_before_first_token` boolean flag.

## Phase 4 Deliverables Summary

### Resilient Concurrency & Queueing Controller
- **Modules:** `the_relay/resilience/limiter.py`, `the_relay/resilience/retry.py`, `the_relay/resilience/dispatcher.py`
- **Key Policies:**
  1. **Sliding-Window Token Bucket & Concurrency Gate:**
     - Enforces 30 RPM and 6,000 TPM with sliding 60-second window tracking.
     - Enforces strict in-flight concurrency cap ($C \le 3$) matching real Groq hardware saturation thresholds.
  2. **Queueing & Backpressure Shedding:**
     - FIFO request queue with `max_queue_depth=25`. When incoming bursts saturate the queue, surplus requests are shed immediately with HTTP 429 rather than hanging indefinitely.
     - `client_timeout_s=12.0s`: Requests exceeding client patience are timed out cleanly (`status=timeout`).
  3. **Daily Budget Exhaustion Policy (3:00 PM Scenario):**
     - When daily token quota hits zero, the dispatcher short-circuits instantly ($<5\text{ ms}$) with zero downstream LLM invocations:
       > *"The daily computational research quota has been exhausted. All available model credits have been utilized. Service will resume at 00:00 UTC."*
  4. **Exponential Backoff with Full Jitter & Global Retry Budget:**
     - Formula: $T_{\text{backoff}} = \min(T_{\text{max}}, T_{\text{base}} \times 2^{\text{attempt}}) \pm \text{jitter}$ with AWS Full Jitter.
     - Strict per-request retry ceiling: 2 attempts maximum.
     - Systemic retry budget: Retries cannot exceed 20% of total system bandwidth, preventing cascading brownouts.
     - Non-retryable guard: Never retries on 400 Bad Request, Daily Quota Exhaustion, or Client Timeouts.
  5. **Thundering Herd Mitigation:** Jitter breaks lock-step retry synchronization during burst failures.

## Phase 5 Deliverables Summary

### Benchmark Execution Across 5 Mandatory Scenarios
- **Modules:** `the_relay/benchmarks/scenarios.py`, `the_relay/benchmarks/runner.py`, `scripts/run_benchmarks.py`
- **Raw Data:** `results/results.csv` (110 rows across all 5 scenarios with required schemas).
- **Table 2a: Latency:**

| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |
|---|---|---|---|---|---|---|
| steady | stand-in | 15 | 34.6 | 26.0 | 20 / 129 / 157 | 9 / 1584 / 2365 |
| burst | stand-in | 20 | 1.0 | 1200.0 | 251 / 271 / 271 | 30 / 464 / 710 |
| sustained_overload | stand-in | 35 | 15.5 | 135.5 | 20 / 20 / 20 | 7 / 12016 / 12165 |
| per_minute_limit_halved | stand-in | 25 | 29.1 | 51.6 | 0 / 18 / 20 | 12013 / 12019 / 12230 |
| daily_budget_exhausted | stand-in | 15 | 0.1 | 9000.0 | 0 / 0 / 0 | 1 / 1 / 1 |

- **Table 2b: Outcomes:**

| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |
|---|---|---|---|---|---|---|---|
| steady | 33.3% | 66.7% | 0.0% | 0.33 | 1 | 30 | $0.00 |
| burst | 20.0% | 80.0% | 0.0% | 0.1 | 10 | 29 | $0.00 |
| sustained_overload | 11.4% | 60.0% | 28.6% | 0.11 | 10 | 29 | $0.00 |
| per_minute_limit_halved | 4.0% | 40.0% | 56.0% | 0.12 | 8 | 29 | $0.00 |
| daily_budget_exhausted | 0.0% | 100.0% | 0.0% | 0.0 | 1 | 21 | $0.00 |

### Critical Empirical Finding ("A number we could not have guessed without running it")
While the API advertises **30 Requests Per Minute**, retrieving 5 passages consumes **~3,000 tokens of input context per query**. Because the Free Tier limits tokens to **6,000 TPM**, the bottleneck is NOT requests, but tokens: the system is choked by TPM after just **2 queries in a 60-second window**, forcing the remaining requests in the steady stream to queue or be rate-limited unless degraded retrieval (2 chunks) is applied.

## Phase 6 & 7 Deliverables Summary

### Data Synthesis & Technical Report
- **Technical Report:** [`REPORT.md`](file:///c:/Users/harsh/Desktop/The%20Relay/REPORT.md) — 2-page report strictly adhering to all required headings, including:
  - Architecture design & core decision (TPM budget protection).
  - Stand-in calibration against Groq measurements.
  - Complete tables: Table 1 (Limits), Table 2a (Latency), Table 2b (Outcomes), Table 3 (Retrieval).
  - Latency dissection (waiting vs working vs retrying).
  - Limit hit policies (RPM vs RPD, degradation, shedding).
  - Retrieval quality vs cost analysis.
  - 3 ways this library misleads a naive retriever.
  - Retry policy and the thundering herd case study.
  - 3:00 PM daily budget exhaustion message.
  - Things unsure about, left out, and learned.
  - Feedback on Vaani.
  - Full analysis of assistant interactions (prompts that mattered, AI mistakes caught, overruled decisions, personal distinct signature).
- **Verification Harness:** [`scripts/verify_results.py`](file:///c:/Users/harsh/Desktop/The%20Relay/scripts/verify_results.py) recomputes all table figures directly from [`results/results.csv`](file:///c:/Users/harsh/Desktop/The%20Relay/results/results.csv) to guarantee 100% mathematical consistency.
- **Conversation Logs:** [`data/conversation_history.md`](file:///c:/Users/harsh/Desktop/The%20Relay/data/conversation_history.md) preserves the raw, uncleaned prompt history across all project phases.
