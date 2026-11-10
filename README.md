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
- [ ] **Phase 2: Ingestion & Multi-Config Retrieval Engine**
- [ ] **Phase 3: 4-Step Resilient Pipeline with Tool Call & Branch/Loop**
- [ ] **Phase 4: Concurrency, Rate Limiting & Queueing Policies**
- [ ] **Phase 5: Load Testing & Experiment Execution**
- [ ] **Phase 6: Data Synthesis & `results.csv` Verification**
- [ ] **Phase 7: 2-Page Executive Technical Report**

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
