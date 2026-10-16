# System Architecture & Component Specification

This document details the end-to-end architecture and runtime execution models of the Khmer Legal Retrieval and Question-Answering system.

---

## 1. End-to-End Pipeline Data Flow

```mermaid
flowchart TD
    subgraph S1["1. Data Ingestion & Font Normalization"]
        PDF["Official Khmer Legal Codes<br/>(Civil Code 2007 • Criminal Code 2009)"]
        Converter["Limon Converter & Hierarchical Chunker<br/>(Unicode Mapping • Syllable Reordering)"]
        Corpus[("Canonical Statutory Corpus<br/>(1,976 Articles • data/04_chunks)")]
        PDF --> Converter --> Corpus
    end

    subgraph S2["2. Deterministic Partitioning (Seed 42)"]
        Guard["Leakage Quarantine Guard<br/>(Strictly Excludes 261 Cited Articles)"]
        TrainVal["Training & Validation Splits<br/>(1,176 Train Pairs • 147 Val Pairs)"]
        Benchmarks["Dual Test Benchmarks<br/>(200 T-Q Questions • 148 T-T Titles)"]
        Corpus -->|Exclude 261 Cited Articles| Guard
        Guard --> TrainVal
        Guard --> Benchmarks
    end

    subgraph S3["3. PyTorch Deep Learning Retrievers (InfoNCE Loss)"]
        A1["A1: BiLSTM Dual Encoder<br/>(Scratch • 1.97M Params • 512-dim)"]
        A2["A2: XLM-R Linear Probe<br/>(Frozen • 0.59M Params • 768-dim)"]
        A3["A3: XLM-R Full Fine-Tune<br/>(Unfrozen • 278M Params • Best Model)"]
        A4["A4: PrahokBART Dual Encoder<br/>(Pretrained • 35.8M Params • 512-dim)"]
        TrainVal -->|1,176 Pairs| A1
        TrainVal --> A2
        TrainVal --> A3
        TrainVal --> A4
    end

    subgraph S4["4. Shared Evaluation & Metrics"]
        Harness["Shared Evaluation Harness<br/>(Recall@k • MRR@10 • 10k Bootstrap CIs)"]
        Results["Consolidated Leaderboard<br/>(summary.csv • Model Selection)"]
        A1 --> Harness
        A2 --> Harness
        A3 --> Harness
        A4 --> Harness
        Benchmarks -->|Evaluate T-Q & T-T| Harness
        Harness --> Results
    end

    subgraph S5["5. Citation-Grounded Assistant Prototype"]
        Retriever["Hybrid Retriever & RRF Fusion<br/>(A3 Dense Embeddings + BM25 Lexical)"]
        LLM["DeepSeek Flash (deepseek-chat)<br/>(Statutory-Conditioned Generation)"]
        Verifier["Citation Verification Engine<br/>(Validates មាត្រា N Citations in Context)"]
        Answer["Grounded Legal Response<br/>(Streamlit Web UI & FastAPI Service)"]
        Retriever -->|Top-k Retrieved Context| LLM
        LLM --> Verifier
        Verifier --> Answer
    end

    Results -->|Deploy Best Retriever A3| Retriever
```

---

## 2. Interactive RAG Prototype Execution Path

The project provides two execution paths designed according to **Clean Architecture** principles:

### Path A: Interactive Streamlit UI (In-Process Defense Mode)
The interactive demo (`streamlit_app.py`) executes use cases directly **in-process** without requiring external database or microservice overhead:
1. **Retriever Ingestion & Model Selection:** Ingests the 1,976 Khmer statutory articles and runs local retrieval using precomputed PyTorch embeddings and NumPy similarity search (`data/indices/khmer_benchmark_embeddings.pt` via our winning Approach A3 checkpoint) or the lexical BM25 baseline with `khmer-nltk` tokenization.
2. **Grounded Synthesis:** Calls DeepSeek Flash (`deepseek-chat`) with strict system prompts requiring article citations (`មាត្រា N` / `(Civil Code 2007, Article N)`).
3. **Compound Citation Verification:** Verifies citations against compound `(law_name, article_number)` identifiers to avoid Civil vs Criminal article collisions, and flags whether cited articles exist in the retrieved context. (Note: Citation verification checks whether cited articles are present in the retrieved context; it does not guarantee that the legal explanation or reasoning is factually correct).
4. **Live Benchmark Audit:** Surfaces real metrics from `results/metrics/summary.csv` and side-by-side comparative diagnostics.

```mermaid
sequenceDiagram
    autonumber
    actor User as Citizen / Researcher
    participant UI as Streamlit Web App (streamlit_app.py)
    participant Ret as In-Process Retriever (A3 PyTorch / BM25)
    participant LLM as DeepSeek Flash (v4 API)
    participant Guard as Compound Citation Verifier

    User->>UI: Select retriever (A3 XLM-R / BM25) & ask Khmer legal question
    UI->>Ret: Search question against 1,976 articles (PyTorch/NumPy)
    Ret-->>UI: Top-5 retrieved statutory articles with text & similarity
    UI->>LLM: Grounded prompt (question + retrieved articles)
    LLM-->>UI: Generated legal response citing articles
    UI->>Guard: Verify (law, article) presence in retrieved context
    Guard-->>UI: Verified / unverified context match status
    UI-->>User: Render legal analysis, citation badges, and retrieved statutory context
```

### Path B: REST Microservice API (Clean Architecture Backend)
The repository also includes a production-style REST API (`FastAPI` at `src/interfaces/api/main:app`) exposing standard OpenAPI endpoints (`/api/v1/qa`, `/api/v1/retrieval`):
- **Dependency Injection:** Resolves domain ports to infrastructure adapters via `src/interfaces/api/dependencies.py`.
- **Flexible Embedders:** Prioritizes local fine-tuned PyTorch A3 weights (`TorchEncoderEmbedding`) when available, falling back gracefully to external embeddings (`OpenAIEmbedding`).
- **Storage Layer:** Integrates PostgreSQL + pgvector for persistent vector indexing, with automatic in-memory fallback when unconfigured.

> **Academic Prototype & Legal Disclaimer:**  
> The application is an academic research prototype, and citation verification does not guarantee legal correctness. Official Khmer gazettes remain the sole legal authority.
