# System Architecture & Component Specification

This document details the end-to-end architecture and runtime execution models of the Khmer Legal Retrieval and Question-Answering system.

---

## 1. End-to-End Pipeline Data Flow

```mermaid
flowchart TD
    subgraph Data["1. Data Ingestion & Font Normalization Pipeline"]
        PDF["Official Khmer PDFs (Civil & Criminal Code)"] --> Extractor["LimonPdfExtractor & Syllable Engine"]
        Extractor --> Converter["LimonConverter (Unicode Mapping & Syllable Reordering)"]
        Converter --> Unicode["Normalized Khmer Text (0 Latin / 0 glyph errors)"]
        Unicode --> Chunker["LegalHierarchicalChunker (គន្ថី, ជំពូក, មាត្រា)"]
        Chunker --> Corpus["1,976 Statutory Articles"]
    end

    subgraph Splits["2. Deterministic Leakage-Guarded Splits (Seed 42)"]
        Corpus --> Guard["Leakage Guard (Exclude 261 Cited Articles)"]
        Guard --> TrainPairs["Training Pairs (1,176 articles)"]
        Guard --> ValPairs["Validation Pairs (147 articles)"]
        Guard --> TT["Held-Out Titles T-T (148 articles)"]
        Guard --> TQ["Human Questions T-Q (200 questions)"]
    end

    subgraph DL["3. PyTorch Deep Learning Retrievers"]
        TrainPairs --> Loss["InfoNCE Contrastive Loss (tau = 0.05)"]
        Loss --> A1["A1: BiLSTM Dual Encoder (From Scratch, 1.97M)"]
        Loss --> A2["A2: XLM-R Linear Probe (Frozen Backbone, 0.59M)"]
        Loss --> A3["A3: XLM-R Full Fine-Tune (278M, Tesla T4 GPU) [Best]"]
        Loss --> A4["A4: PrahokBART Dual Encoder (Khmer Pretrained, 35.8M)"]
    end

    subgraph Eval["4. Dual Benchmark Evaluation & Reporting"]
        A1 --> Harness["Shared Evaluation Harness (Recall@k, MRR@10, 95% CIs)"]
        A2 --> Harness
        A3 --> Harness
        A4 --> Harness
        TT --> Harness
        TQ --> Harness
        Harness --> Tables["Consolidated Leaderboard & Significance (p < 0.001)"]
        Harness --> Plots["4 Publication Figures (results/figures/*.png)"]
    end

    subgraph App["5. Citation-Grounded Khmer RAG Application"]
        UserQ["User Khmer Legal Question"] --> Retriever["Hybrid Retriever (A3 Dense + BM25 Lexical)"]
        A3 --> Retriever
        Retriever --> Rerank["RRF Fusion & Cross-Encoder Reranker"]
        Rerank --> LLM["DeepSeek Flash (v4) Grounded Generation"]
        LLM --> Verify["Statutory Citation Verification (មាត្រា N)"]
        Verify --> Answer["Grounded Answer with Verified Citations"]
    end
```

---

## 2. Interactive RAG Prototype Execution Path

The project provides two execution paths designed according to **Clean Architecture** principles:

### Path A: Interactive Streamlit UI (In-Process Defense Mode)
The interactive demo (`streamlit_app.py`) executes use cases directly **in-process** without requiring external database or microservice overhead:
1. **Retriever Ingestion & Model Selection:** Ingests the 1,976 Khmer statutory articles and runs local retrieval via our winning PyTorch checkpoint (Approach A3: XLM-R full fine-tune) or the lexical BM25 baseline with `khmer-nltk` tokenization.
2. **Grounded Synthesis:** Calls DeepSeek Flash (`deepseek-chat`) with strict system prompts requiring article citations (`មាត្រា N` / `(Civil Code 2007, Article N)`).
3. **Compound Citation Verification:** Verifies citations against compound `(law_name, article_number)` identifiers to avoid Civil vs Criminal article collisions, and flags whether retrieved articles provide factual support vs explicit lack-of-coverage disclaimers.
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
    UI->>Ret: Search question against 1,976 articles
    Ret-->>UI: Top-5 retrieved statutory articles with text & similarity
    UI->>LLM: Grounded prompt (question + retrieved articles)
    LLM-->>UI: Generated legal response citing articles
    UI->>Guard: Verify (law, article) existence in retrieved context & check factual support
    Guard-->>UI: Verified / unverified status + factual support notes
    UI-->>User: Render legal analysis, citation badges, and retrieved statutory context
```

### Path B: REST Microservice API (Clean Architecture Backend)
The repository also includes a production-style REST API (`FastAPI` at `src/interfaces/api/main:app`) exposing standard OpenAPI endpoints (`/api/v1/qa`, `/api/v1/retrieval`):
- **Dependency Injection:** Resolves domain ports to infrastructure adapters via `src/interfaces/api/dependencies.py`.
- **Flexible Embedders:** Prioritizes local fine-tuned PyTorch A3 weights (`TorchEncoderEmbedding`) when available, falling back gracefully to external embeddings (`OpenAIEmbedding`).
- **Storage Layer:** Integrates PostgreSQL + pgvector for persistent vector indexing, with automatic in-memory fallback.
