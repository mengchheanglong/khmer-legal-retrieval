# Defense Review — 2026-10-15

## Overview

This document records findings from the final pre-defense review of the Khmer Legal Retrieval project.

## Slide Deck

The current presentation is `slides/Khmer_Legal_Retrieval_Defense.pdf` (14 main slides + 4 appendix slides).
Older HTML/PDF decks have been archived to `slides/archive/`.

## Key Corrections Made

| Location | Issue | Correction |
|:---|:---|:---|
| Slide 6 (Contrastive Learning) | Stated batch of 32 with 31 negatives | Qualified as Illustrative; stated actual A3 batch size B=16 (15 negatives) |
| Slide 10 (Error Analysis) | A3 Recall@10 = 64.3% | Corrected to **59.83%** per `results/metrics/summary.csv` |
| Slide 11 (Live Demo) | BM25 rank 3 = Article 1107 | Corrected to **Article 1124** per `results/error_analysis/bm25_rankings.json` |
| Slide 13 (Architecture Table) | PrahokBART embedding dim = 768 | Corrected to **512** per `src/dl/models/prahokbart.py` |
| Slide 15 (Bootstrap Table) | Discrepant CIs and W/L/T counts | Synchronized exactly with `docs/statistical_significance.md` |
| Slide 15 (Methodology) | "Paired studentized bootstrap" | Corrected to **"Paired non-parametric bootstrap"** |
| README.md | Relative Recall@5 gain stated incorrectly | Corrected to +26.2% relative Recall@5; +44.1% relative MRR@10 |
| README.md | Error categories presented as proven causes | Clarified as heuristic labels from automatic categorizer |
| Streamlit UI | A1, A2, A4 selector options fell through to BM25 | Removed non-implemented options; selector now shows only A3, Hybrid, and BM25 |
| Streamlit UI | Benchmark questions had incorrect article references | Questions now loaded directly from `tests/evaluation/ground_truth_qa_kh.json` |

## Demo Rehearsal Plan (90 seconds)

1. **Open** `http://localhost:8501` — verify 3 retriever options visible in sidebar (A3, Hybrid, BM25).
2. **Navigate** to Tab 2 "Statutory Article Search".
3. **Load** the benchmark demo query: «តើនរណាខ្លះអាចស្នើសុំឱ្យតុលាការប្រកាសបើកអាណាព្យាបាលទូទៅ?» (Expected: Art. 24).
4. **Switch** to BM25 in the sidebar — show Article 24 is **missing** from top 10.
5. **Switch** to A3 — show Article 24 retrieved at **Rank #1** with cosine similarity ~0.62.
6. **Point out** the mechanism: BM25 matches "អាណាព្យាបាល" in Book 8 admin articles; A3's self-attention aligns "ស្នើសុំបើក" with Article 24's commencement provisions.

### Pre-Demo Setup

```bash
# Build the demo index (run once after downloading the A3 checkpoint)
python scripts/build_demo_index.py

# Start the Streamlit UI
streamlit run streamlit_app.py
```

## Outstanding Limitations

1. **Manual Khmer extraction verification**: The author has not manually compared 20–30 extracted articles against the official Khmer PDFs. The extraction error rate is **unmeasured**. Consecutive article numbering and automated tests do not establish textual accuracy.
2. **Topic approval**: Verbal approval from Mr. Soklong HIM was confirmed; the original approval date was not recorded.
3. **Error categories**: The vocabulary-mismatch label (60.5% of A3 misses) is a residual category from automatic heuristic classification, not a manually established cause.
4. **Citation verification**: The "Verified Citation" badge confirms that the article number exists in the retrieved context. It does not guarantee every claim in the generated answer is factually correct.
