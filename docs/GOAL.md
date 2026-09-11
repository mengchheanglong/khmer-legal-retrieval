# 🎯 Project Goal

## Context

This repository is the **individual Deep Learning Final Project** (50% of the course grade;
CLO-03, CLO-04, CLO-05). The course requires the project to:

- solve a real-world problem on a real dataset;
- **train or fine-tune ≥ 2 (ideally 3) distinct deep learning approaches in PyTorch**;
- evaluate every approach on the **same held-out test set** with the same metrics;
- explain with evidence which approach works best and why.

The lecturer encourages Khmer-language projects, and the study is **pure Khmer**: Khmer corpus, Khmer
training pairs, Khmer test questions. The RAG legal assistant is the **prototype** that uses the best
model. Calling APIs (DeepSeek, OpenAI) or using off-the-shelf models without training them does
**not** count as a deep learning approach.

---

## Research Question

> For retrieving the relevant Cambodian law article for a Khmer question, how does a **BiLSTM trained
> from scratch** on segmented Khmer compare with a pre-trained **XLM-RoBERTa encoder used frozen (linear
> probe)** and the same encoder **fully fine-tuned**, when only about 1.7K title-based Khmer training
> pairs are available?

## Problem Definition

| | |
|---|---|
| Input | Legal question in Khmer |
| Output | Top-*k* ranked articles from a fixed corpus of 1,976 Khmer articles |
| Type | Dense retrieval / learning to rank (contrastive metric learning) |
| Primary metric | Recall@5 on ~200 human-verified Khmer questions (T-Q) |
| Secondary metrics | Recall@1/10, MRR@10, nDCG@10, Hit@5; params, training time, latency |

## Approaches (see README §3)

| ID | Architecture (dim. A) | Training strategy (dim. B) |
|----|----------------------|----------------------------|
| A1 | BiLSTM dual encoder on `khmer-nltk` word tokens | From scratch, random embeddings |
| A2 | XLM-RoBERTa encoder (`multilingual-e5-base`) | Frozen backbone + linear projection head |
| A3 | XLM-RoBERTa encoder (`multilingual-e5-base`) | Full fine-tuning |
| *Baselines* | BM25 on segmented Khmer; E5-base zero-shot | Not counted as approaches |

---

## Corpus

| Document | Year | Khmer PDF | Status | Articles |
|----------|------|-----------|--------|---------:|
| Civil Code (ក្រមរដ្ឋប្បវេណី) | 2007 | Limon fonts, text layer | ✅ converted + chunked | 1,304 |
| Criminal Code (ក្រមព្រហ្មទណ្ឌ) | 2009 | Limon fonts, text layer | ✅ converted + chunked | 672 |
| Labour Law (ច្បាប់ស្តីពីការងារ) | 1997 | Scanned images | ⬜ needs OCR (stretch) | — |
| Law on Commercial Arbitration | 2006 | Not on ODC | ❌ excluded | — |

The corpus is **frozen once the splits are generated**. Adding a document later would change the test
conditions for every approach.

---

## Success Criteria (mapped to the grading rubric)

### Technical & Intellectual Depth (50%)
- [ ] 3 distinct DL approaches, all correctly implemented and trained in PyTorch
- [ ] Identical corpus, split, test-time preprocessing and metrics for every approach; no statute filter at test time
- [ ] No leakage: test-question articles removed from training; title line removed from training passages; test sets used once
- [ ] Hyperparameter tuning (learning rate + ≥ 1 regularisation choice) for at least A3, documented
- [ ] Train/val learning curves for every approach, with over/under-fitting discussed
- [ ] Error analysis of failed queries (including segmentation and shared-title cases) and a limitations section
- [ ] Results explained with course concepts (capacity, transfer learning, inductive bias, tokenization, data size, regularisation)

### Documentation & Prototyping (15%)
- [ ] README complete (title/name, problem, dataset, approaches, results table, how to run, citations, AI-use note)
- [ ] `requirements.txt` with exact pinned versions, including `torch`, `transformers`, `khmer-nltk`
- [ ] Fixed seeds; checkpoint save/resume; one command per experiment
- [ ] `results/` holds metrics (CSV/JSON), figures, tuning logs, weights or download links
- [ ] Regular, meaningful commits across the project period
- [ ] Working prototype: the best trained retriever over the Khmer corpus in the RAG app

### Presentation, Communication, Q&A (35%)
- [ ] 10–20 slides in `slides/`, following the required structure, delivered in 10 minutes
- [ ] Able to explain every line of code, including the Limon converter, InfoNCE, pooling and checkpointing

---

## Key Technical Decisions

### 1. Pure Khmer instead of bilingual
- **Authoritative text:** Khmer is the official language of the law; the English texts are unofficial translations.
- **Lecturer's encouragement:** the course brief welcomes Khmer projects.
- **English doesn't line up:** the English Criminal Code PDF has 703 numbered articles against 672 in Khmer, so article-number alignment would be wrong.
- **Simpler:** one language keeps the pipeline and the explanation simple.

### 2. Convert the Limon text layer instead of OCR
The Khmer PDFs have a complete text layer in legacy Limon fonts. Converting it with the KhmerConverter
mapping and reordering rules is lossless apart from rare edge cases, and much more accurate than Khmer
OCR. Scanned documents (Labour Law) are out of scope for now.

### 3. Article-level chunking with hierarchy
Each article (`មាត្រា N.- title`) is one retrieval unit with Book / Title / Chapter / Section metadata.
Table-of-contents repeats, cross-references and signature blocks are excluded.

### 4. Article titles as weak supervision
There is no labelled Khmer legal-question dataset. Every article title is a free query→article pair.
- **Shared titles excluded:** titles used by more than one article (270 articles) are ambiguous and are not used as queries.
- **Title removed from passage:** the title line is stripped from the passage so the model can't just copy-match it.

### 5. Tokenization is part of the comparison
Khmer has no spaces between words.
- **A1 and BM25:** a word segmenter (`khmer-nltk`)
- **A2 and A3:** XLM-R's SentencePiece vocabulary, which includes Khmer

This contrast is part of the explanation of the results.

### 6. Same loss and evaluation harness for all approaches
InfoNCE with in-batch negatives and one shared evaluation module, so the differences come only from
architecture and training strategy.

---

## Milestones (aligned with the course timeline)

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| ✅ Done | RAG app; Khmer corpus (Limon conversion, 1,976 articles) | `data/04_chunks/*_kh_chunks.json` |
| **Week 7** | **Topic approval** (problem, Khmer corpus, A1–A3, metrics) | Lecturer sign-off, recorded in README |
| Week 8 | Manual spot-check of converted text; Khmer test questions; fixed splits; shared seed/checkpoint/eval modules | `data/05_splits/`, `src/dl/` |
| **Week 10** | **Progress check:** show the split and at least one trained approach | A2 or A1 learning curves + val metrics |
| Weeks 11–12 | Train all approaches, tune A3, final test evaluation | `results/metrics/`, `results/tuning/` |
| Week 13 | Figures, error analysis, limitations; best model in the Khmer RAG app | `results/figures/`, updated README |
| **Week 14** | **Final submission:** GitHub link with slides inside | `slides/*.pdf` |
| **Week 15** | 10-min presentation + 5-min Q&A | — |

**Rule:** don't start the final experiments before the topic is approved. Otherwise the project mark
is capped at 60%.
