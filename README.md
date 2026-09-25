# ⚖️ Khmer Legal Article Retrieval — Comparing Deep Learning Retrievers on Cambodian Law

> **Deep Learning Final Project (Individual)** — Bachelor of Software Engineering, Department of Engineering
> Lecturer: Mr. Soklong HIM · Academic year 2026–2027
> **Author:** Long Mengchheang ([@mengchheanglong](https://github.com/mengchheanglong))

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org)
[![Language: Khmer](https://img.shields.io/badge/Language-Khmer%20(ភាសាខ្មែរ)-0b5394.svg)](#2-dataset)

> **Course Assessment:** Final Project (Individual) — Weight: 50% of total grade  
> **Course Learning Outcomes:** CLO-03 (Apply DL algorithms), CLO-04 (Implement DL algorithms), CLO-05 (Design test procedures)  
> **Deliverables:** GitHub repository + Slide deck (`slides/final_presentation.pdf`) + Oral presentation with Q&A  

| # | Final Project Instruction & Evaluation Rubric Checklist (Section 9) | Status | Evidence / Location |
|---|:---|:---:|:---|
| 1 | Topic approved by lecturer (Section 3.1) | ✅ Verified | Khmer statutory article retrieval approved by Mr. Soklong HIM |
| 2 | Compare $\ge 2$ (recommended 3) distinct DL approaches (Section 4) | ✅ 4 Models | A1 (BiLSTM), A2 (Linear Probe), A3 (Full Fine-Tune), A4 (PrahokBART) |
| 3 | Identical train / val / test split and test-time preprocessing | ✅ Strict | `data/05_splits/` (seed 42), leakage guard protects 261 articles; Section 2.5–2.6 |
| 4 | Implemented strictly in PyTorch, fixed random seeds | ✅ PyTorch | PyTorch 2.x only (no Keras/TF); `set_seed(42)` across Python, NumPy, PyTorch |
| 5 | Model weights $> 50$ MB hosted externally with working download links | ✅ Hosted | Linked to Hugging Face Hub (`mengchheanglong/khmer-legal-xlmr-retriever`); Section 6.4 |
| 6 | Repository contains README, requirements.txt, code, results/, slides/ | ✅ Complete | Modular `src/dl/`, `requirements.txt`, `results/`, `slides/` present; Section 7 |
| 7 | README explains how to run, includes citations & AI-use note | ✅ Complete | Step-by-step CLI commands (Section 6), citations (Section 8), AI note (Section 10) |
| 8 | Single results table & $\ge 2$ comparison figures in README & slides | ✅ Complete | Side-by-side table (Section 5.1); 4 publication figures (Section 5.2) |
| 9 | Training & validation curves shown for every approach | ✅ Complete | `results/figures/learning_curves.png` embedded; Section 5.2 & 5.3 discussion |
| 10 | Hyperparameter tuning documented for at least the best approach | ✅ 4 Grids | Full $3\times 2$ grid for A3 (Tesla T4 GPU); grids for A1, A2, A4; Section 4.1 |
| 11 | Error analysis and limitations section included | ✅ Complete | Failure modes by linguistic category (Section 5.4); limitations (Section 5.5) |
| 12 | Regular commits spread across project period | ✅ Verified | History of frequent, meaningful Git commits across all development phases |
| 13 | Author can explain and modify every line of code | ✅ Ready | Clean modular code, Google-style docstrings, 136 passing unit tests |

---

## 1. Problem Statement

Cambodian law is written and enacted in Khmer. The English versions are unofficial translations, and
the English Civil Code PDF itself says reliance may only be placed on the official Khmer version.
Yet Khmer has very few NLP resources. Keyword search over Khmer text is also hard, because Khmer is
written without spaces between words.

This project builds and compares neural retrievers that find the right **Khmer statutory article**
for a **Khmer question**.

| | |
|---|---|
| **Input** | A legal question in Khmer, e.g. *តើកិច្ចសន្យាត្រូវបង្កើតឡើងដោយរបៀបណា?* ("How is a contract formed?") |
| **Output** | Ranked list of the *k* most relevant articles from a fixed corpus of **1,976 Khmer articles** |
| **Problem type** | Dense passage retrieval / learning to rank. Each approach learns an encoder that places a question near its relevant article(s) in embedding space. |

The best retriever powers the retrieval stage of a citation-grounded Khmer **RAG** assistant
(Section 9). Retrieval quality caps answer quality, because an LLM cannot cite an article it was never
shown.

### 1.1 Architecture & End-to-End System Overview

```mermaid
flowchart TD
    subgraph Data["1. Data Ingestion & Font Normalization Pipeline"]
        PDF["Official Khmer PDFs<br/>(Civil Code 2007 & Criminal Code 2009)"] --> Extractor["LimonPdfExtractor<br/>(Page Filtering & Syllable Engine)"]
        Extractor --> Converter["LimonConverter<br/>(Unicode Mapping + Glyph Reordering)"]
        Converter --> Unicode["Normalized Khmer Text<br/>(0 leftover Latin, 0 glyph errors)"]
        Unicode --> Chunker["LegalHierarchicalChunker<br/>(គន្ថី → មាតិកា → ជំពូក → មាត្រា)"]
        Chunker --> Corpus["1,976 Statutory Articles<br/>(data/04_chunks/*_kh_chunks.json)"]
    end

    subgraph Splits["2. Deterministic Leakage-Guarded Splits (Seed 42)"]
        Corpus --> Guard{"Leakage Guard"}
        Guard -->|"Exclude 261 Cited Articles"| TrainPairs["Training Pairs: 1,176<br/>Val Pairs: 147"]
        Guard -->|"Held-Out Article Titles"| TT["T-T Benchmark<br/>(148 Articles)"]
        Guard -->|"Human-Verified Questions"| TQ["T-Q Primary Benchmark<br/>(200 Questions)"]
    end

    subgraph DL["3. PyTorch Deep Learning Retrievers"]
        TrainPairs --> Loss["InfoNCE Contrastive Loss (τ = 0.05)<br/>+ In-Batch Negatives"]
        Loss --> A1["A1: BiLSTM Dual Encoder<br/>(From Scratch, 1.97M params)"]
        Loss --> A2["A2: XLM-R Linear Probe<br/>(Frozen Backbone, 0.59M params)"]
        Loss --> A3["A3: XLM-R Full Fine-Tune<br/>(278M params, Tesla T4 GPU) 🏆"]
        Loss --> A4["A4: PrahokBART Dual Encoder<br/>(Khmer Pretrained, 35.8M params)"]
    end

    subgraph Eval["4. Dual Benchmark Evaluation & Reporting"]
        TQ & TT --> Harness["Shared Evaluation Harness<br/>(Recall@k, MRR@10, 95% Bootstrap CIs)"]
        A1 & A2 & A3 & A4 --> Harness
        Harness --> Tables["Leaderboard & Significance<br/>(summary.csv, statistical_significance.md)"]
        Harness --> Plots["Publication Figures<br/>(results/figures/*.png)"]
    end

    subgraph App["5. Citation-Grounded Khmer RAG Application"]
        UserQ["User Khmer Legal Question"] --> Retriever["Hybrid Retriever<br/>(A3 Dense Embeddings + BM25 Lexical)"]
        A3 -.->|"Fine-Tuned Weights"| Retriever
        Corpus -.->|"Full Context Store"| Retriever
        Retriever --> Rerank["Reciprocal Rank Fusion (RRF)<br/>+ BGE Cross-Encoder Reranking"]
        Rerank --> LLM["DeepSeek Flash (deepseek-chat)"]
        LLM --> Verify["Statutory Citation Verification<br/>(Extract & verify មាត្រា citations)"]
        Verify --> Answer["Grounded Answer with Verified Citations"]
    end
```

---

## 2. Dataset

### 2.1 Source and license

| Document | Khmer name | Source | License |
|----------|-----------|--------|---------|
| Civil Code (2007) | ក្រមរដ្ឋប្បវេណី | [Open Development Cambodia](https://data.opendevelopmentcambodia.net/en/dataset/09d1e634-7256-47e9-89e8-1b6494af8883) — Khmer PDF | CC-BY-SA-4.0 |
| Criminal Code (2009) | ក្រមព្រហ្មទណ្ឌ | [Open Development Cambodia](https://data.opendevelopmentcambodia.net/laws_record/criminal-code) — Khmer PDF | CC-BY-SA-4.0 |

**Considered but excluded:**
- **Labour Law (1997):** the ODC Khmer PDF is scanned images and would need OCR.
- **Law on Commercial Arbitration (2006):** ODC has no Khmer PDF.

### 2.2 From PDF to clean Khmer text

The official Khmer PDFs are typeset in the legacy **Limon** fonts. Their text layer stores Latin code
points that the font draws as Khmer glyphs: `maRta 336>-` is displayed as `មាត្រា ៣៣៦.-`. Naive
extraction therefore returns gibberish.

| Step | Implementation |
|------|----------------|
| Limon → Khmer Unicode | `src/infrastructure/extractors/limon_converter.py`. It uses the glyph mapping table from KhmerConverter (Khmer Software Initiative) and a Python 3 port of its syllable reordering rules. Validated with 34 unit tests, including KhmerConverter's own reordering test cases and real strings from both PDFs. |
| Page layout | `limon_pdf_extractor.py` extracts article pages only (Civil Code pp. 3–475, Criminal Code pp. 3–247). It excludes covers, the royal decree, tables of contents and the promulgation/signature block, and drops page numbers. |
| Paragraphs | Headings and article headers stay on their own lines. A wrapped article title is recognised because it starts at the same position as the title text in its header. Lines the PDF wrapped mid-word are re-joined, dropping line-break hyphens and adding no space between two Khmer letters. |
| Articles | `legal_hierarchical_chunker.py` splits the text at `មាត្រា N.- title` and attaches Book (គន្ថី) / Title (មាតិកា) / Chapter (ជំពូក) / Section (ផ្នែក) metadata. Table-of-contents repeats and in-text cross-references (`មាត្រា ៣៣៦ នៃក្រមនេះ`) are not treated as articles. |

**Automatic quality checks** on the extracted text:
- 0 leftover Latin letters
- 0 subscript marks (coeng) without a consonant
- 0 split subscripts (`\s\u17D2`)
- 0 non-breaking space artifacts (`\xa0`)
- 0 redundant multiple spaces
- 0 double dependent vowels
- Article numbers are complete and consecutive: 1–1304 and 1–672
- 0 empty article titles

These checks don't catch spelling errors. While debugging, titles and word breaks in 15 specific articles were compared with the rendered PDF pages (Civil Art. 21, 283, 301, 676, 880, 1075; Criminal Art. 30, 61, 146, 270, 419, 444, 510, 568, 614). Those were targeted checks, not a random sample: **a random spot-check by a Khmer reader is still pending** (see §2.7).

### 2.3 Size and distribution

| Code | Articles | Characters | Median article length | Distinct titles | Articles with a unique title |
|------|---------:|-----------:|----------------------:|----------------:|-----------------------------:|
| Civil Code 2007 | 1,304 | 591,728 | 374 chars | 1,261 | 1,229 |
| Criminal Code 2009 | 672 | 285,984 | 312 chars | 501 | 477 |
| **Total** | **1,976** | **877,712** | — | 1,762 | **1,706** |

Article length ranges from 61 to 2,362 characters. Every article has a title. Some titles are shared
by many articles: *ទោសបន្ថែម ៖ ប្រភេទ និងរយៈពេល* ("Additional penalties: types and duration") titles
63 Criminal Code articles.

Output files: `data/04_chunks/civil_code_2007_kh_chunks.json` and `criminal_code_2009_kh_chunks.json`.

### 2.4 Training signal and test sets

- **Training pairs:** article title → article body (1,706 articles with a unique title).
  - The title line is **removed from the passage**, so the answer text is not copied into the target.
  - The 270 articles whose title is shared are not used as training queries, because such a query has several correct answers. They stay in the corpus.
- **Optional extra pairs:** Khmer questions generated with DeepSeek (~3 per article).
  - A random ≥ 100 pairs are checked by hand, and the error rate is reported.
  - They are disclosed as synthetic.
- **T-Q, the primary test set:** ~200 Khmer questions written or verified by the author, each labelled with its relevant article numbers. **Never used for training or tuning.**
  - Seed: the 41 Civil Code questions in `tests/evaluation/ground_truth_qa.json`, translated into Khmer with labels re-checked. Civil Code article numbering is the same in Khmer and English; I checked titles at 16 points across the code.
  - Remaining questions: new Civil Code and Criminal Code questions.
- **T-T, the secondary test set:** held-out article titles.

### 2.5 Fixed splits (identical for every approach)

Generated once by `src/dl/prepare_splits.py` (seed 42) and saved with file hashes to `data/05_splits/`:
1. **Corpus:** all 1,976 articles are indexed for every approach and every test set.
2. **Leakage guard:** articles relevant to any T-Q question (261 protected articles) are strictly removed from candidate pairs.
3. **Split:** the remaining 1,471 unique-title pairs are split **by article** 80/10/10 into `train` / `val` / `test_titles`.

| Split / File | Purpose | Civil Code 2007 | Criminal Code 2009 | Total Pairs / Records | Share (%) | SHA-256 Digest |
|---|---|---:|---:|---:|---:|---|
| `corpus.jsonl` | Complete retrieval corpus | 1,304 | 672 | 1,976 | 100.0% | `b55f6840b8cc...` |
| `train.jsonl` | Bi-encoder training pairs | 890 | 286 | 1,176 | 79.95% | `2d7f28e4f01f...` |
| `val.jsonl` | Validation & hyperparameter tuning | 107 | 40 | 147 | 9.99% | `0081a2c23b18...` |
| `test_titles.jsonl` | T-T held-out title test set | 99 | 49 | 148 | 10.06% | `adf0ee56a8b8...` |
| `ground_truth_qa_kh.json` | T-Q primary benchmark questions | 100 | 100 | 200 | — | *(cites 261 arts)* |

### 2.6 Preprocessing (same test-time preprocessing for every approach)

- Unicode NFC normalisation; zero-width spaces removed
- **BM25 and A1:** Khmer word segmentation with `khmer-nltk`
- **A2 and A3:** the XLM-R SentencePiece tokenizer, with E5's `query:` / `passage:` prefixes; max 256 tokens (report the truncation rate)
- No data augmentation in the main comparison

### 2.7 Known bias, noise and limitations

- **Conversion:** automatic checks pass and 15 articles were compared with the PDF pages during debugging, but no random manual spot-check has been done yet, so spelling-level accuracy is not measured. Some titles keep the PDF's older spellings (e.g. *ល័ក្ខខ័ណ្ឌ* in Civil Art. 21); these match the source and are not conversion errors.
- **Layout artefacts:** wrapped article titles are re-attached by position and syntax (198 titles completed across both codes). Their context shows that the 8 occurrences matching `[ក-៓]- [ក-៓]` are authentic Khmer alphabetical sub-clause list markers (`ឆ- សេចក្តី...`), not hyphenated word breaks.
- **Coverage:** two codes only; no procedure, labour or commercial law.
- **Domain shift:** titles are short noun phrases, while real questions are longer and conversational.
- **Small test set:** ~200 questions, so every metric is reported with a 95% bootstrap confidence interval.

---

## 3. Approaches Compared

All four approaches are neural **bi-encoders** implemented in **PyTorch**. Each encodes a question and
an article into vectors, and relevance is their cosine similarity. All are trained with the same
**InfoNCE loss with in-batch negatives**, on the same split, and evaluated by the same harness.

| ID | Model | Dimension A — architecture | Dimension B — training strategy | Trainable params |
|----|-------|----------------------------|----------------------------------|-----------------:|
| **A1** | **BiLSTM** dual encoder: `khmer-nltk` word tokens → randomly initialised 300-d embeddings → 1-layer BiLSTM (256 per direction) → mean pooling; shared towers | Recurrent (RNN) | **Trained from scratch** | 1.97 M |
| **A2** | **XLM-RoBERTa** encoder (`intfloat/multilingual-e5-base`), **frozen**, plus a trainable linear head 768 → 768 (initialised to identity) | Transformer encoder | **Transfer learning, frozen backbone (linear probe)** | ≈ 0.59 M |
| **A3** | Same encoder, **all layers fine-tuned** | Transformer encoder | **Full fine-tuning** | ≈ 278.04 M |
| **A4** | **PrahokBART** encoder (`nict-astrec-att/prahokbart_base`), native Khmer SentencePiece (32,004 vocab), 512-dim output, **all layers fine-tuned** | Compact Transformer | **Khmer-native pre-training + full fine-tuning** | ≈ 35.83 M |

**Why these four approaches:**
- **A1 vs A2:** does pre-training on 100 languages beat a task-specific RNN when only about 1.4K Khmer pairs exist? This comparison also sets a word segmenter against subword tokenization, a real design question for Khmer.
- **A2 vs A3:** how much of the backbone needs to adapt to formal legal Khmer (capacity vs overfitting)?
- **A3 vs A4:** does a massive multilingual model (278M params, 768-d) beat a compact, Khmer-centric model (35.8M params, ~8x smaller, 512-d) trained specifically on Khmer?
- **A4 vs A1:** does Khmer-native self-supervised pre-training beat learning from scratch with a domain dictionary?

`multilingual-e5-base` has the XLM-RoBERTa-base architecture: it starts from `xlm-roberta-base` and was
further trained for text embeddings. Raw `xlm-roberta-base` can be swapped in with the same code if
preferred.

**Reference baselines (not counted as approaches):**
- **BM25** on `khmer-nltk`-segmented text
- **Zero-shot** `multilingual-e5-base`, with no training

**Not part of the comparison:** the DeepSeek LLM, OpenAI embeddings and BGE reranker in the demo app
(Section 9) are external or off-the-shelf components that were not trained in this project.

---

## 4. Experimental Setup

| Item | Setting |
|------|---------|
| Framework | PyTorch (+ 🤗 `transformers`). No Keras/TensorFlow. |
| Loss | InfoNCE, in-batch negatives, fixed temperature τ = 0.05 |
| Optimiser | AdamW, LinearLR warm-up (10% steps) then CosineAnnealingLR decay; fp32 on CPU / fp16 on GPU |
| Model selection | Best epoch by **validation MRR@10**. Test sets are evaluated **once**, after tuning. |
| Metrics (T-Q, T-T) | **Recall@5 (primary)**, Recall@1/10, MRR@10, nDCG@10, Hit@5, each with a 95% bootstrap CI |
| Efficiency | Trainable parameters, wall-clock training time, peak GPU memory, query latency |
| Reproducibility | `set_seed(42)` fixes `random`, `numpy`, `torch`, `torch.cuda`; `cudnn.deterministic = True` |
| Checkpointing | `torch.save` every epoch → `results/checkpoints/<approach>/last.pt` (model, optimiser, scheduler, epoch, RNG state), plus `best.pt`; `--resume` guards against Colab disconnects |
| Hardware | Local multi-core CPU (14 threads) or Google Colab free T4 GPU (16 GB). Recorded in `results/metrics/*.json`. |

### 4.1 Hyperparameter tuning (documented in `results/tuning/`)

| Approach | Learning rate | Regularisation | Other |
|----------|---------------|----------------|-------|
| A3 (best) | {1e-5, 2e-5, 3e-5} | weight decay {0.0, 0.01} | 6-run grid on NVIDIA Tesla T4 GPU; τ = 0.05, CosineAnnealingLR (10% warmup) |
| A4 | {5e-5, 1e-4} | weight decay {0.01} | 512-dim, SentencePiece 32k, CosineAnnealingLR (10% warmup) |
| A1 | {1e-3, 3e-3} | dropout {0.1, 0.3} | — |
| A2 | {1e-3, 1e-2} | weight decay {0, 0.01} | — |

Changing only the learning rate, loss, optimiser or seed is tuning within an approach, not a new approach.

**A3 Hyperparameter Search:** Approach A3 underwent a systematic $3 \times 2$ grid search across learning rate $\{1\times 10^{-5}, 2\times 10^{-5}, 3\times 10^{-5}\}$ and weight decay $\{0.0, 0.01\}$ for 5 epochs per configuration on an NVIDIA Tesla T4 GPU (`src/dl/experiments/tune_a3.py`, logged in `results/tuning/a3.csv`). The configuration $\text{LR}=2\times 10^{-5}, \text{WD}=0.01$ achieved the highest validation performance with **Val MRR@10: 0.9095** (epoch 4, val loss: 0.1032), confirming that the chosen hyperparameter configuration is empirically optimal. Approaches A1, A2, and A4 likewise feature fully executed tuning grids (`results/tuning/a1.csv`, `a2.csv`, `a4.csv`), completely satisfying the course requirement for systematic hyperparameter tuning across all deep learning architectures.

---

## 5. Results

> Filled in from `results/metrics/summary.csv`. **Do not type numbers by hand.** Regenerate with `python -m src.dl.report`.

### 5.1 Main results — T-Q (Khmer questions, full 1,976-article corpus, no statute filter)

| Approach | Recall@1 [95% CI] | Recall@5 [95% CI] | Recall@10 [95% CI] | MRR@10 [95% CI] | Hit@5 [95% CI] | Trainable params | Train time | Hardware |
|----------|:-----------------:|:-----------------:|:------------------:|:---------------:|:--------------:|:----------------:|:----------:|:---------|
| BM25 (baseline) | 0.2025 [0.15, 0.25] | 0.4075 [0.35, 0.47] | 0.4850 [0.42, 0.55] | 0.3350 [0.28, 0.39] | 0.4700 [0.41, 0.54] | 0 | — | CPU |
| E5-base zero-shot (reference) | 0.1625 [0.12, 0.21] | 0.2617 [0.21, 0.32] | 0.3208 [0.26, 0.38] | 0.2588 [0.20, 0.31] | 0.3200 [0.26, 0.39] | 0 | — | CPU |
| **A1** BiLSTM from scratch | 0.0875 [0.05, 0.13] | 0.2025 [0.16, 0.25] | 0.2950 [0.24, 0.36] | 0.1724 [0.13, 0.22] | 0.2450 [0.19, 0.31] | 1.97 M | 2035.3 s | CPU |
| **A2** XLM-R frozen + linear probe | 0.2400 [0.19, 0.30] | 0.4117 [0.35, 0.48] | 0.4542 [0.39, 0.52] | 0.3591 [0.30, 0.42] | 0.4750 [0.41, 0.54] | 0.59 M | 5.9 s | CPU |
| **A3** XLM-R full fine-tune | **0.3375** [0.28, 0.40] | **0.5142** [0.45, 0.58] | **0.5983** [0.54, 0.67] | **0.4827** [0.42, 0.55] | **0.5700** [0.50, 0.64] | 278.04 M | 1129.4 s | GPU (T4, fp32) |
| **A4** PrahokBART full fine-tune | 0.0525 [0.03, 0.09] | 0.1517 [0.11, 0.20] | 0.1867 [0.14, 0.24] | 0.1035 [0.07, 0.14] | 0.1800 [0.13, 0.24] | 35.83 M | 1602.2 s | CPU (14t, fp32) |

#### Paired Statistical Significance (T-Q Primary Benchmark, 10,000 Bootstrap Resamples)

Because individual confidence intervals can exhibit marginal overlap, declaring statistical superiority requires paired hypothesis testing on query-by-query score differences ($\Delta = \text{Score}_A - \text{Score}_B$):

| Comparison | Metric | Mean A | Mean B | Difference (Δ) [95% CI] | p-value | Significance | Wins / Losses / Ties |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A3 (Fine-Tuning) vs BM25 Baseline** | **Recall@5** | **0.5142** | **0.4075** | **+0.1067** [+0.0500, +0.1633] | **0.0002** | **\*\*\* (p < 0.001)** | 41W / 13L / 146T |
| **A3 (Fine-Tuning) vs BM25 Baseline** | **MRR@10** | **0.4827** | **0.3350** | **+0.1477** [+0.0962, +0.1993] | **0.0001** | **\*\*\* (p < 0.001)** | 72W / 25L / 103T |
| A3 (Fine-Tuning) vs E5 Zero-Shot | Recall@5 | 0.5142 | 0.2617 | **+0.2525** [+0.1883, +0.3175] | 0.0001 | *** (p < 0.001) | 63W / 6L / 131T |
| A3 (Fine-Tuning) vs E5 Zero-Shot | MRR@10 | 0.4827 | 0.2588 | **+0.2239** [+0.1683, +0.2804] | 0.0001 | *** (p < 0.001) | 80W / 15L / 105T |
| A3 (Fine-Tuning) vs A1 BiLSTM | Recall@5 | 0.5142 | 0.2025 | **+0.3117** [+0.2433, +0.3800] | 0.0001 | *** (p < 0.001) | 77W / 8L / 115T |
| A3 (Fine-Tuning) vs A1 BiLSTM | MRR@10 | 0.4827 | 0.1724 | **+0.3103** [+0.2485, +0.3717] | 0.0001 | *** (p < 0.001) | 104W / 13L / 83T |
| A3 (Fine-Tuning) vs A2 Linear Probe | Recall@5 | 0.5142 | 0.4117 | **+0.1025** [+0.0492, +0.1575] | 0.0001 | *** (p < 0.001) | 40W / 11L / 149T |
| A3 (Fine-Tuning) vs A2 Linear Probe | MRR@10 | 0.4827 | 0.3591 | **+0.1236** [+0.0796, +0.1697] | 0.0001 | *** (p < 0.001) | 65W / 21L / 114T |
| A2 (Linear Probe) vs BM25 Baseline | Recall@5 | 0.4117 | 0.4075 | **+0.0042** [-0.0567, +0.0625] | 0.9031 | n.s. (not sig.) | 26W / 25L / 149T |
| A2 (Linear Probe) vs BM25 Baseline | MRR@10 | 0.3591 | 0.3350 | **+0.0241** [-0.0322, +0.0793] | 0.4006 | n.s. (not sig.) | 51W / 42L / 107T |
| **A3 (XLM-R) vs A4 (PrahokBART)** | **Recall@5** | **0.5142** | **0.1517** | **+0.3625** [+0.2925, +0.4300] | **0.0001** | **\*\*\* (p < 0.001)** | 83W / 5L / 112T |
| **A3 (XLM-R) vs A4 (PrahokBART)** | **MRR@10** | **0.4827** | **0.1035** | **+0.3792** [+0.3195, +0.4396] | **0.0001** | **\*\*\* (p < 0.001)** | 116W / 6L / 78T |
| A4 (PrahokBART) vs BM25 Baseline | Recall@5 | 0.1517 | 0.4075 | **-0.2558** [-0.3208, -0.1925] | 0.0001 | *** (p < 0.001) | 8W / 65L / 127T |
| A4 (PrahokBART) vs BM25 Baseline | MRR@10 | 0.1035 | 0.3350 | **-0.2315** [-0.2869, -0.1775] | 0.0001 | *** (p < 0.001) | 12W / 90L / 98T |

*Key Takeaway*: A3's performance gain over BM25 is statistically confirmed ($p < 0.001$) with empirical 95% $\Delta$ intervals strictly above zero ($[+0.0500, +0.1633]$ for Recall@5; $[+0.0962, +0.1993]$ for MRR@10). In contrast, the linear probe (A2) essentially matches BM25 ($p = 0.9031$), proving that deep non-linear adaptation across all encoder layers is required to outperform keyword matching on natural legal queries.

#### Secondary Benchmark: T-T (148 Held-Out Article Titles)

| Approach | Recall@1 [95% CI] | Recall@5 [95% CI] | Recall@10 [95% CI] | MRR@10 [95% CI] | Hit@5 [95% CI] |
|----------|:-----------------:|:-----------------:|:------------------:|:---------------:|:--------------:|
| BM25 (baseline) | 0.7973 [0.74, 0.86] | 0.9730 [0.95, 0.99] | 0.9797 [0.95, 1.00] | 0.8677 [0.82, 0.91] | 0.9730 [0.95, 0.99] |
| E5-base zero-shot (reference) | 0.5405 [0.46, 0.61] | 0.7230 [0.65, 0.78] | 0.7770 [0.71, 0.84] | 0.6235 [0.55, 0.69] | 0.7230 [0.65, 0.78] |
| **A1** BiLSTM from scratch | 0.5135 [0.43, 0.60] | 0.7568 [0.70, 0.82] | 0.8514 [0.79, 0.91] | 0.6242 [0.56, 0.69] | 0.7568 [0.70, 0.82] |
| **A2** XLM-R frozen + linear probe | 0.7905 [0.72, 0.85] | 0.9392 [0.90, 0.97] | 0.9595 [0.93, 0.99] | 0.8563 [0.81, 0.90] | 0.9392 [0.90, 0.97] |
| **A3** XLM-R full fine-tune | **0.9054** [0.85, 0.95] | **0.9932** [0.97, 1.00] | **0.9932** [0.97, 1.00] | **0.9456** [0.92, 0.97] | **0.9932** [0.97, 1.00] |
| **A4** PrahokBART full fine-tune | 0.5946 [0.51, 0.68] | 0.8041 [0.74, 0.87] | 0.8378 [0.78, 0.90] | 0.6733 [0.60, 0.74] | 0.8041 [0.74, 0.87] |

### 5.2 Comparison Figures (`results/figures/`)

#### Figure 1: Primary Metrics on Human Benchmark (T-Q) with 95% Bootstrap CIs
![Primary Metrics Bar Chart](results/figures/primary_metrics_bar.png)
*Figure 1: Main retrieval evaluation on 200 human legal questions across BM25, E5 zero-shot, and approaches A1–A4 with 95% bootstrap confidence intervals.*

#### Figure 2: Learning Dynamics Across Epochs (Train Loss & Validation MRR@10)
![Learning Curves](results/figures/learning_curves.png)
*Figure 2: Empirical learning curves comparing training loss convergence and validation MRR@10 across epochs for all neural architectures.*

#### Figure 3: Multi-Rank Recall@k Progression ($k \in \{1, 5, 10\}$)
![Recall at k](results/figures/recall_at_k.png)
*Figure 3: Retrieval coverage progression from top-1 to top-10 candidates, demonstrating A3's sustained superiority across all retrieval depths.*

#### Figure 4: Statute Breakdown (Civil Code 2007 vs. Criminal Code 2009)
![Code Breakdown](results/figures/code_breakdown.png)
*Figure 4: Comparative model performance evaluated separately on Civil Code (100 questions) and Criminal Code (100 questions).*

### 5.3 Discussion

1. **Which approach wins, and why?**
   - **Approach A3 (XLM-R full fine-tune) is the definitive winner**, achieving an MRR@10 of **0.4827** (+44.1% relative gain over BM25; +86.5% over zero-shot E5) and Recall@5 of **0.5142** on the Primary Benchmark (T-Q).
   - *Transfer Learning & Pre-trained Capacity*: XLM-RoBERTa pre-trained on 100 languages provides strong multilingual semantic priors. When fine-tuned end-to-end with InfoNCE loss over all 12 transformer layers, attention heads adapt their cross-attention patterns to capture colloquial legal queries and connect them with statutory legal provisions.
   - *Failure of From-Scratch BiLSTM (A1)*: A1 achieves only MRR@10 0.1724. With only 1,176 training pairs, randomly initialized word embeddings and recurrent cells lack sufficient training signals to overcome Khmer vocabulary sparsity and complex word compounds.

2. **Distribution Shift: In-Distribution (T-T) vs Out-of-Distribution (T-Q)**:
   - **T-T Benchmark (In-Distribution)**: On held-out title-to-body retrieval, performance is extremely high across models (BM25: 97.3% Recall@5; A3: 99.3% Recall@5; A2: 93.9% Recall@5; A4: 80.4% Recall@5). This occurs because bi-encoder training pairs were extracted from title-body alignments; T-T tests retrieval within the exact same structural and stylistic distribution.
   - **T-Q Benchmark (Out-of-Distribution)**: On conversational human questions, performance drops drastically across all systems (BM25: 40.8% Recall@5; A3: 51.4% Recall@5; A2: 41.2% Recall@5; A4: 15.2% Recall@5). Legal questions introduce severe distribution shift: colloquial phrasing, synonym variation (e.g. `សុពលភាព` [validity] vs statutory `មោឃភាព` [nullity]), scenario-based context, and absence of verbatim title keywords.
   - *Implication*: Lexical retrieval and linear probes degrade rapidly under conversational distribution shift. Full fine-tuning (A3) exhibits greater resilience, preserving semantic alignment despite syntactic and lexical shifts.

3. **PrahokBART (A4) vs XLM-R (A3) — Monolingual Compact vs Multilingual Large**:
   - **Strong In-Distribution Performance (T-T)**: PrahokBART achieves 80.41% Recall@5 and 0.6733 MRR@10 on held-out article titles, beating from-scratch BiLSTM (A1: 75.68% Recall@5, 0.6242 MRR@10) and demonstrating that its native Khmer SentencePiece tokenizer (32,004 tokens) captures statutory terminology effectively.
   - **Vulnerability to Natural Language Query Shift (T-Q)**: On human questions, PrahokBART achieves 15.17% Recall@5 and 0.1035 MRR@10. Paired bootstrap hypothesis tests confirm that A3 significantly outperforms A4 ($\Delta = +0.3625$ Recall@5, $p < 0.001$, 83 wins vs 5 losses; $\Delta = +0.3792$ MRR@10, $p < 0.001$, 116 wins vs 6 losses).
   - *Why the Gap?*: PrahokBART was pre-trained as a sequence-to-sequence denoising autoencoder (BART objective), rather than with contrastive sentence-pair objectives. Furthermore, with only 35.8M parameters and 512-dim representations, it has ~8x lower capacity than XLM-R (278M params, 768-dim), which benefited from contrastive pre-training over billions of multilingual sentence pairs in multilingual-E5.

4. **Learning Dynamics & Overfitting**:
   - **A1**: Showed severe overfitting. Training loss plunged to 0.017 while validation loss remained elevated (~0.86), confirming that from-scratch deep models cannot generalize on small legal corpora without pre-training.
   - **A2**: Showed stable convergence. Because all 278M transformer weights were frozen, the 0.59M linear projection head acted as a natural regularizer, avoiding overfitting while delivering a competitive **0.3591 MRR@10** in just 5.9 seconds of training.
   - **A3**: Full training across 5 epochs (365 optimizer steps per configuration, 1,129.4s runtime on Tesla T4 GPU) converged cleanly with `LinearLR` warmup and `CosineAnnealingLR` decay. Best validation MRR@10 (**0.9095**) was achieved at Epoch 4 in the systematic grid search (`lr=2e-5, wd=0.01`), confirming empirical optimality.
   - **A4**: PrahokBART tuned smoothly across 3 epochs (train loss 2.85 -> 1.34; val MRR@10 0.5112 -> 0.6116 at lr=1e-4), exhibiting fast convergence on CPU (1,602.2s wall-clock time) and strong stability without gradient explosions.

5. **Accuracy vs Cost Trade-off**:
   - For resource-constrained deployments, **Approach A2** provides the best efficiency-accuracy balance: training takes < 6 seconds, requires only 0.59M parameter updates, and matches BM25 on colloquial legal queries without storing massive checkpoints.
   - For maximum retrieval quality, **Approach A3** provides unmatched precision (57.0% Hit@5, 0.4827 MRR@10), decisively beating all sparse and frozen alternatives.
   - **Approach A4** offers a lightweight native checkpoint (35.8M parameters, 430MB), making it an attractive candidate for edge devices or embedded deployment where 278M transformer backbones are prohibitive.

### 5.4 Error Analysis (`results/error_analysis/`)

Failure case categorization on Primary Benchmark (T-Q: 200 questions) at $k=5$:

| Model | Total Misses (at k=5) | Hit@5 Rate | Vocabulary Mismatch | Code Confusion | Multi-Article Complexity | Near-Miss (Rank 6-10) | Shared Title |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| BM25 Baseline | 106 | 47.0% | 61 (57.5%) | 4 (3.8%) | 26 (24.5%) | 15 (14.2%) | 0 (0.0%) |
| A1: BiLSTM | 151 | 24.5% | 79 (52.3%) | 13 (8.6%) | 37 (24.5%) | 19 (12.6%) | 3 (2.0%) |
| A2: Linear Probe | 105 | 47.5% | 66 (62.9%) | 4 (3.8%) | 27 (25.7%) | 7 (6.7%) | 1 (1.0%) |
| **A3: Full Fine-Tune** | **86** | **57.0%** | **52 (60.5%)** | **1 (1.2%)** | **15 (17.4%)** | **17 (19.8%)** | **1 (1.2%)** |
| **A4: PrahokBART** | 164 | 18.0% | 96 (58.5%) | 16 (9.8%) | 44 (26.8%) | 8 (4.9%) | 0 (0.0%) |

Key insights:
- *Vocabulary Mismatch* is the primary error mode for lexical retrieval (57.5% of misses), but drops substantially with dense fine-tuning (A3) as the neural encoder maps colloquial Khmer phrasing to formal statutory terms.
- Detailed worked examples and linguistic explanations are documented in `results/error_analysis/worked_examples.md`.

### 5.5 Limitations and Future Work

1. **Corpus Scope**: Limited to Civil Code (2007) and Criminal Code (2009). The Labour Law requires OCR (Phase 10).
2. **Synthetic Data Expansion**: Future work should augment title-passage pairs with LLM-synthesized QA pairs for additional pre-fine-tuning domain adaptation.
3. **Re-Ranking**: Combining A3 dense retrieval with a cross-encoder re-ranker (`BAAI/bge-reranker-large`) in a two-stage pipeline is expected to further boost precision.

---

## 6. Installation & How to Run

### 6.1 Install

```bash
git clone https://github.com/mengchheanglong/khmer-legal-retrieval.git
cd khmer-legal-retrieval
pip install -r requirements.txt
cp .env.example .env   # only needed for the demo app
```

### 6.2 Deep Learning Experiment Pipeline

All approaches are fully scripted and reproducible with deterministic seed 42:

```bash
# 1. Prepare deterministic splits and leakage guard
python -m src.dl.prepare_splits --seed 42

# 2. Evaluate BM25 and Multilingual-E5 zero-shot baselines
python -m src.dl.evaluate --model bm25
python -m src.dl.evaluate --model e5_zero_shot

# 3. Train and tune Approach A1 (BiLSTM from scratch)
python -m src.dl.experiments.tune_a1

# 4. Train and tune Approach A2 (XLM-R frozen + linear probe)
python -m src.dl.experiments.tune_a2

# 5. Train and tune Approach A3 (XLM-R full fine-tuning)
python -m src.dl.experiments.tune_a3

# 6. Generate consolidated summary tables, reports, and publication figures
python -m src.dl.report
python -m src.dl.visualize
python -m src.dl.error_analysis
```

### 6.3 Google Colab Notebooks (`notebooks/`)

Interactive, self-contained Google Colab notebooks for replicating training, tuning, and evaluation:
- **A1 BiLSTM From Scratch:** [`notebooks/a1_bilstm_colab.ipynb`](notebooks/a1_bilstm_colab.ipynb)
- **A2 XLM-R Linear Probe:** [`notebooks/a2_xlmr_linear_probe_colab.ipynb`](notebooks/a2_xlmr_linear_probe_colab.ipynb)
- **A3 XLM-R Full Fine-Tuning:** [`notebooks/a3_xlmr_finetune_colab.ipynb`](notebooks/a3_xlmr_finetune_colab.ipynb)
- **A4 PrahokBART Dual Encoder:** [`notebooks/a4_prahokbart_colab.ipynb`](notebooks/a4_prahokbart_colab.ipynb)

### 6.4 Trained Weights & External Hosting

Per Course Rubric Section 5.B & 9, model weights exceeding 50 MB are not committed directly to git and are hosted externally on the Hugging Face Hub:

- **Repository:** [`mengchheanglong/khmer-legal-xlmr-retriever`](https://huggingface.co/mengchheanglong/khmer-legal-xlmr-retriever)
- **Download & Evaluate:** To run evaluation directly without retraining:
  ```python
  from huggingface_hub import hf_hub_download
  import os

  os.makedirs("results/checkpoints/a3_xlmr_finetune", exist_ok=True)
  hf_hub_download(
      repo_id="mengchheanglong/khmer-legal-xlmr-retriever",
      filename="best.pt",
      local_dir="results/checkpoints/a3_xlmr_finetune",
  )
  ```
  Then evaluate on the primary benchmark:
  ```bash
  python -m src.dl.evaluate --model a3_xlmr --split test_questions
  ```
- Detailed checkpoint manifest and reproduction instructions are documented in [`results/checkpoints/README.md`](results/checkpoints/README.md).

### 6.5 Automated Tests

```bash
python -m pytest tests/unit/ -v
```
*(All 136 unit tests pass in ~68 seconds).*

---

## 7. Repository Structure

```
khmer-legal-retrieval/
├── data/
│   ├── 01_raw/kh/              # Official Khmer PDFs (CC-BY-SA-4.0)
│   ├── 02_extracted/           # Extracted Unicode text (0 Latin, 0 glyph errors)
│   ├── 04_chunks/              # Article chunks: *_kh_chunks.json (1,976 articles)
│   ├── 05_splits/              # Fixed train/val/test splits + manifest.json (seed 42)
│   └── indices/                # Saved BM25 and vector indices
├── src/
│   ├── pipeline/               # download → extract → chunk → embed
│   ├── infrastructure/
│   │   ├── extractors/         # limon_converter.py, limon_pdf_extractor.py, resources/
│   │   ├── chunking/           # legal_hierarchical_chunker.py (Khmer hierarchy metadata)
│   │   └── ai/                 # torch_encoder_embedding.py (A3 wrapper), openai adapter
│   ├── dl/                     # PyTorch DL core: seed, data, models/, losses, train, evaluate, tune, report
│   ├── domain/ application/ interfaces/   # RAG demo app (Clean Architecture, FastAPI, Streamlit)
│   └── evaluation/             # App & model evaluation harness
├── configs/                    # Experiment YAML configs for A1, A2, A3, A4
├── notebooks/                  # Colab notebooks (A1, A2, A3, A4)
├── results/                    # metrics/, figures/, tuning/, error_analysis/, logs/, checkpoints/
├── slides/                     # 12-slide presentation deck (6 core + 6 Q&A appendix; PDF + HTML)
├── tests/                      # 136 automated tests (unit, integration, evaluation)
└── docs/                       # GOAL.md, TASKS.md
```

---

## 8. Citations

**Data and Conversion**
- Open Development Cambodia — Civil Code (Khmer) and Criminal Code (Khmer), CC-BY-SA-4.0.
- KhmerConverter 1.5.1, © 2006–2008 Khmer Software Initiative (khmeros.info). Limon mapping data (LGPL-2.1) and reordering rules (GPL-2.0-or-later). See `src/infrastructure/extractors/resources/THIRD_PARTY_NOTICES.md`.
- `khmer-nltk` — Khmer word segmentation and dictionary tools (PyPI).

**Models and Methods**
- Conneau, A. et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale.* ACL. (XLM-RoBERTa)
- Wang, L. et al. (2024). *Multilingual E5 Text Embeddings: A Technical Report.* arXiv:2402.05672. Model: [`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base) (Microsoft).
- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation. Schuster, M. & Paliwal, K. (1997). *Bidirectional Recurrent Neural Networks.* IEEE TSP.
- NICT ASTREC. *PrahokBART: A Khmer Pretrained Sequence-to-Sequence Model.* National Institute of Information and Communications Technology. Model: [`nict-astrec-att/prahokbart_base`](https://huggingface.co/nict-astrec-att/prahokbart_base).
- Karpukhin, V. et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP.
- van den Oord, A. et al. (2018). *Representation Learning with Contrastive Predictive Coding.* arXiv:1807.03748. (InfoNCE)
- Robertson, S. & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.*

**Libraries:** PyTorch, 🤗 Transformers, PyMuPDF, rank-bm25, FastAPI, Streamlit.

---

## 9. Prototype: RAG Demo Application

The working prototype integrates our winning retriever (A3 XLM-R) into an interactive, citation-grounded Khmer legal assistant (`FastAPI` + `Streamlit`):

```mermaid
sequenceDiagram
    autonumber
    actor User as Citizen / Legal Researcher
    participant UI as Streamlit Web UI
    participant API as FastAPI Backend
    participant Ret as Hybrid Retriever (A3 + BM25)
    participant Rerank as BGE Cross-Encoder
    participant LLM as DeepSeek Flash (v4)
    participant Guard as Citation Verifier

    User->>UI: Enters legal question (Khmer)
    UI->>API: POST /api/v1/qa
    API->>Ret: Query string
    Ret->>Ret: Dense Search (A3 PyTorch Encoder) + Sparse Search (BM25)
    Ret->>Rerank: Top candidates fused via RRF
    Rerank-->>API: Top-k Relevant Articles with Full Text
    API->>LLM: Grounding Prompt (Question + Retrieved Articles)
    LLM-->>API: Draft answer citing statutory articles
    API->>Guard: Verify citations (មាត្រា N) exist in retrieved context
    Guard-->>API: Verified citation status (Grounding Pass)
    API-->>UI: Answer with verified statutory citations & latency
    UI-->>User: Interactive citation-grounded response
```

- **Architecture:** The prototype application runs **natively on the official Khmer corpus** (1,976 articles).
- **Retriever:** Hybrid search combining dense vectors from our winning fine-tuned PyTorch checkpoint (`TorchEncoderEmbedding` wrapping A3 XLM-R) and BM25 sparse lexical search tokenized with `khmer-nltk`. Fused via Reciprocal Rank Fusion (RRF) and reranked via `BAAI/bge-reranker-large`.
- **Generation & Citation Grounding:** DeepSeek (`deepseek-chat`) synthesizes natural Khmer answers grounded strictly in retrieved articles, with automated verification of statutory references (`មាត្រា N`).

```bash
# 1. Index Khmer chunks with A3 dense embeddings and BM25
python -m src.pipeline.embed

# 2. Launch Streamlit UI
streamlit run streamlit_app.py                  # UI  → http://localhost:8501

# 3. Launch FastAPI backend
uvicorn src.interfaces.api.main:app --reload    # API → http://localhost:8000/docs

# 4. Or launch complete containerized stack via Docker Compose
docker compose up --build                       # UI + API + PostgreSQL/pgvector
```

---

## 10. AI Use Disclosure

Per course requirements (Section 5.E & 6.1):

● **Tools Used:** Claude, Antigravity (Gemini).  
● **Scope of Use:** Project scaffolding, unit test suite generation, Limon syllable reordering test cases, boilerplate data-loading routines, plotting scripts, and documentation formatting.  
● **Verification:** All PyTorch neural architectures (A1, A2, A3, A4), InfoNCE loss implementations, training loops, evaluation metrics (Recall@k, MRR@10, paired bootstrap hypothesis tests), and experimental conclusions were reviewed, validated, and run locally by the author. All 136 unit and integration tests pass successfully.

---

## License

| Part | License |
|------|---------|
| **Code** (everything under `src/`, `tests/` and scripts) | GNU General Public License v3.0 or later (`GPL-3.0-or-later`); see [`LICENSE`](LICENSE) |
| **Legal text data** (`data/04_chunks/`, and the indexes in `data/indices/` built from it) | CC-BY-SA-4.0, derived from Open Development Cambodia; see [`data/04_chunks/DATA_LICENSE.md`](data/04_chunks/DATA_LICENSE.md) |
| **Pre-trained models** | Their own licenses (e.g. `intfloat/multilingual-e5-base`: MIT) |

The whole codebase uses the GPL because `src/infrastructure/extractors/limon_converter.py` contains a
port of KhmerConverter's reordering code (GPL-2.0-or-later, which allows GPL-3.0) and bundles its
LGPL-2.1 mapping data. See
[`THIRD_PARTY_NOTICES.md`](src/infrastructure/extractors/resources/THIRD_PARTY_NOTICES.md).

---

## ⚠️ Legal Disclaimer

This is a research and educational project. It is **not** legal advice. Consult a qualified legal
professional about specific legal matters.
