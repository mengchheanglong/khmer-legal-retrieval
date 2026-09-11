# ⚖️ Khmer Legal Article Retrieval — Comparing Deep Learning Retrievers on Cambodian Law

> **Deep Learning Final Project (Individual)** — Bachelor of Software Engineering, Department of Engineering
> Lecturer: Mr. Soklong HIM · Academic year 2026–2027
> **Author:** Long Mengchheang ([@mengchheanglong](https://github.com/mengchheanglong))

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org)
[![Language: Khmer](https://img.shields.io/badge/Language-Khmer%20(ភាសាខ្មែរ)-0b5394.svg)](#2-dataset)

> [!IMPORTANT]
> **Project status (keep this box up to date)**
> - ☐ Topic approved by lecturer (deadline: Week 7)
> - ✅ Khmer corpus built: Civil Code 2007 (1,304 articles) + Criminal Code 2009 (672 articles), converted from legacy Limon fonts to Unicode and split into articles
> - ✅ RAG demo app (FastAPI + Streamlit) — currently runs on the English corpus; switching to Khmer is planned
> - ☐ Khmer test set (~200 human-verified questions)
> - ☐ Deep learning experiments A1–A3 (PyTorch) — **planned, not yet run**
> - ☐ Results table, figures, error analysis — **pending** (cells marked `—` are not results)
> - ☐ Slides in `slides/`

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

The best retriever will power the retrieval stage of a citation-grounded Khmer **RAG** assistant
(Section 9). Retrieval quality caps answer quality, because an LLM cannot cite an article it was never
shown.

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
2. **Leakage guard:** articles relevant to any T-Q question are removed from the training pairs.
3. **Split:** the remaining unique-title pairs are split **by article** 80/10/10 into `train` / `val` / `test_titles`.

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

All three approaches are neural **bi-encoders** implemented in **PyTorch**. Each encodes a question and
an article into vectors, and relevance is their cosine similarity. All are trained with the same
**InfoNCE loss with in-batch negatives**, on the same split, and evaluated by the same harness.

| ID | Model | Dimension A — architecture | Dimension B — training strategy | Trainable params |
|----|-------|----------------------------|----------------------------------|-----------------:|
| **A1** | **BiLSTM** dual encoder: `khmer-nltk` word tokens → randomly initialised 300-d embeddings → 1-layer BiLSTM (256 per direction) → mean pooling; shared towers | Recurrent (RNN) | **Trained from scratch** | — (vocab-dependent; report) |
| **A2** | **XLM-RoBERTa** encoder (`intfloat/multilingual-e5-base`), **frozen**, plus a trainable linear head 768 → 768 (initialised to identity) | Transformer encoder | **Transfer learning, frozen backbone (linear probe)** | ≈ 0.59 M |
| **A3** | Same encoder, **all layers fine-tuned** | Transformer encoder | **Full fine-tuning** | ≈ 278 M |

**Why these three:**
- **A1 vs A2:** does pre-training on 100 languages beat a task-specific RNN when only about 1.4K Khmer pairs exist? This comparison also sets a word segmenter against subword tokenization, a real design question for Khmer.
- **A2 vs A3:** how much of the backbone needs to adapt to formal legal Khmer (capacity vs overfitting)?

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
| Loss | InfoNCE, in-batch negatives, temperature τ |
| Optimiser | AdamW, linear warm-up then linear decay; fp16 on GPU |
| Model selection | Best epoch by **validation MRR@10**. Test sets are evaluated **once**, after tuning. |
| Metrics (T-Q, T-T) | **Recall@5 (primary)**, Recall@1/10, MRR@10, nDCG@10, Hit@5, each with a 95% bootstrap CI |
| Efficiency | Trainable parameters, wall-clock training time, peak GPU memory, query latency |
| Reproducibility | `set_seed(42)` fixes `random`, `numpy`, `torch`, `torch.cuda`; `cudnn.deterministic = True` |
| Checkpointing | `torch.save` every epoch → `results/checkpoints/<approach>/last.pt` (model, optimiser, scheduler, epoch, RNG state), plus `best.pt`; `--resume` guards against Colab disconnects |
| Hardware | Google Colab free T4 GPU (16 GB) or local laptop. The actual device is recorded in `results/metrics/*.json`. |

### 4.1 Hyperparameter tuning (documented in `results/tuning/`)

| Approach | Learning rate | Regularisation | Other |
|----------|---------------|----------------|-------|
| A3 (expected best) | {1e-5, 2e-5, 3e-5} | weight decay {0, 0.01} | τ {0.02, 0.05} |
| A1 | {1e-3, 3e-3} | dropout {0.1, 0.3} | — |
| A2 | {1e-3, 1e-2} | weight decay {0, 0.01} | — |

Changing only the learning rate, loss, optimiser or seed is tuning within an approach, not a new approach.

---

## 5. Results

> Filled in from `results/metrics/summary.csv`. **Do not type numbers by hand.** Regenerate with `python -m src.dl.report`.

### 5.1 Main results — T-Q (Khmer questions, full 1,976-article corpus, no statute filter)

| Approach | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Trainable params | Train time | Hardware |
|----------|---------:|---------:|----------:|-------:|--------:|-----------------:|-----------:|----------|
| BM25 (baseline) | — | — | — | — | — | 0 | — | CPU |
| E5-base zero-shot (reference) | — | — | — | — | — | 0 | — | — |
| **A1** BiLSTM from scratch | — | — | — | — | — | — | — | — |
| **A2** XLM-R frozen + linear probe | — | — | — | — | — | ≈ 0.59 M | — | — |
| **A3** XLM-R full fine-tune | — | — | — | — | — | ≈ 278 M | — | — |

The earlier `data/evaluation_report.json` was produced on the **English** corpus with a gold statute
filter. It is not comparable and will not be reported.

### 5.2 Figures (saved to `results/figures/`, also in the slides)

| Figure | File |
|--------|------|
| Test metrics per approach (grouped bars with CIs) | `results/figures/metrics_bar.png` — pending |
| Overlaid train/val loss and val MRR@10 per approach | `results/figures/learning_curves.png` — pending |
| Recall@k for k = 1…20 | `results/figures/recall_at_k.png` — pending |
| Per-code breakdown (Civil vs Criminal) | `results/figures/per_code.png` — pending |

### 5.3 Discussion (after experiments)

- Which approach wins, and why? Relate it to capacity, transfer learning, inductive bias, tokenization for an unsegmented script, data size and regularisation.
- Over- or under-fitting in each learning curve. A1 is expected to overfit about 1.4K pairs.
- Accuracy vs cost: parameters, training time, latency.

### 5.4 Error analysis (`results/error_analysis/`)

For every approach, list the T-Q questions whose relevant articles are missing from the top 5, then
sort each miss into one of these categories:
1. Vocabulary mismatch: everyday Khmer vs formal legal Khmer
2. Word-segmentation errors (A1, BM25)
3. Multi-article answers
4. Articles with shared titles
5. Civil vs Criminal confusion
6. Near-miss: the relevant article is ranked 6–10

Show 3–5 worked examples with retrieved vs expected article text.

### 5.5 Limitations and future work

- Two codes only.
- Title-based training data does not match real questions well.
- The test set is small.
- Next steps: OCR the Labour Law, add more human-written questions, hard-negative mining, and a trained cross-encoder re-ranker.

---

## 6. Installation & How to Run

### 6.1 Install

```bash
git clone https://github.com/mengchheanglong/rag_cambodia_civil_commercial_law.git
cd rag_cambodia_civil_commercial_law
pip install -r requirements.txt
cp .env.example .env   # only needed for the demo app: DEEPSEEK_API_KEY (+ optional OPENAI_API_KEY)
```

### 6.2 Build the Khmer corpus (chunks are committed in `data/04_chunks/`; re-run to reproduce)

```bash
python -m src.pipeline.download     # official PDFs from ODC → data/01_raw/{kh,en}/
python -m src.pipeline.extract      # Limon → Unicode, page ranges, paragraphs → data/02_extracted/
python -m src.pipeline.chunk        # article chunks with hierarchy → data/04_chunks/
```

### 6.3 Deep learning experiments — *planned commands; not implemented yet*

```bash
python -m src.dl.prepare_splits --seed 42
python -m src.dl.train --config configs/a1_bilstm.yaml
python -m src.dl.train --config configs/a2_xlmr_linear_probe.yaml
python -m src.dl.train --config configs/a3_xlmr_finetune.yaml     # add --resume after a disconnect
python -m src.dl.tune  --config configs/a3_xlmr_finetune.yaml
python -m src.dl.evaluate --all
python -m src.dl.report
```

`notebooks/` will contain one Colab notebook per approach, each calling the same modules.

### 6.4 Trained weights

The A3 checkpoint (≈ 1.1 GB) is larger than 50 MB and is **not** committed. Download link: *to add
(Hugging Face Hub / Google Drive)*. A1 and A2 weights are committed to `results/checkpoints/` if they
are under 50 MB.

### 6.5 Tests

```bash
pytest tests/ -v
```

---

## 7. Repository Structure

```
rag_cambodia_civil_commercial_law/
├── data/
│   ├── 01_raw/kh/              # Official Khmer PDFs (git-ignored; re-download with the pipeline)
│   ├── 01_raw/en/              # English PDFs used by the current demo app
│   ├── 02_extracted/           # Extracted Unicode text (git-ignored)
│   ├── 04_chunks/              # Article chunks: *_kh_chunks.json (study corpus), *_en_chunks.json (app)
│   ├── 05_splits/              # [planned] fixed train/val/test splits + manifest
│   └── indices/                # BM25 index (app)
├── src/
│   ├── pipeline/               # download → extract → chunk
│   ├── infrastructure/
│   │   ├── extractors/         # limon_converter.py, limon_pdf_extractor.py, resources/ (KhmerConverter data + notices)
│   │   └── chunking/           # legal_hierarchical_chunker.py (Khmer + English)
│   ├── dl/                     # [planned] PyTorch: seed, data, models/, losses, train, evaluate, tune, report
│   ├── domain/ application/ interfaces/   # RAG demo app (Clean Architecture)
│   └── evaluation/             # Existing RAG evaluation script
├── configs/                    # [planned] one YAML per approach
├── notebooks/                  # [planned] Colab notebooks, one per approach
├── results/                    # [planned] metrics/ figures/ tuning/ error_analysis/ logs/ checkpoints/
├── slides/                     # [planned] final presentation (PDF/PPTX)
├── tests/                      # unit, integration, evaluation/
└── docs/                       # GOAL.md, TASKS.md
```

---

## 8. Citations

**Data and conversion**
- Open Development Cambodia — Civil Code (Khmer) and Criminal Code (Khmer), CC-BY-SA-4.0.
- KhmerConverter 1.5.1, © 2006–2008 Khmer Software Initiative (khmeros.info). Limon mapping data (LGPL-2.1) and reordering rules (GPL-2.0-or-later). See `src/infrastructure/extractors/resources/THIRD_PARTY_NOTICES.md`.
- `khmer-nltk` — Khmer word segmentation (PyPI).

**Models and methods**
- Conneau, A. et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale.* ACL. (XLM-RoBERTa)
- Wang, L. et al. (2024). *Multilingual E5 Text Embeddings: A Technical Report.* arXiv:2402.05672. Model: [`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base)
- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation. Schuster, M. & Paliwal, K. (1997). *Bidirectional Recurrent Neural Networks.* IEEE TSP.
- Karpukhin, V. et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP.
- van den Oord, A. et al. (2018). *Representation Learning with Contrastive Predictive Coding.* arXiv:1807.03748. (InfoNCE)
- Robertson, S. & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.*

**Libraries:** PyTorch, 🤗 Transformers, PyMuPDF, rank-bm25, FastAPI, Streamlit.

---

## 9. Prototype: RAG Demo Application

```
User question (KH) → Retriever (best of A1–A3) → Top-k Khmer articles → DeepSeek LLM → Khmer answer with citations → Citation check (មាត្រា + Khmer numerals)
```

- **Today:** the app (FastAPI + Streamlit) searches the English corpus using BM25 + OpenAI embeddings + a BGE reranker. Khmer questions are mapped to English terms, and answers can already be generated in Khmer with Khmer citation checks.
- **Planned:** index the Khmer chunks and plug in the best trained retriever through the existing `EmbeddingPort`.

```bash
streamlit run streamlit_app.py                  # UI  → http://localhost:8501
uvicorn src.interfaces.api.main:app --reload    # API → http://localhost:8000/docs
docker compose up --build                       # UI + API + PostgreSQL/pgvector
```

---

## 10. AI Use Disclosure

> *Complete honestly before submission. You must be able to explain and modify every line of code during Q&A.*

- **Tools used:** Claude (Claude Code); *add any others, e.g. Gemini, ChatGPT, GitHub Copilot*
- **Scope of use:** *e.g. RAG app scaffolding; Limon→Unicode converter and Khmer chunking; documentation drafts. List specifics.*
- **Verification:** *e.g. all code reviewed and run manually; converted Khmer text spot-checked; experiment design, results and analysis are my own work.*

---

## ⚠️ Legal Disclaimer

This is a research and educational project. It is **not** legal advice. Consult a qualified legal
professional about specific legal matters.
