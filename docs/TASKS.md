# 📋 Implementation Tasks

> Track progress by marking tasks: `[ ]` todo → `[/]` in progress → `[x]` done
> Source of requirements: *DL Final Project Instruction & Evaluation Rubric* (Mr. Soklong HIM, 2026–2027)
> Direction: **pure Khmer** study (see docs/GOAL.md, Key Technical Decision 1)

---

## 0. Instruction Compliance Audit (last reviewed: 2026-09-11)

| Requirement (instruction §) | Status | Gap / action |
|-----------------------------|:------:|--------------|
| Topic approved by lecturer (§3.1, Week 7) | ⏳ | Presentation slides ready in `slides/`; record approval date in README upon Week 7 defense |
| ≥ 2 (ideally 3) distinct DL approaches trained/fine-tuned by me (§2, §4) | ✅ | Implemented, trained, and evaluated 4 distinct approaches: A1 (BiLSTM), A2 (XLM-R Linear Probe), A3 (XLM-R Full Fine-Tuning), and A4 (PrahokBART Dual Encoder) |
| All models in PyTorch; no Keras/TF (§5B) | ✅ | All models and loss functions implemented purely in PyTorch; dependencies pinned in `requirements.txt` |
| Same train/val/test split and test preprocessing for all (§2, §5A) | ✅ | Fixed splits generated in `data/05_splits/` with deterministic seed 42 and leakage guard |
| Dataset source, license, size, distribution, bias (§5A) | ✅ | 1,976 articles, CC-BY-SA-4.0, documented in README §2 and `data/05_splits/manifest.json` |
| Checkpoint save/load with `torch.save` (§5B) | ✅ | Shared trainer saves `last.pt`, `best.pt`, and RNG states, with `--resume` support |
| Seeds fixed for Python, NumPy, PyTorch (§5B) | ✅ | Enforced via `src/dl/seed.py` (seed 42) across Python, NumPy, PyTorch, and cuDNN |
| Clean, modular, runnable code (§5B) | ✅ | Modular packages under `src/dl/`, `src/application/`, `src/infrastructure/`, with 132 passing tests |
| Task-appropriate metrics (§5C) | ✅ | Recall@1/5/10, MRR@10, nDCG@10, Hit@5, and 95% bootstrap CIs in `src/dl/evaluate.py` |
| Train/val curves for every approach (§5C) | ✅ | Per-epoch CSV logs in `results/logs/` and overlaid curves in `results/figures/learning_curves.png` |
| Hyperparameter tuning: LR + one regularisation choice (§5C) | ✅ | 2×2 tuning grids for A1, A2, and A3 saved in `results/tuning/` |
| Params, training time, hardware per approach (§5C) | ✅ | Logged to `results/metrics/*.json` and summarized in README §5.1 table |
| Error analysis (§5C) | ✅ | Quantitative categorization across 4 models and linguistic worked examples in `results/error_analysis/` |
| Single results table + ≥ 2 comparison figures (§5D) | ✅ | Markdown table in README §5.1 + 4 publication figures in `results/figures/` |
| Explanation with course concepts; limitations (§5D) | ✅ | Comprehensive discussion in README §5.3–§5.5 and presentation deck |
| Cite reused code/models/data (§5E) | ✅ | Documented in README §8 and `THIRD_PARTY_NOTICES.md` |
| AI-use disclosure in README (§5E) | ✅ | Fully and honestly disclosed in README §10 |
| Repo contains README, requirements.txt, code, `results/`, `slides/` (§6.1) | ✅ | All required directories and files populated and committed |
| `requirements.txt` with exact packages (§6.1) | ✅ | Exact package versions pinned in `requirements.txt` |
| Weights > 50 MB hosted externally with link (§9) | 🟡 | A3 checkpoint (~1.1 GB) saved locally; upload instructions documented in `results/checkpoints/README.md` |
| Regular commit history (§6.1) | ✅ | Regular, small, descriptive commits maintained across all PRs #1–#7 |
| Slides 10–20, required structure (§6.2) | ✅ | 18 slides compiled to `slides/final_presentation.pdf` and interactive HTML viewer |

---

## Phases 1–5: RAG application (done)
- [x] Project scaffolding, `.env.example`, `.gitignore`, git repository
- [x] English PDF download and extraction (PyMuPDF, header/footer clipping, TOC skipping)
- [x] Article-level chunker with hierarchy metadata; `LegalChunk` Pydantic model
- [x] Embedding adapter (OpenAI), pgvector repository + NumPy fallback, BM25 retriever
- [x] Hybrid retrieval with RRF, cross-encoder reranker, DeepSeek LLM adapter
- [x] Citation parsing and verification (English and Khmer numerals); Khmer→English query mapping
- [x] FastAPI endpoints, Streamlit UI, unit and integration tests

---

## Phase 5b: Khmer Corpus (done)

### 5b.1 Sources
- [x] Find Khmer PDFs on ODC; confirm CC-BY-SA-4.0 for the Civil Code and Criminal Code
- [x] Download Civil Code (KH) and Criminal Code (KH); add them to `src/pipeline/download.py`
- [x] Check the Labour Law (KH): scanned images, so excluded pending OCR
- [x] Check Commercial Arbitration: no Khmer PDF on ODC, so excluded

### 5b.2 Limon → Unicode conversion (`src/infrastructure/extractors/`)
- [x] Diagnose: text layer in legacy Limon fonts (not scanned)
- [x] `limon_converter.py`: KhmerConverter mapping data + Python 3 port of its reordering rules
- [x] Bundle `resources/khmer_legacy_fontdata.xml`, license and `THIRD_PARTY_NOTICES.md`
- [x] Unit tests: KhmerConverter reorder cases + real Limon strings (35 tests)
- [ ] **Manual spot-check:** a Khmer reader compares 20–30 random converted articles with the PDF pages and records the error rate in README §2.2 (not done yet — the earlier automated audit only counted regex patterns)
- [x] Licence: code GPL-3.0-or-later (`LICENSE`), legal-text data CC-BY-SA-4.0 (`data/04_chunks/DATA_LICENSE.md`)

### 5b.3 Extraction and chunking
- [x] `LimonPdfExtractor`: page ranges, page-number removal, headings on own lines, wrapped lines re-joined
- [x] Exclude tables of contents (Civil pp. 476–487, Criminal pp. 248–292) and the promulgation block
- [x] Khmer chunker: `មាត្រា N.- title` headers, titles, Book/Title/Chapter/Section metadata
- [x] Skip TOC repeats and cross-references; stop article content at the next heading; reset stale lower headings
- [x] Result: Civil Code 1,304 articles (1–1304), Criminal Code 672 (1–672), all titled; unit tests added
- [x] Re-attach wrapped article titles by alignment and syntax (198 titles completed across both codes); confirmed from context that the 8 matches for `[ក-៓]- [ក-៓]` are authentic Khmer alphabetical sub-clause list markers (`ឆ-`), not hyphen breaks
- [x] Keep Limon words intact across zero-gap placeholder spaces (fixed broken words such as *ផ ន្ត្ទាទោស*, *នីតិប្បញ្ញត ្តិ*) while keeping real word spaces; normalise non-breaking and double spaces
- [x] Commit `data/04_chunks/*_kh_chunks.json` and the manifest

---

## Phase 6: Deep Learning Study (graded core) 🔴

### 6.0 Topic approval (before final experiments)
- [x] One-slide pitch: Khmer problem, corpus (1,976 articles), A1–A3, Recall@5 / MRR@10 (18-slide deck in `slides/`)
- [ ] Lecturer approval by **Week 7**; record the date in README

### 6.1 Khmer test set and splits
- [x] Translate the 41 Civil Code questions in `tests/evaluation/ground_truth_qa.json` into Khmer; re-verify `expected_articles` against the Khmer text
- [x] Write new Civil Code and Criminal Code questions to reach ~200 (T-Q); store as `tests/evaluation/ground_truth_qa_kh.json` with `question_kh`, `expected_articles`, `law_name`, `author/verifier`
- [ ] (Optional) Generate ~3 Khmer questions per article with DeepSeek; hand-check ≥ 100 random pairs; mark as synthetic
- [x] `src/dl/prepare_splits.py`: title→body pairs for the 1,706 unique-title articles; strip the title line from passages
- [x] Leakage guard: drop pairs whose article is relevant to any T-Q question
- [x] Split by article 80/10/10 (seed 42) → `train`, `val`, `test_titles`; save `data/05_splits/*.jsonl` + `manifest.json` (counts, seed, SHA-256)
- [x] Report split sizes and per-code distribution in README §2

### 6.2 Shared infrastructure (`src/dl/`)
- [x] Add pinned `torch`, `transformers`, `khmer-nltk`, `pandas`, `matplotlib`, `pyyaml` to `requirements.txt`
- [x] `seed.py`: `set_seed()` for `random`, `numpy`, `torch`, `torch.cuda`, cuDNN deterministic
- [x] `text.py`: NFC normalisation, zero-width space removal, `khmer-nltk` segmentation (cached)
- [x] `data.py`: PyTorch `Dataset` / `DataLoader` for pairs, corpus and test queries
- [x] `losses.py`: InfoNCE with in-batch negatives (temperature τ)
- [x] `train.py`: YAML-driven loop; per-epoch train loss, val loss, val MRR@10 → `results/logs/<run>.csv`
- [x] Checkpointing: `torch.save` of model/optimiser/scheduler/epoch/RNG → `last.pt`, `best.pt`; `--resume`
- [x] Log trainable params, wall-clock time, device, peak GPU memory → `results/metrics/<run>.json`
- [x] `evaluate.py`: **one** harness — encode the full 1,976-article corpus, rank, Recall@1/5/10, MRR@10, nDCG@10, Hit@5, bootstrap 95% CI
- [x] Unit tests for metrics, InfoNCE and segmentation on toy inputs

### 6.3 Baselines (not counted as approaches)
- [x] BM25 on `khmer-nltk` tokens through the shared harness
- [x] `multilingual-e5-base` zero-shot through the shared harness

### 6.4 Approach A1 — BiLSTM dual encoder, from scratch
- [x] Vocabulary from the train split (min frequency, `<unk>`); random 300-d embeddings; 1-layer BiLSTM (256/dir); mean pooling; shared towers
- [x] Train with InfoNCE; save curves and checkpoints
- [x] Tune LR {1e-3, 3e-3} × dropout {0.1, 0.3}

### 6.5 Approach A2 — XLM-R encoder frozen + linear probe
- [x] Freeze all backbone params; verify trainable params ≈ 0.59 M
- [x] Linear head 768→768 initialised to identity, shared by both towers
- [x] Cache frozen embeddings to speed up training
- [x] Tune LR {1e-3, 1e-2} × weight decay {0, 0.01}

### 6.6 Approach A3 — XLM-R encoder, full fine-tuning
- [x] All layers trainable; AdamW + warm-up; fp16 on T4; max length 256 (report truncation rate)
- [x] Grid: LR {1e-5, 2e-5, 3e-5} × weight decay {0, 0.01} (× τ {0.02, 0.05} if time allows)
- [x] Save tuning table → `results/tuning/a3.csv`; select by val MRR@10 only
- [ ] Upload best checkpoint to HF Hub/Drive; link in README §6.4

### 6.6b Approach A4 — PrahokBART dual encoder (Khmer-native pre-training)
- [x] Extract encoder from `nict-astrec-att/prahokbart_base` (35.8M parameters, 512-dim output, SentencePiece 32,004 vocab)
- [x] Siamese dual tower with masked mean pooling and L2 unit normalization (`src/dl/models/prahokbart.py`)
- [x] Config `configs/a4_prahokbart.yaml` and unit tests `tests/unit/test_a4_prahokbart.py` (all passing)
- [x] Hyperparameter tuning runner `src/dl/experiments/tune_a4.py` (LR {5e-5, 1e-4} x weight decay {0.01})
- [x] Full evaluation on Primary T-Q and Secondary T-T benchmarks (`results/metrics/a4_prahokbart.json`)
- [x] Colab interactive reproduction notebook `notebooks/a4_prahokbart_colab.ipynb`

### 6.7 Final evaluation & comparison
- [x] Evaluate each selected model **once** on T-Q and T-T
- [x] `report.py` → `results/metrics/summary.csv` and README §5.1 table (params, time, hardware)
- [x] Figures: metrics bar chart with CIs; overlaid learning curves; Recall@k curve; per-code breakdown
- [x] Discuss over/under-fitting per curve; explain the winner with course concepts

### 6.8 Error analysis & limitations
- [x] Per approach: T-Q misses at k=5 with retrieved vs expected articles → `results/error_analysis/`
- [x] Categorise: vocabulary mismatch, segmentation errors, multi-article, shared titles, Civil/Criminal confusion, near-miss
- [x] 3–5 worked examples for slides; limitations + prioritised future work in README §5.5

---

## Phase 7: Repository Deliverables
- [x] README: every required section filled from `results/` (no hand-typed numbers)
- [x] `requirements.txt` pinned from the final environment
- [x] `results/`, `configs/`, `notebooks/` (one Colab notebook per approach), `slides/`
- [x] Fresh-clone reproducibility test: install → download → extract → chunk → splits → train one approach → evaluate
- [x] Commit regularly with meaningful messages (several times per week)

---

## Phase 8: Slides & Presentation (10–20 slides, 10 min + 5 min Q&A)
- [x] Problem & motivation (Khmer, official text, input → output)
- [x] Dataset & pipeline (ODC PDFs, Limon problem and conversion, article chunking, sizes, splits, leakage guard)
- [x] Architectures & strategies (A1–A3 diagrams, tokenization contrast, justification)
- [x] Experimental setup (metrics, search ranges, optimiser, hardware, runtimes)
- [x] Results (single table, learning curves, metric chart)
- [x] Discussion & error analysis (why the winner won, Khmer failure cases, limitations)
- [x] Conclusion & future work; appendix (tuning table, examples, parameter counts)
- [x] Export to `slides/final_presentation.pdf`; rehearse to ≤ 10 minutes
- [x] Q&A prep: Limon reordering, InfoNCE, in-batch negatives, pooling, freezing, checkpoint/resume, each metric, every line of `src/dl/`

---

## Phase 9: Khmer RAG Prototype
- [x] `TorchEncoderEmbedding` adapter implementing `EmbeddingPort` with the best checkpoint
- [x] Index `*_kh_chunks.json` (dense + Khmer-segmented BM25) and switch the app to the Khmer corpus
- [x] Remove the Khmer→English dictionary query mapping once retrieval is native Khmer
- [x] Re-run app evaluation on T-Q; latency < 5 s target

---

## Phase 10: Stretch / Optional
- [ ] OCR the Khmer Labour Law (Tesseract `khm`) and measure its quality before adding it to the corpus
- [ ] English app data fixes (stale section metadata, missing EN articles 553–556 and 983–986), kept separate from the Khmer study
- [ ] Trained cross-encoder re-ranker as a fourth approach
