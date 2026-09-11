# AGENTS.md — RAG Cambodia Law

## Project Overview
This repository is an **individual Deep Learning final project** on **Khmer** legal retrieval. It
compares three PyTorch retrievers trained on the official Khmer Civil Code (2007) and Criminal Code
(2009):
- **A1:** BiLSTM trained from scratch
- **A2:** XLM-R encoder, frozen, with a linear probe
- **A3:** XLM-R encoder, fully fine-tuned

The best model then powers a citation-grounded Khmer **RAG** legal assistant (FastAPI + Streamlit).

- Requirements and grading: `docs/GOAL.md` (goal, rubric mapping) and `docs/TASKS.md` (compliance audit + tasks)
- The DL comparison is the graded core; the RAG app is the prototype.

## Tech Stack
- **Language**: Python 3.11+
- **Deep learning**: **PyTorch only** (+ Hugging Face `transformers`). No Keras/TensorFlow.
- **Pre-trained backbone**: `intfloat/multilingual-e5-base` (XLM-RoBERTa-base architecture)
- **Khmer text**: Limon→Unicode converter (`src/infrastructure/extractors/limon_converter.py`), `khmer-nltk` word segmentation
- **PDF extraction**: PyMuPDF (fitz)
- **Sparse retrieval (baseline)**: BM25 (rank-bm25)
- **App**: FastAPI, Streamlit, DeepSeek `deepseek-chat`, PostgreSQL + pgvector, `BAAI/bge-reranker-large`

## Khmer Corpus Rules
1. **Source:** official Khmer PDFs from Open Development Cambodia (CC-BY-SA-4.0) are the source of truth. English texts are unofficial translations and are not part of the study.
2. **Extraction:** Khmer PDFs use legacy **Limon** fonts. Always extract with `LimonPdfExtractor`, never raw `get_text()`. Keep page ranges and stop markers in `src/pipeline/extract.py`.
3. **Articles:** one article per chunk (`មាត្រា N.- title`). Numbering must stay consecutive (Civil 1–1304, Criminal 1–672). Re-validate after any pipeline change.
4. **Attribution:** keep KhmerConverter attribution and licences in `src/infrastructure/extractors/resources/`.
5. **Scanned PDFs** (e.g. Labour Law) need OCR and are out of scope until their quality is measured.

## Deep Learning Experiment Rules (from the course instruction)
1. **Distinct approaches** differ in architecture and/or training strategy. Changing only LR, batch size, epochs, seed, loss, optimiser or augmentation is tuning, not a new approach.
2. **Same data for all:** one fixed split in `data/05_splits/`, the same 1,976-article corpus, the same test-time preprocessing, one shared evaluation harness (`src/dl/evaluate.py`).
3. **No leakage:**
   - Never train or tune on test questions or their relevant articles.
   - Strip the title line from training passages.
   - Select models on validation only and evaluate test sets once.
   - Use no statute filter at test time.
4. **Reproducibility:** call `set_seed(42)` (Python, NumPy, PyTorch, CUDA) at the start of every script.
5. **Checkpointing:** `torch.save` model/optimiser/scheduler/epoch/RNG each epoch (`last.pt`, `best.pt`), with `--resume` support.
6. **Log everything:** per-epoch train/val curves, trainable params, training time and hardware → `results/`.
7. **Never fabricate or hand-edit results.** README tables and figures are generated from `results/` files.
8. **Cite** every pre-trained model, dataset and reused code in the README.
9. **Weights over 50 MB** are not committed; host externally and link from the README.

## Architecture Principles
1. **Clean separation of concerns**: domain / application / infrastructure / interfaces; pipeline stages are independent modules.
2. **Reproducibility**: every step from raw PDF to trained model is scripted and repeatable.
3. **Article-level chunking**: hierarchy (គន្ថី → មាតិកា → ជំពូក → ផ្នែក → មាត្រា) is kept as metadata.
4. **Strict citation (app)**: every generated answer cites Law and Article, verified against retrieved context.

## Coding Standards
- Use type hints for all function signatures.
- Write Google-style docstrings for public functions and classes. Comment non-obvious logic (Khmer text handling, DL training); the author must be able to explain every line in Q&A.
- Keep functions small and testable (< 50 lines preferred).
- Use `pydantic` models for data passed between app modules and YAML configs for experiments.
- Load environment variables from `.env` files (never commit secrets).
- Make small, frequent commits with meaningful messages (commit history is graded).

## Key Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Khmer corpus pipeline
python -m src.pipeline.download
python -m src.pipeline.extract
python -m src.pipeline.chunk

# Deep learning experiments (planned — see docs/TASKS.md Phase 6)
python -m src.dl.prepare_splits --seed 42
python -m src.dl.train --config configs/a3_xlmr_finetune.yaml [--resume]
python -m src.dl.evaluate --all
python -m src.dl.report

# RAG app
uvicorn src.interfaces.api.main:app --reload
streamlit run streamlit_app.py

# Tests
pytest tests/ -v
```

## Data Flow
```
ODC Khmer PDFs (Limon fonts) → LimonPdfExtractor (→ Unicode) → Khmer chunker → data/04_chunks/*_kh_chunks.json (1,976 articles)
                                                                                   ↓
                        Title→article pairs + Khmer test questions → Fixed splits → Train A1 / A2 / A3 (PyTorch) → Shared evaluation → results/
                                                                                                          ↓ best model
Khmer question → Trained retriever → Top-k Khmer articles → DeepSeek → Khmer answer with verified citations
```
