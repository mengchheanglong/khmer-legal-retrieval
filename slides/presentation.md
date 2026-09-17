---
marp: true
theme: default
paginate: true
header: "Khmer Legal Retrieval — Deep Learning Final Project"
footer: "Royal University of Phnom Penh / CADT | Mengchheang Long"
style: |
  section {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-size: 22px;
    padding: 40px;
  }
  h1 {
    color: #1e3a8a;
  }
  h2 {
    color: #2563eb;
    font-size: 28px;
  }
  table {
    font-size: 17px;
  }
  th {
    background-color: #f1f5f9;
    color: #0f172a;
  }
  .highlight {
    background-color: #fef08a;
    padding: 2px 6px;
    border-radius: 4px;
  }
---

# Neural Information Retrieval on the Official Khmer Legal Corpus
## Comparative Study of From-Scratch, Frozen Probe, and Fine-Tuned Dual Encoders

**Author**: Mengchheang Long  
**Course**: Deep Learning Final Project  
**Date**: September 2026  
**Corpus**: Cambodian Civil Code (2007) & Criminal Code (2009) (1,976 Articles)

---

## 1. Problem & Motivation

* **Context**: Legal information retrieval in low-resource languages like Khmer is challenging due to complex morphology, absence of whitespace between words, and limited pre-trained domain models.
* **The Goal**: Given a natural language legal question in Khmer, retrieve the exact statutory article(s) (e.g. *Article 315 of Civil Code* or *Article 336 of Criminal Code*) from the official legal corpus.
* **Input $\to$ Output**:
  * **Input (Khmer Query)**: *"តើកិច្ចសន្យាបង្កើតឡើងដោយរបៀបណា?"* (How is a contract formed?)
  * **Output (Statutory Grounding)**: Civil Code Article 315 (*"ការបង្កើតកិច្ចសន្យា"*) with rank $\le 5$.
* **Why Deep Learning?**
  * Lexical search (BM25) fails when colloquial questions use terms distinct from formal statutory phrasing (*vocabulary mismatch*).
  * Neural dual encoders project queries and statutory passages into a shared continuous semantic space.

---

## 2. The Khmer Text Pipeline & The Limon Legacy Challenge

* **Source of Truth**: Official Khmer PDFs published by Open Development Cambodia (ODC).
* **The Limon Problem**:
  * Official PDFs do not use modern Unicode encoding. Instead, they use legacy 8-bit **Limon fonts** (e.g. `Limon S1`, `Limon R1`).
  * The PDF character stream stores Latin ASCII codes (e.g. `maRta 336>-` displays visually as `មាត្រា ៣៣៦.-`).
* **Conversion & Syllable Reordering Algorithm**:
  * Step 1: 8-bit glyph mapping to Unicode codepoints using XML lookup tables (`resources/khmer_legacy_fontdata.xml`).
  * Step 2: Finite State Machine for Khmer syllable reordering:
    $$\text{Base Consonant} + \text{Subscript (ជើង)} + \text{Pre-Vowel (ស្រៈមុខ)} + \text{Post-Vowel (ស្រៈក្រោយ)}$$
  * Converts visual left-placed vowels (e.g. េ, ែ, ៃ) to logical Unicode order.
* **Extraction Guarantee**: Consecutive article extraction across 1,976 articles (Civil: 1–1,304; Criminal: 1–672) with zero missing sections.

---

## 3. Dataset & Leakage-Guarded Experimental Split

* **Total Legal Corpus**: 1,976 verified statutory articles.
* **Primary Benchmark (T-Q)**:
  * **200 human-verified legal questions** (100 Civil Code, 100 Criminal Code).
  * Authored independently by Cambodian legal researchers to reflect realistic citizen queries.
* **Secondary Benchmark (T-T)**:
  * **148 held-out statutory article titles** evaluated on retrieving their corresponding passages.
* **Zero-Leakage Guarantee**:
  * All 200 benchmark questions and their citing articles are **strictly excluded** from training.
  * Passage titles are stripped during training so the network cannot memorize trivial lexical cues.
  * Training pairs: 1,176 title-to-article pairs from the remaining training corpus.
  * Validation split: 130 pairs used strictly for early stopping and hyperparameter selection.

---

## 4. Evaluated Retrieval Architectures

We benchmark 5 distinct systems representing three paradigm levels:

1. **BM25 Baseline (Sparse Lexical)**:
   * Rank-BM25 with `khmer-nltk` CRF word tokenization ($k_1=1.5, b=0.75$).
2. **Multilingual-E5-Base (Zero-Shot Dense Reference)**:
   * Pretrained 12-layer XLM-RoBERTa backbone evaluated zero-shot without fine-tuning.
3. **Approach A1: BiLSTM Dual Encoder from Scratch (1.97M Params)**:
   * Trainable word embedding layer + 2-layer BiLSTM + mean pooling + cosine similarity.
4. **Approach A2: XLM-RoBERTa Frozen Linear Probe (0.59M Params)**:
   * Frozen XLM-R backbone + trainable $768 \to 768$ projection head initialized to Identity.
5. **Approach A3: XLM-RoBERTa Full Fine-Tuning (278M Params)**:
   * End-to-end backpropagation through all 12 transformer encoder layers.

---

## 5. Architectural Deep-Dive: A1 vs A2 vs A3

| Dimension | A1: BiLSTM (Scratch) | A2: XLM-R (Linear Probe) | A3: XLM-R (Full Fine-Tune) |
|:---|:---|:---|:---|
| **Backbone** | BiLSTM (2 layers, hidden 128/256) | Frozen XLM-RoBERTa-base | Trainable XLM-RoBERTa-base |
| **Tokenization** | Word tokens (`khmer-nltk`) | Subword BPE (250k vocab) | Subword BPE (250k vocab) |
| **Trainable Params**| 1,971,968 (100%) | 590,592 (0.21%) | 278,043,648 (100%) |
| **Passage Repr.** | Computed dynamically | Pre-computed & cached offline | Dynamically backpropagated |
| **Training Time** | ~33 minutes (CPU) | **5.9 seconds** (CPU) | ~5.3 minutes (GPU) |
| **Regularization** | Dropout ($p=0.3$), Weight decay | Frozen backbone acts as prior | AdamW weight decay + warmup |

---

## 6. Training Objective: InfoNCE Loss with In-Batch Negatives

* **Objective Formulation**:
  For a batch of $B$ query-passage pairs $(q_i, p_i)$, all other passages $p_j$ ($j \neq i$) within the mini-batch serve as negatives:
  $$\mathcal{L}_{\text{InfoNCE}} = - \frac{1}{B} \sum_{i=1}^B \log \frac{\exp(\text{sim}(q_i, p_i) / \tau)}{\sum_{j=1}^B \exp(\text{sim}(q_i, p_j) / \tau)}$$
* **Temperature**: $\tau = 0.05$ sharpens the contrastive distribution.
* **Similarity Metric**: Cosine similarity $\text{sim}(u, v) = \frac{u^\top v}{\|u\|_2 \|v\|_2}$ over $L_2$-normalized embeddings.
* **In-Batch Negatives**: Effective negative sample pool size of $B-1$ negatives per query without extra computational forward passes.

---

## 7. Hyperparameter Tuning & Selection Protocol

All models tuned systematically on the validation set (**never on test data**):

* **A1 (BiLSTM)**:
  * Grid: $\text{Hidden Dim} \in \{128, 256\} \times \text{LR} \in \{0.0005, 0.001\}$
  * **Selected Best**: Hidden 128, LR $0.001$, Batch 32 $\to$ Val MRR@10: **0.1706**
* **A2 (XLM-R Linear Probe)**:
  * Grid: $\text{LR} \in \{0.0005, 0.001\} \times \text{Weight Decay} \in \{0.0, 0.01\}$
  * **Selected Best**: LR $0.001$, Weight Decay $0.01$, Batch 32 $\to$ Val MRR@10: **0.3444**
* **A3 (XLM-R Full Fine-Tuning)**:
  * Grid: $\text{LR} \in \{1\times 10^{-5}, 2\times 10^{-5}, 3\times 10^{-5}\} \times \text{Weight Decay} \in \{0.0, 0.01\}$
  * **Selected Best**: LR $2\times 10^{-5}$, Weight Decay $0.01$, Warmup 10%, Batch 16 $\to$ Val MRR@10: **0.4285**

---

## 8. Primary Benchmark Results (T-Q: 200 Questions)

Single test evaluation on the full 1,976-article corpus with 95% bootstrap confidence intervals ($B=1,000$ resamples):

| Rank | Model / Approach | Recall@1 [95% CI] | Recall@5 [95% CI] | Recall@10 [95% CI] | MRR@10 [95% CI] | Hit@5 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **A3: XLM-R Full Fine-Tune** | **0.3150** [0.26, 0.37] | **0.4942** [0.43, 0.56] | **0.5608** [0.50, 0.63] | **0.4499** [0.39, 0.51] | **55.5%** |
| 🥈 | **A2: XLM-R Linear Probe** | 0.2400 [0.19, 0.30] | 0.4117 [0.35, 0.48] | 0.4542 [0.39, 0.52] | 0.3591 [0.30, 0.42] | 47.5% |
| 🥉 | **BM25 Lexical Baseline** | 0.2025 [0.15, 0.26] | 0.4075 [0.34, 0.47] | 0.4850 [0.42, 0.55] | 0.3350 [0.28, 0.39] | 47.0% |
| 4 | **Multilingual-E5 (Zero-Shot)**| 0.1625 [0.12, 0.21] | 0.2617 [0.20, 0.32] | 0.3208 [0.26, 0.38] | 0.2588 [0.20, 0.31] | 32.0% |
| 5 | **A1: BiLSTM (From Scratch)** | 0.0875 [0.05, 0.13] | 0.2025 [0.16, 0.25] | 0.2950 [0.24, 0.36] | 0.1724 [0.13, 0.22] | 24.5% |

**Key Finding**: A3 beats BM25 by **+34.3% relative MRR@10** ($p < 0.001$). A2 beats BM25 while requiring only 5.9s training time!

---

## 9. Performance Visualizations

![Primary Benchmark Metrics](results/figures/primary_metrics_bar.png)
*Left: Recall@1, Recall@5, MRR@10, and Hit@5 across all 5 models with 95% bootstrap error bars.*

---

## 10. Multi-Rank Progression & Code Breakdown

* **Recall Progression**:
  * A3 maintains the highest recall at every cutoff: $k=1$ (31.5%), $k=5$ (49.4%), and $k=10$ (56.1%).
  * Sparse BM25 narrows the gap at $k=10$ (48.5%) because larger cutoff sizes recover keyword matches.
* **Civil Code vs Criminal Code**:
  * Civil Code (100 Qs): High technicality, contractual terminology $\to$ Dense models excel.
  * Criminal Code (100 Qs): Discrete crime names $\to$ BM25 matches specific penalties effectively, but A3 still wins.

![Recall at k Progression](results/figures/recall_at_k.png)

---

## 11. Learning Dynamics & Overfitting Analysis

![Learning Curves](results/figures/learning_curves.png)

* **Why A1 Failed (Severe Overfitting)**:
  * With only 1,176 training pairs, A1's training loss plunged to **0.017**, while validation loss stalled at **0.86**.
  * From-scratch networks cannot learn generalizable Khmer semantic geometry from tiny datasets.
* **Why A2 Succeeded (Regularized Probe)**:
  * Freezing the 278M transformer weights prevents catastrophic forgetting; the 0.59M linear head acts as a subspace projection.
* **Why A3 Won (Optimal Transfer Learning)**:
  * Full fine-tuning adapts high-level attention heads to legal synonymy without overfitting when warmed up.

---

## 12. Secondary Benchmark Results (T-T: 148 Titles)

Evaluating retrieval of statutory passages given their exact article titles:

| Model / Approach | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **A3: Full Fine-Tune** | **0.9189** | **0.9865** | **0.9932** | **0.9467** | **0.9583** |
| **BM25 Baseline** | 0.7973 | 0.9730 | 0.9797 | 0.8677 | 0.8957 |
| **A2: Linear Probe** | 0.7905 | 0.9392 | 0.9595 | 0.8563 | 0.8819 |
| **A1: BiLSTM (Scratch)** | 0.5135 | 0.7568 | 0.8514 | 0.6242 | 0.6788 |
| **E5-Base Zero-Shot** | 0.5405 | 0.7230 | 0.7770 | 0.6235 | 0.6609 |

* Takeaway: When queries share lexical terms with the passage title, BM25 is strong (0.8677 MRR), but A3 fine-tuning achieves near-perfect retrieval (**0.9467 MRR@10**, 98.6% Hit@5).

---

## 13. Error Analysis Taxonomy ($k=5$ Failure Cases)

We audited all retrieval misses at $k=5$ across the 200 primary benchmark queries:

| Error Category | BM25 (106 misses) | A1 (151 misses) | A2 (105 misses) | A3 (89 misses) | Nature of Failure |
|:---|:---:|:---:|:---:|:---:|:---|
| **Vocabulary Mismatch** | **61 (57.5%)** | 79 (52.3%) | 52 (49.5%) | **41 (46.1%)** | Question uses everyday words not present in statute |
| **Code Confusion** | 4 (3.8%) | 13 (8.6%) | 7 (6.7%) | 5 (5.6%) | Civil query retrieves Criminal Code article (or vice-versa) |
| **Multi-Article** | 26 (24.5%) | 37 (24.5%) | 29 (27.6%) | 28 (31.5%) | Query spans cross-cutting concepts (e.g. lease + breach) |
| **Near-Miss (Rank 6–10)**| 15 (14.2%) | 19 (12.6%) | 17 (16.2%) | 15 (16.9%) | Relevant article ranked just outside top-5 |

**Key Takeaway**: A3 reduces *vocabulary mismatch* errors from 61 to 41 (-32.8%), demonstrating that neural representations successfully map colloquial Khmer to statutory language.

---

## 14. Qualitative Worked Examples

### Case 1: Overcoming Vocabulary Mismatch (Criminal Code Art. 253 — Kidnapping)
* **Query**: *"តើការចាប់មនុស្សបង្ខាំងទុកដោយខុសច្បាប់មានទោសអ្វី?"* (What is the punishment for unlawfully capturing and confining a person?)
* **BM25 Rank**: **Miss (> 50)** $\to$ BM25 searches for "ចាប់" and fails to match legal codification.
* **A3 Rank**: **Rank 1 (Hit)** $\to$ Maps colloquial phrase directly to Article 253 (*"ការចាប់ ការឃុំឃាំង និងការបង្ខាំងមនុស្សដោយខុសច្បាប់"*).

### Case 2: Multi-Article Contractual Complexity (Civil Code Art. 384)
* **Query**: *"ប្រសិនបើអ្នកលក់មិនព្រមប្រគល់ទំនិញ តើអ្នកទិញអាចទាមទារអ្វីបាន?"* (If a seller refuses to deliver goods, what can the buyer claim?)
* **Retrieved**: Article 387 (Right to demand performance) and Article 398 (Damages for delay).
* **Expected**: Article 384 (Default of obligor). Both are legally valid; highlights the need for multi-hop retrieval!

---

## 15. The Full Khmer RAG Legal Assistant Pipeline

```
Citizen Query: "តើកិច្ចសន្យាបង្កើតឡើងដោយរបៀបណា?"
                                 ↓
           A3 Fine-Tuned Dense Retriever (TorchEncoderEmbedding)
                                 ↓
         Top-k Khmer Statutory Passages (e.g. Civil Code Art. 315)
                                 ↓
   DeepSeek-V3 LLM Prompt: [Statutory Context + Question + Strict Citation Guard]
                                 ↓
   Answer: "យោងតាមមាត្រា ៣១៥ នៃក្រមរដ្ឋប្បវេណី កិច្ចសន្យាត្រូវបានបង្កើតឡើង..."
                                 ↓
     Citation Verification Layer: checks statute name & article number against context
```

* **Production Speed**: End-to-end retrieval in **< 120 ms** on CPU.
* **Truthfulness**: Citations verified against retrieved statutory chunks before presentation.

---

## 16. Limitations & Prioritized Future Work

1. **Corpus Scope**:
   * Current corpus covers the Civil Code (2007) and Criminal Code (2009).
   * Khmer Labour Law (1997) exists only as scanned image PDFs and requires specialized Khmer Tesseract OCR before inclusion.
2. **Synthetic Data Augmentation**:
   * Training data is currently limited to 1,176 title-to-article pairs.
   * Synthesizing multi-turn citizen queries via LLM prompt inversion will further improve dense domain adaptation.
3. **Two-Stage Re-Ranking**:
   * Integrating a cross-encoder (`BAAI/bge-reranker-large`) after A3 candidate retrieval to further suppress near-misses.

---

## 17. Conclusions & Takeaways

1. **Pre-trained Multilingual Knowledge is Essential**:
   * Training from scratch (BiLSTM) completely fails on low-resource legal domains (0.1724 MRR@10) due to vocabulary sparsity.
2. **The Accuracy vs Cost Frontier**:
   * **A2 (Linear Probe)**: Trains in **5.9 seconds** on CPU, requires only 0.59M parameters, and outperforms BM25. Perfect for ultra-low compute constraints.
   * **A3 (Full Fine-Tuning)**: The undisputed champion, achieving **0.4499 MRR@10** (+34.3% over BM25) and **55.5% Hit@5**.
3. **Rigorous Methodology**:
   * Deterministic seeding, zero test leakage, bootstrap confidence intervals, and 100% reproducible open-source pipeline.

---

## 18. Appendix & Q&A Reference

* **Why Limon Syllable Reordering is Necessary**:
  * Limon displays vowels visually left of consonants (e.g. `e` before `k`), but Khmer Unicode standard mandates logical encoding: base consonant first, followed by vowel.
* **InfoNCE In-Batch Negatives Efficiency**:
  * Cross-entropy over similarity matrix eliminates the need for expensive explicit negative mining during initial convergence.
* **Hardware & Runtime Summary**:
  * Baselines & A2: Local CPU (< 10 seconds total).
  * A1: Local CPU (2,035 seconds, 10 epochs).
  * A3: Google Colab T4 GPU (319.8 seconds, 5 epochs with fp16).
* **Codebase & Artifacts**:
  * GitHub: `https://github.com/mengchheanglong/khmer-legal-retrieval`
  * Checkpoints, evaluation logs, and raw metrics committed in `results/`.
