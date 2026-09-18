---
marp: true
theme: default
paginate: true
header: "Khmer Legal Retrieval — Deep Learning Final Project (5-Minute Defense)"
footer: "Royal University of Phnom Penh / CADT | Mengchheang Long"
style: |
  section {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-size: 20px;
    padding: 36px;
  }
  h1 {
    color: #1e3a8a;
    font-size: 28px;
  }
  h2 {
    color: #2563eb;
    font-size: 24px;
    margin-bottom: 12px;
  }
  table {
    font-size: 15px;
  }
  th {
    background-color: #f1f5f9;
    color: #0f172a;
  }
  .highlight {
    background-color: #dbeafe;
    padding: 2px 6px;
    border-radius: 4px;
  }
  .time-badge {
    background: #f1f5f9;
    color: #334155;
    border: 1px solid #cbd5e1;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 14px;
    font-weight: bold;
  }
---

# Neural Information Retrieval on the Official Khmer Legal Corpus
## A Comparative Study: From-Scratch BiLSTM vs Frozen Linear Probe vs XLM-R vs PrahokBART Fine-Tuning
**Deep Learning Course Final Project | 5-Minute Defense**

* **Author**: Mengchheang Long
* **Corpus**: Cambodian Civil Code (2007) & Criminal Code (2009) — 1,976 Articles
* **Scale**: 1,976 Articles | 200 Human Legal Questions | 148 Held-Out Titles
* **Core Takeaway**: Full fine-tuning of XLM-R (**A3**) is the decisive champion (**0.4827 MRR@10**, **51.42% Recall@5**, +44.1% over BM25, $p=0.0001$). A frozen linear probe (**A2**) matches BM25 in only 5.9s on CPU. Native Khmer PrahokBART (**A4**) excels at formal titles (80.4% Hit@5) but drops on conversational queries.

---
<!--
SPEAKER NOTES [0:00 - 0:25] (25 seconds):
"Good morning, Professor. Today I present our Deep Learning final project: Neural Information Retrieval on the Official Khmer Legal Corpus.
Our research question investigates whether multilingual transfer learning outperforms from-scratch recurrent networks and compact monolingual models when retrieving Cambodian law for conversational citizen queries.
Our primary finding: full fine-tuning of XLM-RoBERTa achieves state-of-the-art retrieval with 0.4827 MRR@10, outperforming BM25 by 44% with statistically confirmed significance at p=0.0001."
-->

## 1. Problem Formulation & The Limon Font Challenge <span class="time-badge">⏱️ 0:25 - 0:55 (30s)</span>

### The Real-World Citizen Challenge
* **The Problem**: Cambodian citizens struggle to find statutory protections because laws are long, unspaced, and written in formal legal Khmer.
* **Task**: Given a conversational Khmer query, retrieve the exact statutory article ($k \le 5$) from the full 1,976-article corpus without statute filtering.
* **Example**: *"តើកិច្ចសន្យាបង្កើតឡើងដោយរបៀបណា?"* $\to$ Civil Code Article 315 (*"ការបង្កើតកិច្ចសន្យា"*).
* **BM25 Failure**: Suffers **57.5% vocabulary mismatch** on conversational citizen queries.

### The Limon Legacy Font Extraction Challenge
* **The Root Cause**: Official ODC PDFs use legacy 8-bit **Limon fonts** (`Limon S1`, `Limon R1`), storing ASCII keystrokes (`maRta 336>-` $\to$ មាត្រា ៣៣៦.-). Standard extractors produce total gibberish.
* **Finite State Machine (FSM) Syllable Reordering**: Reorders pre-base vowels (េ, ែ, ៃ) into canonical Unicode consonant-first order.
* **Integrity Guarantee**: 100% consecutive extraction across **1,976 articles** (Civil 1–1304, Criminal 1–672).

---
<!--
SPEAKER NOTES [0:25 - 0:55] (30 seconds):
"Access to justice in Cambodia is hindered because citizens ask questions in everyday conversational language, whereas the law is written in formal statutory terms. BM25 suffers a 57.5% vocabulary mismatch failure rate.
Compounding this, official government PDFs do not store Unicode; they use legacy 8-bit Limon fonts storing raw Latin keystrokes.
We engineered a lossless Finite State Machine that converts glyphs and reorders visual vowels into canonical Unicode sequence, guaranteeing 100% consecutive extraction across all 1,976 statutory articles."
-->

## 2. Legal Corpus & Strict Zero-Leakage Protocol <span class="time-badge">⏱️ 0:55 - 1:25 (30s)</span>

### Dual Evaluation Benchmarks
* **Primary Benchmark (T-Q: 200 Questions)**:
  * 100 Civil Code + 100 Criminal Code questions independently authored by Cambodian legal researchers.
  * Evaluated across the entire 1,976-article corpus without statute hints.
* **Secondary Benchmark (T-T: 148 Titles)**:
  * 148 held-out statutory titles evaluating in-distribution passage retrieval.

### Four Strict Leakage Guards (Course Mandate)
1. **Question Exclusion**: All 200 benchmark questions and their 261 citing articles are **strictly excluded** from training.
2. **Title Stripping**: Passage titles are stripped during training so models cannot exploit trivial lexical memorization.
3. **Fixed Splits**: 1,176 training pairs, 130 validation pairs (seed 42). Model selection is guided strictly by Val MRR@10.
4. **Single Evaluation Run**: Test sets are evaluated strictly once to eliminate multiple-testing bias.

---
<!--
SPEAKER NOTES [0:55 - 1:25] (30 seconds):
"To evaluate rigorously, we constructed dual benchmarks: a Primary Benchmark of 200 human-verified questions written by Cambodian legal researchers, and a Secondary Benchmark of 148 held-out article titles.
Following course integrity rules, we enforce four strict leakage guards: all 200 benchmark questions and their 261 citing articles are quarantined from training; passage titles are stripped; splits are locked with fixed seed 42; and test sets are evaluated strictly once after model selection."
-->

## 3. Four Deep Learning Approaches + Baseline <span class="time-badge">⏱️ 1:25 - 2:05 (40s)</span>

| Dimension | BM25 (Sparse) | A1: BiLSTM (Scratch) | A2: XLM-R (Linear Probe) | A3: XLM-R (Fine-Tune) | A4: PrahokBART (Native) |
|:---|:---|:---|:---|:---|:---|
| **Backbone** | Lexical TF-IDF | 2-layer BiLSTM | Frozen XLM-R Base | Trainable XLM-R Base | Khmer BART Encoder |
| **Tokenization**| `khmer-nltk` CRF | `khmer-nltk` (2,544 vocab)| SentencePiece (250k) | SentencePiece (250k) | SentencePiece (32k) |
| **Params** | 0 | 1.97 M (100% trainable)| 0.59 M (0.21% trainable)| 278.04 M (100% trainable)| 35.83 M (~8x smaller) |
| **Training Time**| 0 s | 2,035.3 s (CPU) | **5.9 s** (CPU) | 7,237.5 s (CPU) / 320 s (T4)| 1,602.2 s (CPU) |
| **Strategy** | Word overlap | Random embeddings | Subspace projection | End-to-end backprop | Native Khmer pre-train |

* **Tokenization Contrast**: CRF word segmentation (`khmer-nltk`) for A1 & BM25 vs Subword SentencePiece for A2, A3, and A4.

---
<!--
SPEAKER NOTES [1:25 - 2:05] (40 seconds):
"We benchmark five systems across three distinct paradigms:
First, sparse BM25 with CRF word segmentation.
Approach A1: a from-scratch BiLSTM dual encoder with 1.97M trainable parameters.
Approach A2: a frozen XLM-RoBERTa backbone with a 768-to-768 linear probe, training only 0.59M parameters in an astonishing 5.9 seconds on CPU.
Approach A3: full end-to-end fine-tuning across all 278M parameters.
Approach A4: native Khmer PrahokBART, testing whether a compact 35.8M parameter model pre-trained solely on Khmer can rival large multilingual transformers."
-->

## 4. Contrastive Objective & Learning Dynamics <span class="time-badge">⏱️ 2:05 - 2:35 (30s)</span>

### InfoNCE Loss with In-Batch Negatives
$$\mathcal{L}_{\text{InfoNCE}} = - \frac{1}{B} \sum_{i=1}^B \log \frac{\exp(\text{sim}(q_i, p_i) / \tau)}{\sum_{j=1}^B \exp(\text{sim}(q_i, p_j) / \tau)}$$

* **Temperature $\tau = 0.05$**: Fixed sharpening parameter pushing representations onto compact angular clusters.
* **Optimization**: AdamW (weight decay $0.01$) + 10% Linear Warmup + Cosine Annealing decay ($B=16$).

### Overfitting & Dynamics Analysis
* **A1 (BiLSTM) Severe Overfitting**: Training loss plunged to **0.017**, while validation loss stalled at **0.86**. From-scratch models cannot learn semantic manifolds from 1,176 training pairs without pre-training priors.
* **A2 Stability**: The frozen backbone acts as an invariant semantic prior; overfitting is structurally impossible.
* **A3 Convergence**: Warmup and cosine decay allow deep attention heads to adapt without catastrophic forgetting.

---
<!--
SPEAKER NOTES [2:05 - 2:35] (30 seconds):
"All neural models are trained using InfoNCE loss with in-batch negatives at temperature tau=0.05, optimized with AdamW, a 10% linear warmup, and cosine annealing decay.
Inspecting our learning curves reveals crucial course insights:
Approach A1 severely overfits—its training loss collapses to 0.017 while validation loss stalls at 0.86. A from-scratch model simply cannot generalize from 1,176 pairs.
In contrast, A2's frozen backbone acts as an unyielding regularizer, and A3's gentle warmup enables smooth adaptation to the legal domain without catastrophic forgetting."
-->

## 5. Primary Benchmark Leaderboard (T-Q: 200 Questions) <span class="time-badge">⏱️ 2:35 - 3:15 (40s)</span>

Evaluated on the full 1,976-article corpus with 95% bootstrap confidence intervals ($B=1,000$):

| Rank | Model / Approach | Type | Trainable Params | Recall@1 [95% CI] | Recall@5 [95% CI] | MRR@10 [95% CI] | Hit@5 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **A3: XLM-R Full Fine-Tune** | Full Fine-Tuning | 278.04 M | **0.3375** [0.28, 0.40] | **0.5142** [0.45, 0.58] | **0.4827** [0.42, 0.55] | **57.0%** |
| 🥈 | **A2: XLM-R Linear Probe** | Frozen + Linear | 0.59 M | 0.2400 [0.19, 0.30] | 0.4117 [0.35, 0.48] | 0.3591 [0.30, 0.42] | 47.5% |
| 🥉 | **BM25 Baseline (Lexical)** | Sparse Lexical | 0 | 0.2025 [0.15, 0.25] | 0.4075 [0.35, 0.47] | 0.3350 [0.28, 0.39] | 47.0% |
| 4 | **Multilingual-E5 (Zero-Shot)**| Pretrained Dense | 0 | 0.1625 [0.12, 0.21] | 0.2617 [0.21, 0.32] | 0.2588 [0.20, 0.31] | 32.0% |
| 5 | **A1: BiLSTM (From Scratch)** | Scratch Dense | 1.97 M | 0.0875 [0.05, 0.13] | 0.2025 [0.16, 0.25] | 0.1724 [0.13, 0.22] | 24.5% |
| 6 | **A4: PrahokBART (Khmer)** | Khmer Pretrained | 35.83 M | 0.0525 [0.03, 0.09] | 0.1517 [0.11, 0.20] | 0.1035 [0.07, 0.14] | 18.0% |

* **Undisputed Champion (A3)**: Beats BM25 by **+44.1% relative MRR@10** ($p=0.0001$) and Zero-Shot E5 by **+86.5%**.
* **Linear Probe Feat (A2)**: Matches BM25 (0.3591 vs 0.3350 MRR) with only 0.59M parameters trained in **5.9 seconds**!

---
<!--
SPEAKER NOTES [2:35 - 3:15] (40 seconds):
"Here is the primary benchmark leaderboard across the 200 human-verified legal questions.
Approach A3 is the undisputed champion, achieving 51.42% Recall@5 and 0.4827 MRR@10. This is a +44.1% relative improvement over BM25 and an 86% improvement over zero-shot E5.
Notice Approach A2: with only 0.59 million weights trained in under 6 seconds, it matches BM25 parity, proving that pre-trained multilingual embeddings contain high-quality latent legal knowledge.
Zero-shot E5 underperforms BM25, demonstrating that fine-tuning is strictly required."
-->

## 6. Statistical Significance & Performance Visualizations <span class="time-badge">⏱️ 3:15 - 3:45 (30s)</span>

### Paired Bootstrap Hypothesis Testing (10,000 Resamples)
| Comparison | Metric | Mean Diff ($\Delta$) [95% CI] | $p$-value | Significance | Wins / Losses / Ties |
|:---|:---:|:---:|:---:|:---:|:---:|
| **A3 (Fine-Tuning) vs BM25** | Recall@5 | **+0.1067** [+0.0500, +0.1633] | **0.0002** | *** ($p < 0.001$) | 41W / 13L / 146T |
| **A3 (Fine-Tuning) vs BM25** | MRR@10 | **+0.1477** [+0.0962, +0.1993] | **0.0001** | *** ($p < 0.001$) | 72W / 25L / 103T |
| **A3 (Fine-Tuning) vs E5 Zero-Shot**| MRR@10 | **+0.2239** [+0.1683, +0.2804] | **0.0001** | *** ($p < 0.001$) | 80W / 15L / 105T |
| **A3 (Fine-Tuning) vs A4 (BART)** | MRR@10 | **+0.3792** [+0.3195, +0.4396] | **0.0001** | *** ($p < 0.001$) | 116W / 6L / 78T |
| **A2 (Probe) vs BM25 Baseline** | Recall@5 | **+0.0042** [-0.0567, +0.0625] | **0.9031** | n.s. (Parity) | 26W / 25L / 149T |

* **Scientific Verification**: Paired testing eliminates query variance. A3 beats BM25 on **72 queries** vs 25 losses ($p=0.0001$).
* **Linear Probe Parity**: The $p$-value of 0.9031 statistically confirms A2 and BM25 achieve parity.

---
<!--
SPEAKER NOTES [3:15 - 3:45] (30 seconds):
"Because independent confidence intervals can overlap, we conducted rigorous paired bootstrap hypothesis testing with 10,000 resamples.
The difference between A3 and BM25 is statistically significant at p=0.0001, with A3 winning on 72 queries and losing on only 25.
Similarly, A3 decisively outperforms PrahokBART (p=0.0001) and zero-shot E5.
Finally, the paired test between A2 and BM25 yields a p-value of 0.9031, formally confirming that an ultra-lightweight linear probe achieves true parity with BM25."
-->

## 7. Distribution Shift: In-Distribution (T-T) vs Conversational (T-Q) <span class="time-badge">⏱️ 3:45 - 4:20 (35s)</span>

### The Generalization Gap (Recall@5)
| Model / Approach | In-Distribution (T-T: 148 Titles) | Conversational (T-Q: 200 Questions) | Distribution Drop |
|:---|:---:|:---:|:---:|
| **A3: XLM-R Full Fine-Tune** | **99.3%** | **51.4%** | -47.9% |
| **BM25 Baseline (Lexical)** | 97.3% | 40.8% | -56.5% |
| **A2: XLM-R Linear Probe** | 93.9% | 41.2% | -52.7% |
| **A4: PrahokBART (Khmer Native)** | 80.4% | 15.2% | -65.2% |
| **A1: BiLSTM (From Scratch)** | 75.7% | 20.3% | -55.4% |
| **Multilingual-E5 (Zero-Shot)** | 72.3% | 26.2% | -46.1% |

### Why Did PrahokBART (A4) Struggle on Conversational Queries?
1. **Pre-training Task Mismatch**: PrahokBART was pre-trained for seq2seq text reconstruction (denoising), not sentence-level contrastive representation.
2. **Capacity Gap**: 35.8M parameters / 512-dim vs XLM-R's 278M parameters / 768-dim with rich multilingual transfer.
3. **The T-T Proof**: On formal titles (T-T), PrahokBART achieves a strong **80.4% Hit@5**, beating BiLSTM (75.7%) and zero-shot E5 (72.3%), proving strong Khmer lexical representation, but struggles with colloquial abstraction.

---
<!--
SPEAKER NOTES [3:45 - 4:20] (35 seconds):
"A core course takeaway is the danger of evaluating only in-distribution. On held-out statutory titles (T-T), all models appear near-perfect: A3 hits 99.3% and BM25 hits 97.3%.
However, on realistic conversational questions (T-Q), performance drops by roughly 50%.
This explains why native PrahokBART scored 80.4% on titles—beating scratch BiLSTM and zero-shot E5—yet dropped to 15.2% on conversational queries. Its seq2seq denoising pre-training and 35M capacity lacked the semantic abstraction power of XLM-RoBERTa's 278M multilingual foundation."
-->

## 8. Error Analysis & Qualitative Case Studies <span class="time-badge">⏱️ 4:20 - 4:45 (25s)</span>

### Failure Breakdown at $k=5$ (200 Questions)
* **Vocabulary Mismatch**: BM25 misses 61 queries; A3 reduces this to 52 (-14.8% error count), bridging conversational phrasing.
* **Code Confusion (Civil vs Criminal)**: BiLSTM confuses Civil and Criminal statutes 13 times (8.6%). A3 makes only **1 single confusion** across all 200 queries (0.5% rate)!

### Worked Case Studies
* **Case 1: Overcoming Vocabulary Mismatch (Criminal Code Art. 253 — Kidnapping)**:
  * *Query*: *"តើការចាប់មនុស្សបង្ខាំងទុកដោយខុសច្បាប់មានទោសអ្វី?"* (Unlawful kidnapping/detention)
  * *BM25*: **Miss (> Rank 50)** $\to$ lexical search fails on informal "ចាប់មនុស្ស".
  * *A3*: **Rank 1 (Hit)** $\to$ semantic projection aligns directly to Article 253 (*"ការចាប់ ការឃុំឃាំង និងការបង្ខាំងមនុស្សដោយខុសច្បាប់"*).
* **Case 2: Multi-Article Contractual Complexity (Civil Code Art. 384)**:
  * Query on seller default retrieves Art. 387 (Demand performance) and Art. 398 (Damages). Both are legally valid remedies!

---
<!--
SPEAKER NOTES [4:20 - 4:45] (25 seconds):
"Our error taxonomy reveals that A3 nearly eliminates Code Confusion, dropping cross-code errors to a single query out of 200.
In our qualitative case study on unlawful detention, colloquial phrasing caused BM25 to miss past rank 50, whereas A3 immediately retrieved the exact kidnapping statute at Rank 1.
Cases where A3 missed at k=5 were primarily multi-article queries where the query legitimately touched multiple valid legal remedies."
-->

## 9. Khmer RAG Assistant Prototype & Key Takeaways <span class="time-badge">⏱️ 4:45 - 5:00 (15s)</span>

### Full System Integration & Latency
* **FastAPI + Streamlit Prototype**: Powered by our best A3 checkpoint.
* **Retrieval Latency**: **< 85 ms** on CPU; end-to-end citation-verified answer in **1.5 seconds**.
* **Citation Verification Guard**: Automated parser validates statutory citations against retrieved chunks before display.

### Three Key Conclusions
1. **Pre-Training is Mandatory**: From-scratch BiLSTM fails completely (0.1724 MRR) due to vocabulary sparsity.
2. **Linear Probing Maximizes Efficiency**: A2 achieves BM25 parity with only 0.59M weights trained in **5.9 seconds**.
3. **Fine-Tuning Bridges the Phrasing Gap**: A3 achieves state-of-the-art accuracy (**0.4827 MRR@10**, 57% Hit@5, $p=0.0001$), successfully delivering an accessible Khmer legal assistant.

---
<!--
SPEAKER NOTES [4:45 - 5:00] (15 seconds):
"We integrated our winning A3 retriever into a production Khmer RAG assistant with FastAPI and Streamlit. Retrieval executes in under 85 milliseconds, followed by strict citation verification.
In conclusion: pre-training is mandatory in low-resource legal domains, linear probing offers remarkable efficiency, and full fine-tuning successfully empowers citizens to navigate Cambodian law.
Thank you, and I welcome your questions."
-->

---

## 10. Appendix A: Secondary Benchmark & Resource Footprint <span class="time-badge">⏱️ Q&A Backup</span>

### Secondary Benchmark Results (T-T: 148 Held-Out Titles)
| Model / Approach | Type | Recall@1 [95% CI] | Recall@5 [95% CI] | MRR@10 [95% CI] | Hit@5 [95% CI] |
|:---|:---|:---:|:---:|:---:|:---:|
| **A3: XLM-R Full Fine-Tune** | Full Fine-Tuning | **0.9054** [0.85, 0.95] | **0.9932** [0.97, 1.00] | **0.9456** [0.92, 0.97] | **99.3%** |
| **BM25 Baseline (Lexical)** | Sparse Lexical | **0.7973** [0.74, 0.86] | **0.9730** [0.95, 0.99] | **0.8677** [0.82, 0.91] | **97.3%** |
| **A2: XLM-R Linear Probe** | Frozen + Linear | **0.7905** [0.72, 0.85] | **0.9392** [0.90, 0.97] | **0.8563** [0.81, 0.90] | **93.9%** |
| **A4: PrahokBART (Khmer Native)**| Khmer Pretrained | **0.5946** [0.51, 0.68] | **0.8041** [0.74, 0.87] | **0.6733** [0.60, 0.74] | **80.4%** |
| **A1: BiLSTM (From Scratch)** | Scratch Dense | **0.5135** [0.43, 0.60] | **0.7568** [0.70, 0.82] | **0.6242** [0.56, 0.69] | **75.7%** |
| **Multilingual-E5 (Zero-Shot)** | Pretrained Dense | **0.5405** [0.46, 0.61] | **0.7230** [0.65, 0.78] | **0.6235** [0.55, 0.69] | **72.3%** |

### Computational Efficiency & Resource Footprint
| Approach / Model | Total Params | Trainable Params | Training Wall-Clock Time | Training Hardware |
|:---|:---:|:---:|:---:|:---|
| **BM25 Baseline** | 0 | 0 | 0 s | Multi-core CPU |
| **Multilingual-E5 (Zero-Shot)** | 278.04 M | 0 | 0 s | Multi-core CPU |
| **A1: BiLSTM Dual Encoder** | 1.97 M | 1.97 M (100%) | 2,035.3 s (33.9 min) | Multi-core CPU |
| **A2: XLM-R Linear Probe** | 278.63 M | **0.59 M** (0.21%) | **5.9 s** | Multi-core CPU |
| **A3: XLM-R Full Fine-Tuning** | 278.04 M | 278.04 M (100%) | 7,237.5 s (CPU) / 319.8 s (T4) | Multi-core CPU / T4 GPU |
| **A4: PrahokBART Dual Encoder** | 35.83 M | 35.83 M (100%) | 1,602.2 s (26.7 min) | Multi-core CPU |

---

## 11. Appendix B: Hyperparameter Tuning Grids & Code Architecture <span class="time-badge">⏱️ Q&A Backup</span>

### Systematic 2×2 Hyperparameter Tuning Grids
| Model | Grid Search Hyperparameters | Best Configuration Selected | Val MRR@10 | Val Recall@5 |
|:---|:---|:---|:---:|:---:|
| **A1 (BiLSTM)** | $\text{Hidden} \in \{128, 256\} \times \text{LR} \in \{1\text{e-}3, 3\text{e-}3\} \times \text{Drop} \in \{0.1, 0.3\}$ | Hidden 128, LR $1\times 10^{-3}$, Drop 0.3 | 0.1706 | 0.2055 |
| **A2 (Probe)** | $\text{LR} \in \{1\text{e-}3, 1\text{e-}2\} \times \text{WD} \in \{0.0, 0.01\}$ | LR $1\times 10^{-3}$, Weight Decay $0.01$ | 0.3444 | 0.4041 |
| **A3 (Fine-Tune)**| $\text{LR} \in \{1\text{e-}5, 2\text{e-}5\} \times \text{WD} \in \{0.0, 0.01\}$ | LR $2\times 10^{-5}$, Weight Decay $0.01$ | **0.4285** | **0.4932** |
| **A4 (BART)** | $\text{LR} \in \{5\text{e-}5, 1\text{e-}4\} \times \text{WD} \in \{0.01\}$ | LR $5\times 10^{-5}$, Weight Decay $0.01$ | 0.2878 | 0.3699 |

### Technical Defense Topics
* **Limon FSM Syllable Reordering**: Custom FSM reorders visual pre-base vowels (េ, ែ, ៃ) to logical Unicode codepoint order, normalising zero-gap space glitches (*ផ ន្ត្ទាទោស* $\to$ *ផ្តន្ទាទោស*).
* **InfoNCE In-Batch Negatives**: With batch size $B=16$, every pair receives 15 dynamic negative passages without extra encoder forward passes.
* **Reproducibility**: Deterministic seed 42 set across Python, NumPy, PyTorch, cuDNN. Checkpoint saving/resuming (`last.pt`, `best.pt`) with full RNG states.
