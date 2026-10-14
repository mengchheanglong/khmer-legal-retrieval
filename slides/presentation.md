# 10-Minute Oral Defense Master Guide & Speaker Notes

> **Official Course Assessment**: Individual Deep Learning Final Project (50% Course Grade)  
> **Course Guidelines Reference**: Department of Engineering · Bachelor of Software Engineering (§6.2, §6.3, §7.1)  
> **Defense Format**: Strictly **10-minute presentation** (11 Main Slides + Live Demo) followed by **5-minute Q&A** (§6.3).  
> **Deck Artifact**: `slides/presentation.html` (Compiled to 16:9 vector PDF: `slides/final_presentation.pdf`, 15 slides total).  
> **Live Demo URL**: `http://localhost:8501` (Localhost Streamlit UI with Approach A3 PyTorch neural encoder).

---

## ⏱️ Master 10-Minute Time Budget

| Block | Slide / Section | Time Window | Duration | Rubric Focus (§6.2 & §7.1) |
|:---|:---|:---:|:---:|:---|
| **Part 1: Motivation & Data** | **Slide 1**: Title & Research Question | 0:00 – 0:20 | 20s | Problem hook, corpus scale, headline empirical result |
| | **Slide 2**: 1. Problem Definition & Motivation | 0:20 – 1:10 | 50s | Task definition ($q \rightarrow d$), vocabulary mismatch, BM25 failure |
| | **Slide 3**: 2. Dataset & Limon Font Extraction | 1:10 – 2:10 | 60s | Obsolete 1990s Limon ASCII, FSM vowel reordering, 1,976 articles |
| | **Slide 4**: 3. Training/Val/Test Splits & Leakage Guard | 2:10 – 3:00 | 50s | Title-to-body weak pairs, title stripping, 200 real questions quarantine |
| **Part 2: Deep Learning Models** | **Slide 5**: 4. Model Architectures & Strategies | 3:00 – 4:15 | 75s | 4 PyTorch models (A1–A4), scratch vs probe vs full FT, tokenization |
| | **Slide 6**: 5. Training Process & InfoNCE Loss | 4:15 – 5:15 | 60s | Symmetric InfoNCE contrastive math, $\tau=0.05$, in-batch negatives, checkpoints |
| | **Slide 7**: 6. Experimental Setup & Tuning Grid | 5:15 – 6:10 | 55s | Tesla T4 GPU, A3 6-run grid search ($2\times 10^{-5}, wd=0.01$, Val MRR 0.9095) |
| **Part 3: Results & Analysis** | **Slide 8**: 7. Empirical Results & Learning Curves | 6:10 – 7:25 | 75s | Primary T-Q table, over/underfitting curves, capacity vs transfer learning |
| | **Slide 9**: 8. Qualitative Error Analysis & Limitations | 7:25 – 8:20 | 55s | 5 failure categories at $k=5$, in-distribution vs real question shift |
| **Part 4: Prototype & Conclusion**| **Slide 10**: 9. Prototype Demonstration | 8:20 – 9:20 | 60s | Live localhost demo on lexical gap query, verified citation anti-hallucination |
| | **Slide 11**: 10. Conclusions & Future Work | 9:20 – 10:00 | 40s | 3 core takeaways, limitations, edge deployment, transition to Q&A |
| **Part 5: Committee Defense** | **Slides A1–A4 (Reserve Appendix)** | **10:00 – 15:00**| **5 min** | **Data-backed answers to anticipated lecturer questions (Rubric: 70%)** |

---

## 🎙️ Word-for-Word Spoken Defense Script

### Slide 1: Title & Research Question (⏱ 0:00 – 0:20)
> *"Good morning Mr. Soklong and committee members. Today I am presenting my Deep Learning final project: **'Finding the Right Law, in Khmer: Comparing Four Deep-Learning Retrievers on Cambodian Statutory Law'**.*
>
> *Our central research question investigates whether neural bi-encoders can overcome the severe vocabulary gap in legal search to retrieve the exact statutory article out of nearly two thousand Cambodian Civil and Criminal Code provisions when citizens ask questions in everyday conversational Khmer. Our winning fine-tuned neural retriever outperforms standard keyword search by **+44.1% on MRR@10** with paired bootstrap statistical significance ($p < 0.001$)."*

---

### Slide 2: Problem Definition & Motivation (⏱ 0:20 – 1:10)
> *"Let us formally define the task. The input is an unstructured legal question $q$ asked in everyday natural Khmer. The output is a ranked set of statutory articles $\{d_1, \dots, d_k\}$ retrieved from our complete corpus of 1,976 statutory articles, evaluated with zero chapter or statute hints.*
>
> *In Cambodia, citizens cannot navigate the legal system because statutes are written in formal, civil-law terminology. Consider this authentic benchmark question:*
>
> *«តើកិច្ចសន្យាបង្កើតឡើងដោយរបៀបណា?» ('How is a contract formed under the Civil Code?').*
>
> *The ground-truth answers are **Civil Code Article 336** ('Formation of contract by offer and acceptance') and **Article 311** ('Definition of contract').*
>
> *Traditional keyword matching like BM25 fails completely on this query: BM25 matches the colloquial question word «របៀប» ('manner' or 'how') and gets distracted into returning Article 623 ('Manner of dividing proceeds in co-ownership'). In fact, our quantitative error analysis reveals that **57.5% of BM25's retrieval misses are caused purely by this vocabulary mismatch**. Dense neural encoders solve this by projecting semantic meaning into a continuous latent space."*

---

### Slide 3: Dataset & Preprocessing: The Limon Font Challenge (⏱ 1:10 – 2:10)
> *"Before building any deep learning model, we encountered a fundamental engineering hurdle: the official government PDFs from Open Development Cambodia are not encoded in Unicode.*
>
> *They were typeset in the 1990s using legacy **Limon ASCII fonts**. If you extract text with standard tools like PyMuPDF or Tesseract, you get meaningless Latin keystrokes such as `maRta 336>- karbegáItkic©snñ`.*
>
> *Furthermore, Khmer writing features visual grapheme inversion: pre-base vowels like េ, ែ, and ៃ, as well as subscript consonants, are typed visually before the base consonant, whereas logical Unicode encoding requires them after.*
>
> *To solve this without noisy OCR, I engineered `LimonPdfExtractor`, implementing a byte-level glyph mapping and a finite-state machine that reconstructs logical Unicode clusters. This achieved **100% consecutive article extraction across all 1,976 articles** (1,304 Civil Code + 672 Criminal Code articles), audited with zero missing articles, and preserved complete statutory breadcrumb metadata from Book down to Article."*

---

### Slide 4: Training, Validation and Test Splits: Zero-Leakage Protocol (⏱ 2:10 – 3:00)
> *"Because labeled Khmer legal question-answering datasets do not exist, we leveraged the structure of statutory law for weak supervision, generating 1,472 unique article title-to-body pairs. We systematically identified and removed 270 articles sharing ambiguous, identical titles (such as 'Definitions' or 'Competence') to prevent label ambiguity.*
>
> *To satisfy Section 5.A of the course instructions, we enforced an uncompromising data-leakage quarantine:*
>
> 1. *First, we compiled a **Primary Benchmark (T-Q) of 200 authentic human legal questions** created by legal scholars, citing 261 specific articles.*
> 2. *Every single one of these 261 target articles was strictly quarantined so that no training pair was ever generated from them.*
> 3. *The title line was stripped from all training passages, forcing neural encoders to learn conceptual semantics rather than lexical copy-pasting.*
> 4. *The remaining corpus was deterministically split with random seed 42 into 1,176 training pairs, 148 validation pairs, and 148 held-out test title pairs (T-T Secondary Benchmark), with test sets evaluated strictly once."*

---

### Slide 5: Four Neural Architectures & Strategies (⏱ 3:00 – 4:15)
> *"To ensure a rigorous, rubric-compliant study under Section 4, we designed and implemented **four distinct PyTorch deep learning architectures** spanning both architecture families and training strategies:*
>
> 1. * **A1: BiLSTM Dual Encoder from scratch (1.97M trainable parameters)**. Uses 2 bidirectional LSTM layers over segmented Khmer word tokens using `khmer-nltk` CRF word tokenization.*
> 2. * **A2: XLM-RoBERTa Linear Probe (0.59M trainable parameters)**. We freeze all 278M transformer weights of `multilingual-e5-base` and train only a linear projection head. Training takes just 5.9 seconds on CPU.*
> 3. * **A3: XLM-RoBERTa Full Fine-Tuning (278.04M trainable parameters)**. We adapt all 12 transformer encoder layers end-to-end — this is our winning approach.*
> 4. * **A4: PrahokBART Dual Encoder (35.83M trainable parameters)**. A native Khmer sequence-to-sequence model pre-trained on news text, adapted with mean pooling as a dual encoder.*
>
> *We benchmark all four against BM25 keyword search and zero-shot Multilingual-E5."*

---

### Slide 6: Training Process & InfoNCE Contrastive Loss (⏱ 4:15 – 5:15)
> *"All models were trained under the exact same contrastive metric learning formulation using **symmetric InfoNCE loss** with in-batch negatives.*
>
> *For a mini-batch of $B$ query-article pairs, we compute cosine similarities scaled by temperature $\tau = 0.05$. The loss pulls the positive article vector toward the query while repelling the $B-1$ negative articles in the batch symmetrically in both directions.*
>
> *We optimized using AdamW with a 10% linear warmup schedule followed by cosine learning rate decay. Per course requirements in Section 5.B, we implemented full model checkpointing saving optimizer, scheduler, and PyTorch RNG state every epoch to `last.pt` and `best.pt`, with automated `--resume` support."*

---

### Slide 7: Experimental Setup & Hyperparameter Tuning (⏱ 5:15 – 6:10)
> *"To satisfy Section 5.C, we conducted systematic hyperparameter tuning for all models and recorded the complete search grids in `results/tuning/`.*
>
> *For our primary model, **Approach A3**, we executed a full 6-configuration grid search on an NVIDIA Tesla T4 GPU across:*
> - *Learning Rates: $\{1\times 10^{-5}, 2\times 10^{-5}, 3\times 10^{-5}\}$*
> - *Weight Decay: $\{0.0, 0.01\}$*
>
> *As shown in the table on the left, configuration `a3_lr2e-5_wd0.01` emerged as the empirical winner, reaching a **Validation MRR@10 of 0.9095** and validation loss of 0.1032 at epoch 4.*
>
> *Similarly, A1 was tuned across learning rates and dropout $\{0.1, 0.3\}$, A2 across projection learning rates $\{1\times 10^{-3}, 1\times 10^{-2}\}$, and A4 across learning rates $\{5\times 10^{-5}, 1\times 10^{-4}\}$. Global random seed 42 was fixed across Python, NumPy, PyTorch, and cuDNN."*

---

### Slide 8: Empirical Results & Overlaid Learning Curves (⏱ 6:10 – 7:25)
> *"Let us examine the empirical results on the Primary Benchmark (T-Q: 200 real human questions):*
>
> - * **Approach A3 (XLM-R Full Fine-Tune) is the definitive winner**, achieving **51.42% Recall@5**, **0.4827 MRR@10**, and **57.0% Hit@5**, outperforming BM25 by **+44.1% relative MRR** ($p = 0.0001$).*
> - * **Approach A2 (Linear Probe)** achieves **41.17% Recall@5** and **0.3591 MRR@10**, statistically tying BM25 ($p = 0.9031$) while requiring only 5.9 seconds of training.*
> - * **Approach A1 (BiLSTM)** reaches only **20.25% Recall@5**, while **A4 (PrahokBART)** reaches **15.17% Recall@5**.*
>
> *The learning curves on the right explain these outcomes through core deep learning concepts:*
> 1. * **A1 (from scratch)** exhibits severe **overfitting**: training loss collapses to 0.02 while validation loss stalls at 0.82. With only ~1,100 training pairs, random embeddings cannot learn the vast morphological complexity of Khmer legal terms.*
> 2. * **A2 (frozen probe)** cannot overfit, but suffers an **inductive capacity bottleneck**: the single linear layer cannot warp generic multilingual embeddings into nuanced legal geometry.*
> 3. * **A3 (full fine-tuning)** achieves both low loss and high generalization, successfully adapting attention heads to connect citizen queries with statutory clauses."*

---

### Slide 9: Qualitative Error Analysis & Study Limitations (⏱ 7:25 – 8:20)
> *"To understand remaining failure modes, we categorized all misses at $k=5$ across the 200 test questions:*
> - * **Vocabulary Mismatch (60.5% of misses)** remains the largest challenge, occurring on queries lacking key statutory qualifiers.*
> - * **Near-Misses (19.8%)**: In nearly a fifth of misses, A3 retrieved the correct article just outside the top-5 window (ranks 6–10).*
> - * **Multi-Article Questions (17.4%)**: Questions requiring two distinct articles often retrieve one in top-5 while the second falls to rank 7.*
>
> *We also observed a dramatic **Distribution Shift**: on held-out titles (T-T, in-distribution), all models achieve near-ceiling scores (A3: 99.3%, BM25: 97.3%). But on natural language questions (T-Q), scores drop by 45 to 65 percentage points. This confirms that real citizen questions introduce an authentic linguistic shift.*
>
> *Study limitations include evaluation on two statutory codes (Civil and Criminal) and 200 human benchmark questions. Processing scanned PDFs like the Labour Law requires OCR and is slated for future work."*

---

### Slide 10: Citation-Grounded Prototype Demonstration (⏱ 8:20 – 9:20)
> *"We deployed our winning Approach A3 model into an end-to-end citation-grounded Khmer legal assistant running on `localhost:8501`.*
>
> *The architecture operates in under 100 milliseconds: the user query is encoded into a 768-dimensional normalized vector, ranked across all 1,976 pre-computed article vectors using cosine similarity, and fed into DeepSeek Flash with an automated citation verification engine.*
>
> *(Brief demonstration on screen): When we run our challenging query — «តើកិច្ចសន្យាបង្កើតឡើងដោយរបៀបណា?» — BM25 fails and returns Article 623. Our A3 neural retriever correctly retrieves **Article 311** at Rank 1 (similarity 0.608) and **Article 336** at Rank 2 (similarity 0.531).*
>
> *The LLM synthesizes an accurate legal explanation citing Article 336, and our automated verification engine confirms the citation exists verbatim in the retrieved statutory chunk, displaying a green **[Verified Citation]** badge to eliminate hallucination."*

---

### Slide 11: Conclusions, Practical Takeaways & Future Work (⏱ 9:20 – 10:00)
> *"To conclude our defense, three core takeaways emerge from this study:*
>
> 1. * **Pre-training is non-negotiable** for low-resource legal retrieval: from-scratch neural models fail outright on small corpora.*
> 2. * **Full fine-tuning (A3) is strictly necessary** to bridge citizen conversational language and formal statutory law, providing a statistically confirmed **+44.1% improvement over keyword search** ($p < 0.001$).*
> 3. * **Engineering integrity**: 100% consecutive Limon extraction, zero test leakage, 136 passing automated unit tests, and open-source code under GPL-3.0.*
>
> *Future work will expand our corpus with OCR for scanned labor laws and investigate cross-encoder rerankers.*
>
> *Thank you very much, Mr. Soklong and committee members. I am now eager to take your questions."*

---

## 🛡️ 5-Minute Q&A Defense Playbook (Slides A1 – A4)

During the 5-minute Q&A, the lecturer assesses your depth, accuracy, and honesty (Rubric: 10% Response to Questions + 50% Technical Depth). Use arrow keys or the slide counter to navigate to the reserve appendix slides:

### 1. "How do you know A3's gain over BM25 is statistically significant and not just luck?"
- **Navigate to**: **Slide A1 (Paired Bootstrap Hypothesis Testing)**
- **Answer**:
  > *"We performed paired bootstrap hypothesis testing with 10,000 resamples directly on query-by-query score differences ($\Delta = \text{Score}_{A3} - \text{Score}_{BM25}$).*
  > *For Recall@5, A3 achieves a mean gain of $+0.1067$ with an empirical 95% confidence interval strictly above zero ($[+0.0500, +0.1633]$) and a $p$-value of $0.0002$ (41 wins, 13 losses, 146 ties).*
  > *For MRR@10, the gain is $+0.1477$ ($p = 0.0001$, 72 wins vs 25 losses). Because $p < 0.001$, the superiority is statistically confirmed at the 99.9% confidence level.*
  > *Interestingly, testing A2 against BM25 yields $p = 0.9031$, confirming that a frozen linear probe achieves true statistical parity with BM25."*

### 2. "Why did PrahokBART (A4) perform worse on real questions than XLM-R (A3)?"
- **Navigate to**: **Slide A2 & Slide A3**
- **Answer**:
  > *"PrahokBART has two structural differences: first, its parameter scale is 35.8M parameters (~8× smaller than XLM-R's 278M). Second, it was pre-trained as a sequence-to-sequence generation model on news text using BART denoising objectives, rather than masked language modeling.*
  > *On in-distribution title retrieval (T-T), PrahokBART's native 32k Khmer SentencePiece tokenizer helped it achieve 80.4% Recall@5. However, on conversational questions (T-Q), it lacks the multi-layer semantic abstraction capacity required to map colloquial synonyms into statutory concepts, dropping to 15.17% Recall@5."*

### 3. "Explain your loss function: why InfoNCE and why temperature $\tau = 0.05$?"
- **Navigate to**: **Slide 6**
- **Answer**:
  > *"We used symmetric InfoNCE contrastive loss because retrieval is a ranking problem over unbounded negative candidates. By treating all other $B-1$ passages in the mini-batch as negative samples, the model optimizes mutual information lower bounds between questions and matching articles.*
  > *The temperature parameter $\tau$ scales the logits before softmax: a small temperature like $\tau = 0.05$ creates a sharper probability distribution that heavily penalizes false positives and forces the encoder to produce well-separated embeddings on the unit hypersphere."*

### 4. "Show me in the code how you prevented data leakage."
- **Preparation**: Open your laptop editor to `src/dl/prepare_splits.py` lines 80–130.
- **Answer**:
  > *"Data leakage was prevented at three distinct levels in `src/dl/prepare_splits.py`:*
  > 1. *In `quarantine_test_articles()`, we read the 200 benchmark questions and collected all 261 unique ground-truth article IDs. Any title pair originating from these 261 articles was completely excluded from both training and validation splits.*
  > 2. *In `strip_title_from_passage()`, we used regex to remove the leading `មាត្រា N.- Title` line from article bodies so the bi-encoder could not learn trivial lexical matching.*
  > 3. *Splits were generated once with fixed seed 42 into `data/05_splits/`, and test sets were evaluated strictly once after hyperparameter tuning."*

### 5. "What was the role of AI in this project?"
- **Answer**:
  > *"In full compliance with Section 5.E and documented in README Section 10:*
  > *AI assistants (Claude and Antigravity) were used for project scaffolding, generating unit test boilerplate, drafting Limon syllable reordering test cases, and formatting markdown tables.*
  > *All deep learning architectures (A1–A4), the InfoNCE loss function, training loops, evaluation metrics (Recall@k, MRR@10, paired bootstrap tests), and error analyses were implemented in PyTorch, manually verified, and run locally by myself. I wrote and understand every line of code in the repository and can modify any function right now."*

---

## 💻 Repository Code Navigation Quick-Reference for Q&A

If the lecturer asks you to open a file and explain or edit code (§6.3):

| Topic | File Path | Key Functions / Classes |
|:---|:---|:---|
| **A3 Full Fine-Tune** | `src/dl/models/xlmr_finetune.py` | `XLMRFullFinetuneRetriever`, `forward()`, `mean_pooling()` |
| **A2 Linear Probe** | `src/dl/models/linear_probe.py` | `XLMRLinearProbeRetriever` (frozen backbone, linear head) |
| **A1 BiLSTM** | `src/dl/models/bilstm.py` | `BiLSTMRetriever`, embedding layer, packed sequences |
| **A4 PrahokBART** | `src/dl/models/prahokbart.py` | `PrahokBARTRetriever`, BART encoder extraction |
| **Contrastive Loss** | `src/dl/loss.py` | `symmetric_infonce_loss()`, cosine scaling, temperature |
| **Training Loop** | `src/dl/train.py` | `train_epoch()`, `validate()`, optimizer warmup, checkpointing |
| **Evaluation Harness**| `src/dl/evaluate.py` | `evaluate_retriever()`, Recall@k, MRR@10, Hit@5 |
| **Bootstrap Tests** | `src/dl/report.py` | `paired_bootstrap_test()`, 10,000 resamples, confidence intervals |
| **Limon Converter** | `src/infrastructure/extractors/limon_converter.py` | `LimonConverter`, finite-state vowel reordering |
| **Streamlit UI** | `src/interfaces/ui/app.py` | `load_cached_torch_embedder()`, `execute_unified_retrieval()` |
