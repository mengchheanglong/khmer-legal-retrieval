# 10-Minute Oral Defense Master Guide & Spoken Script

> **Assessment**: Individual Deep Learning Final Project (50% Course Grade)  
> **Course Guidelines Reference**: Department of Engineering · Bachelor of Software Engineering (§6.2, §6.3, §7.1)  
> **Defense Format**: Strictly **10-minute presentation** (12 Main Slides + 90s Live Demo) followed by **5-minute Q&A** (§6.3).  
> **Deck Artifact**: `slides/presentation.html` (Compiled to 16:9 vector PDF: `slides/final_presentation.pdf`, 16 slides total).  
> **Live Demo URL**: `http://localhost:8501` (Localhost Streamlit UI with Approach A3 PyTorch neural encoder).

---

## ⏱️ Master 10-Minute Time Budget

| Block | Slide / Section | Time Window | Duration | Key Rubric Focus (§6.2 & §7.1) |
|:---|:---|:---:|:---:|:---|
| **Part 1: The Problem & Data** | **Slide 1**: Title & Research Hook | 0:00 – 0:30 | 30s | Relatable hook, 1,976 articles, headline result (+44% MRR) |
| | **Slide 2**: Why Ordinary Search Struggles | 0:30 – 1:15 | 45s | Vocabulary gap, lack of word spaces, 57.5% BM25 miss rate |
| | **Slide 3**: Before AI, We Needed Usable Data | 1:15 – 2:05 | 50s | 1990s Limon ASCII fonts, FSM syllable reordering, 1,976 articles |
| | **Slide 4**: Teaching a Model Legal Meaning | 2:05 – 2:55 | 50s | Title-to-body weak pairs, title stripping, 261 protected leakage guard |
| **Part 2: The Experiments** | **Slide 5**: Four Experiments, Four Ideas | 2:55 – 4:05 | 70s | 4 PyTorch models (A1–A4), scratch vs probe vs full FT vs BART |
| | **Slide 6**: What Happens During Training? | 4:05 – 4:50 | 45s | In-batch negative contrastive learning, symmetric InfoNCE ($\tau=0.05$) |
| | **Slide 7**: How We Made the Comparison Fair | 4:50 – 5:40 | 50s | Fixed seed 42, A3 tuning grid, learning curves (overfit vs bottleneck) |
| **Part 3: The Findings** | **Slide 8**: Which Architecture Worked Best? | 5:40 – 6:25 | 45s | Consolidated leaderboard table (Recall@5: 51.42%, MRR: 0.4827, $p < 0.001$) |
| | **Slide 9**: The Surprising Discovery (99% vs 51%) | 6:25 – 7:20 | 55s | **Centerpiece**: Title T-T (99.3%) vs Real Questions T-Q (51.4%) |
| | **Slide 10**: Where the Model Still Fails | 7:20 – 8:00 | 40s | 5 failure categories, near-misses, multi-article questions, scope |
| **Part 4: The Proof & Takeaways**| **Slide 11**: Live Localhost Demonstration | 8:00 – 9:30 | 90s | Live BM25 vs A3 comparison on TQ-CIV-009 (Art. 24 guardianship) |
| | **Slide 12**: What We Learned & Takeaways | 9:30 – 10:00 | 30s | 3 core takeaways, engineering integrity, transition to Q&A |
| **Part 5: Committee Defense** | **Slides A1–A4 (Reserve Appendix)** | **10:00 – 15:00**| **5 min** | **Data-backed answers to anticipated lecturer questions (Rubric: 70%)** |

---

## 🎙️ Word-for-Word Spoken Defense Script

### Slide 1: Finding the Right Law, in Khmer (⏱ 0:00 – 0:30)
> *"Good morning Mr. Soklong and esteemed committee members. Today I am presenting my Deep Learning final project: **'Finding the Right Law, in Khmer: Comparing Four Deep-Learning Retrievers on Cambodian Statutory Law'**.*
>
> *Imagine a citizen in Cambodia trying to understand their rights under the law. They ask a simple question in conversational Khmer: **«តើកិច្ចសន្យាត្រូវបង្កើតឡើងដោយរបៀបណា?»** ('How is a contract formed?').*
>
> *Finding the answer requires searching through nearly two thousand articles across the Civil and Criminal Codes, written in dense, formal legal terminology. Our central research question explores whether deep neural bi-encoders can bridge this everyday language gap to retrieve the exact governing article. Our winning fine-tuned retriever outperforms standard keyword search by **+44.1% on MRR@10** with statistical significance ($p < 0.001$)."*

---

### Slide 2: Why Ordinary Keyword Search Struggles (⏱ 0:30 – 1:15)
> *"Why can't we simply use ordinary keyword search like BM25?*
>
> *First, statutory drafting uses specialized vocabulary that everyday citizens rarely use. For example, when a citizen asks «តើកិច្ចសន្យាបង្កើតឡើងដោយរបៀបណា?», the governing statutes are **Civil Code Article 336** ('Formation of contract by offer and acceptance') and **Article 311** ('Definition of contract').*
>
> *Keyword search matches the colloquial question word «របៀប» ('manner' or 'how') and gets misled into retrieving **Article 623** ('Manner of dividing proceeds in co-ownership') — which has nothing to do with contracts! In fact, our quantitative error audit reveals that **57.5% of BM25's retrieval misses are caused purely by this lexical vocabulary mismatch**.*
>
> *Second, Khmer has no word spaces, making keyword search fragile to word segmentation errors. Dense neural retrievers solve this by projecting queries and statutes into a continuous semantic space where proximity reflects legal meaning rather than shared keywords."*

---

### Slide 3: Before AI, We Needed Usable Data: The Limon Font Hurdle (⏱ 1:15 – 2:05)
> *"Before we could train any neural network, we faced a major data engineering challenge: the official government legal PDFs from Open Development Cambodia are not encoded in Unicode.*
>
> *They were typeset in the 1990s using legacy **Limon ASCII fonts**. If you extract text with standard tools, you get meaningless Latin characters such as `maRta 336>- karbegáItkic©snñ`.*
>
> *Even worse, Khmer writing involves visual grapheme inversion: vowels like េ, ែ, and ៃ are typed visually before the consonant, but logical Unicode requires them after.*
>
> *To solve this without noisy OCR, I engineered `LimonPdfExtractor`, implementing a byte-level glyph mapping and a finite-state machine that reconstructs logical Unicode clusters. This produced **100% consecutive article extraction across all 1,976 articles** (1,304 Civil Code + 672 Criminal Code articles), audited with zero missing articles and preserving full statutory hierarchy from Book down to Article."*

---

### Slide 4: Teaching a Model Legal Meaning: Zero-Leakage Weak Supervision (⏱ 2:05 – 2:55)
> *"Because labeled Khmer legal question-answering datasets do not exist, we leveraged the structure of statutory law for weak supervision: matching article titles to article bodies, giving us 1,471 unique training candidates after removing ambiguous repeated titles like 'Definitions'.*
>
> *To satisfy Section 5.A of the course instructions, we enforced an uncompromising data-leakage quarantine:*
>
> 1. *We compiled a **Primary Benchmark (T-Q) of 200 authentic human legal questions**, citing 261 specific articles.*
> 2. *Every single one of these 261 target articles was strictly quarantined — no training pair was ever generated from them.*
> 3. *The title line was stripped from all training passages, forcing encoders to learn semantic concepts rather than lexical copy-pasting.*
> 4. *The remaining corpus was deterministically split with random seed 42 into 1,176 training pairs, 147 validation pairs, and 148 held-out test title pairs (T-T), with test sets evaluated strictly once."*

---

### Slide 5: Four Experiments, Four Different Ideas (⏱ 2:55 – 4:05)
> *"To conduct a rigorous, rubric-compliant study under Section 4, we implemented **four distinct PyTorch deep learning architectures**, each testing a different research hypothesis:*
>
> 1. * **Approach A1: BiLSTM Dual Encoder from scratch (1.97M parameters)**. A single-layer bidirectional LSTM with random embeddings. This tests: *Can an encoder learn legal semantics from scratch with only ~1,100 training pairs?*
> 2. * **Approach A2: XLM-RoBERTa Linear Probe (0.59M trainable parameters)**. We freeze all 278M transformer weights of `multilingual-e5-base` and train only a linear projection head. This tests: *Can fast, parameter-efficient training in 5.9 seconds adapt generic multilingual features to law?*
> 3. * **Approach A3: XLM-RoBERTa Full Fine-Tuning (278.04M trainable parameters)**. We train all 12 transformer encoder layers end-to-end on an NVIDIA Tesla T4 GPU. This tests: *Does adapting deep self-attention allow the model to connect conversational questions with statutory clauses?*
> 4. * **Approach A4: PrahokBART Dual Encoder (35.83M trainable parameters)**. A sequence-to-sequence model pre-trained natively on Khmer text with SentencePiece tokenization. This tests: *Does language-specific Khmer pre-training outperform a larger multilingual backbone?*
>
> *We compare all four against BM25 and zero-shot Multilingual-E5."*

---

### Slide 6: What Happens During Training? Contrastive Metric Learning (⏱ 4:05 – 4:50)
> *"All models were trained using a Siamese dual-tower contrastive metric learning formulation with **symmetric InfoNCE loss** and in-batch negatives.*
>
> *In each mini-batch of query-article pairs—such as our optimal A3 training batch size $B = 16$—the matching article is the positive sample, and the other $B - 1$ (15) articles serve as negative samples. The loss computes cosine similarities scaled by a temperature $\tau = 0.05$, pulling the positive article toward the query while repelling the negative articles symmetrically. The 4×4 grid on screen is an illustrative matrix showing how diagonal positives are maximized against off-diagonal negatives.*
>
> *We optimized using AdamW with a 10% linear warmup schedule followed by cosine learning rate decay. Per course requirements in Section 5.B, we saved model weights, optimizer, scheduler, and PyTorch RNG state each epoch to `last.pt` and `best.pt`, with automated `--resume` support."*

---

### Slide 7: How We Made the Comparison Fair: Setup & Training Dynamics (⏱ 4:50 – 5:40)
> *"To ensure a fair comparison under Section 5.C, all models shared the exact same split, the same 1,976-article corpus, and the same evaluation harness with random seed 42.*
>
> *For our primary model, **Approach A3**, we executed a 6-configuration grid search across learning rates and weight decay on a Tesla T4 GPU. Configuration `a3_lr2e-5_wd0.01` achieved the best Validation MRR@10 of **0.9095** at epoch 4.*
>
> *The learning curves on the right illustrate three core deep learning phenomena:*
> - * **A1 (from scratch)** suffered severe **overfitting**: training loss collapsed to 0.02 while validation loss stalled at 0.82. Eleven hundred pairs cannot train random embeddings from scratch.*
> - * **A2 (frozen probe)** hit an **inductive capacity bottleneck**: the single linear layer cannot warp rigid multilingual space into legal geometry.*
> - * **A3 (full fine-tune)** achieved healthy **generalization**, with validation loss descending to 0.1032."*

---

### Slide 8: Which Architecture Worked Best? Empirical Results (⏱ 5:40 – 6:25)
> *"Here are the empirical results on the Primary Benchmark (T-Q: 200 real human questions evaluated against the full 1,976 articles):*
>
> - * **Approach A3 (XLM-R Full Fine-Tune) is the definitive winner**, achieving **51.42% Recall@5**, **0.4827 MRR@10**, and **57.0% Hit@5**, outperforming BM25 by **+44.1% relative MRR** ($p = 0.0001$).*
> - * **Approach A2 (Linear Probe)** achieved **41.17% Recall@5** and **0.3591 MRR@10**, statistically tying BM25 ($p = 0.9031$) while requiring only 5.9 seconds of training.*
> - * **Approach A1 (BiLSTM)** reached only **20.25% Recall@5**, and **A4 (PrahokBART)** reached **15.17% Recall@5**.*
> - * **Zero-Shot Multilingual-E5** scored only **26.17% Recall@5**, demonstrating that off-the-shelf multilingual models fail without Cambodian legal fine-tuning.*
>
> *Paired bootstrap hypothesis testing with 10,000 resamples confirms that A3's superiority over BM25 is statistically significant ($p < 0.001$)."*

---

### Slide 9: The Surprising Discovery: 99% vs 51% (THE CENTERPIECE) (⏱ 6:25 – 7:20)
> *"Now, I want to highlight the single most important discovery of this entire project.*
>
> *When we evaluated our winning A3 model on **held-out article titles (Test A)**, it achieved an astonishing **99.32% Recall@5** and **0.9456 MRR@10**. It found the right article almost flawlessly.*
>
> *Yet, when evaluated on **real human questions (Test B)**, its Recall@5 dropped to **51.42%** — a gap of nearly 48 percentage points!*
>
> *Why did this happen?*
>
> *Because on titles, the queries match the formal statutory vocabulary used during training. But real citizens don't speak in article titles. They describe situations, use colloquial question words, and omit formal legal terms. This creates an authentic **linguistic distribution shift**.*
>
> *This was my biggest research takeaway: choosing a stronger neural architecture gave us a **+44% MRR improvement**, but the **training data distribution** ultimately determines the real-world performance ceiling."*

---

### Slide 10: Where the Model Still Fails: Error Breakdown & Limitations (⏱ 7:20 – 8:00)
> *"To understand where A3 still fails, we categorized all misses at $k=5$ across the 200 benchmark questions:*
> - * **Vocabulary Mismatch (60.5% of misses)** remains the largest hurdle, occurring when questions describe everyday scenarios without statutory roots.*
> - * **Near-Misses (19.8%)**: In nearly a fifth of misses, A3 retrieved the correct article just outside the top-5 window (ranks 6–10). Its Recall@10 reaches **59.83%**.*
> - * **Multi-Article Questions (17.4%)**: Questions spanning two articles (such as Art. 130 and 131) often catch only one in top-5.*
>
> *Regarding scope limitations: our study evaluated the Civil and Criminal Codes. Scanned laws like the 1997 Labour Law require OCR and are left for future work."*

---

### Slide 11: Live Demonstration: BM25 vs A3 on Localhost (⏱ 8:00 – 9:30)
> *"To demonstrate this in practice, let us examine an authentic benchmark question running live on our `localhost:8501` Streamlit application:*
>
> *«តើនរណាខ្លះអាចស្នើសុំឱ្យតុលាការប្រកាសបើកអាណាព្យាបាលទូទៅ?» ('Who can request that a court initiate general guardianship?').*
>
> *The ground-truth governing law is **Civil Code Article 24** ('Declaration of commencement of general guardianship').*
>
> *(Looking at screen):*
> - *When we run **BM25 Keyword Search**, it completely misses Article 24 outside its top 10! BM25 gets distracted by the repeated term 'general guardianship' in later administrative articles (Articles 1105, 1125, 1124).*
> - *Now, we switch to our trained **Approach A3 Neural Retriever**. In under 50 milliseconds, A3 ranks **Civil Code Article 24 at Rank #1 with a cosine similarity of 0.6205**!*
>
> *Our dense encoder understands the conceptual connection between 'requesting the court to open' and Article 24's statutory scope, even when exact keywords differ. DeepSeek Flash then synthesizes a grounded answer with an automated green **[Verified Citation]** badge confirming the article exists in the retrieved context.*
>
> *This illustrates the real-world advantage of our trained neural retriever."*

---

### Slide 12: What We Learned: Three Core Takeaways & Future Work (⏱ 9:30 – 10:00)
> *"To conclude my presentation, three core takeaways emerge from this research:*
>
> 1. * **Pre-training is non-negotiable** for low-resource legal retrieval: from-scratch neural models fail on small corpora.*
> 2. * **Full fine-tuning (A3) is strictly necessary** to bridge citizen questions and formal law, providing a statistically confirmed **+44.1% MRR improvement over keyword search** ($p < 0.001$).*
> 3. * **The data distribution gap is real**: achieving 99% on titles versus 51% on real questions proves that future gains require conversational synthetic training pairs.*
>
> *This project maintains complete engineering integrity: 100% consecutive Limon extraction, zero data leakage, 138 passing automated unit tests, and open-source code under GPL-3.0.*
>
> *Thank you very much, Mr. Soklong and committee members. I am now ready for your questions."*

---

## 🛡️ 5-Minute Committee Defense Playbook (Slides A1 – A4)

During the 5-minute Q&A, navigate to the reserve appendix slides using arrow keys:

### 1. "How do you know A3's gain over BM25 is statistically significant and not just luck?"
- **Navigate to**: **Slide A3 (Paired Bootstrap Statistical Significance)**
- **Spoken Defense**:
  > *"We performed paired non-parametric bootstrap hypothesis testing with 10,000 resamples directly on query-level delta scores ($\Delta_i = s_i^{(A3)} - s_i^{(BM25)}$).*
  > *For Recall@5, A3 achieves a mean gain of $+0.1067$ with an empirical 95% confidence interval strictly above zero ($[+0.0500, +0.1633]$) and a $p$-value of $0.0002$ (41 wins vs 13 losses, 146 ties).*
  > *For MRR@10, the gain is $+0.1477$ with a 95% confidence interval of $[+0.0962, +0.1993]$ ($p = 0.0001$, 72 wins vs 25 losses, 103 ties). Because $p < 0.001$, the superiority is statistically confirmed at the 99.9% confidence level.*
  > *In contrast, testing A2 against BM25 yields a 95% CI of $[-0.0567, +0.0625]$ and $p = 0.9031$ (26 wins vs 25 losses, 149 ties), confirming that a frozen linear probe achieves true statistical parity with BM25."*

### 2. "Why did PrahokBART (A4) perform worse on real questions than XLM-R (A3)?"
- **Navigate to**: **Slide A1 & Slide 8**
- **Spoken Defense**:
  > *"PrahokBART has two major structural differences: first, it has 35.8M parameters, which is roughly 8 times smaller than XLM-R's 278M parameters. Second, it was pre-trained as a sequence-to-sequence generation model using BART denoising objectives on news text, rather than masked language modeling.*
  > *On in-distribution title retrieval (T-T), PrahokBART's native 32k Khmer SentencePiece tokenizer helped it achieve 80.4% Recall@5. However, on conversational questions (T-Q), it lacks the parameter capacity and abstraction depth required to map colloquial synonyms into statutory concepts, dropping to 15.17% Recall@5."*

### 3. "Explain your loss function: why InfoNCE and why temperature $\tau = 0.05$?"
- **Navigate to**: **Slide A2 & Slide 6**
- **Spoken Defense**:
  > *"We formulated training as contrastive metric learning using symmetric InfoNCE with in-batch negatives. Retrieval is a ranking problem over an unbounded corpus, so optimizing mutual information lower bounds between questions and relevant passages is ideal.*
  > *The temperature parameter $\tau$ controls the entropy of the probability distribution over negatives. A standard temperature like 1.0 produces soft gradients that average across all negatives. A sharp temperature like $\tau = 0.05$ concentrates gradients on hard negatives that are close in cosine distance, forcing the encoder to establish well-separated representations on the unit hypersphere."*

### 4. "Show me in the code how you prevented data leakage."
- **Preparation**: Open editor to `src/dl/prepare_splits.py` lines 80–135.
- **Spoken Defense**:
  > *"Data leakage was prevented at three distinct levels in `src/dl/prepare_splits.py`:*
  > 1. *In `quarantine_test_articles()`, we parsed the 200 benchmark questions and gathered all 261 unique ground-truth article IDs. Any title pair originating from these 261 articles was completely excluded from both training and validation splits.*
  > 2. *In `strip_title_from_passage()`, we used regex to remove the leading `មាត្រា N.- Title` line from article bodies so the bi-encoder could not learn trivial lexical matching.*
  > 3. *Splits were generated once with fixed seed 42 into `data/05_splits/`, and test sets were evaluated strictly once after hyperparameter tuning."*

### 5. "What was the role of AI in this project?"
- **Spoken Defense**:
  > *"In full compliance with Section 5.E of the course guidelines and documented in README Section 10:*
  > *AI assistants were used for project scaffolding, generating unit test boilerplate, drafting Limon syllable reordering test cases, and formatting markdown tables.*
  > *All deep learning architectures (A1–A4), the InfoNCE loss function, training loops, evaluation metrics (Recall@k, MRR@10, paired bootstrap tests), and error analyses were implemented in PyTorch, manually verified, and run locally by myself. I wrote and understand every line of code in the repository and can modify any function right now."*

---

## 💻 Repository Code Navigation Quick-Reference for Q&A

If the committee asks you to open a file and explain code (§6.3):

| Topic | File Path | Key Functions / Classes |
|:---|:---|:---|
| **A3 Full Fine-Tune** | `src/dl/models/xlmr_finetune.py` | `XLMRFullFinetuneRetriever`, `forward()`, `_pool()` |
| **A2 Linear Probe** | `src/dl/models/linear_probe.py` | `XLMRLinearProbeRetriever` (frozen backbone, linear head) |
| **A1 BiLSTM Dual Encoder** | `src/dl/models/bilstm.py` | `BiLSTMEncoder`, `pack_padded_sequence`, masked pooling |
| **A4 Khmer PrahokBART** | `src/dl/models/prahokbart.py` | `PrahokBARTDualEncoder`, mean pooling over attention mask |
| **Contrastive InfoNCE Loss** | `src/dl/losses.py` | `InfoNCELoss`, symmetric logit scaling by $\tau=0.05$ |
| **Training Loop** | `src/dl/train.py` | `RetrieverTrainer`, `train_epoch()`, `evaluate_val()` |
| **Evaluation Harness**| `src/dl/evaluate.py` | `evaluate_retriever()`, Recall@k, MRR@10, Hit@5 |
| **Bootstrap Hypothesis Test**| `src/dl/report.py` | `paired_bootstrap_test()`, 10,000 resamples, confidence intervals |
| **Limon FSM Extractor** | `src/infrastructure/extractors/limon_converter.py` | `LimonConverter`, finite-state syllable reordering |
| **Streamlit Interface** | `src/interfaces/ui/app.py` | `load_cached_torch_embedder()`, `execute_unified_retrieval()` |
