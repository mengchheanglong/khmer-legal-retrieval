# 5-Minute Defense Master Guide & Speaker Notes

> **Defense Format**: Strictly **5-minute total presentation** (Slides + Live Demo) followed by **5-minute Q&A**.  
> **Source of Truth**: `slides/presentation.html` (exported to `slides/final_presentation.pdf`).  
> **Running Demo URL**: `http://localhost:8501` (Local Streamlit UI with Approach A3 fine-tuned model).

---

## ⏱️ Master 5-Minute Time Budget

| Block | Slide / Section | Time Window | Duration | Focus |
|:---|:---|:---:|:---:|:---|
| **Part 1: Research & DL** | **Slide 1**: Title & Research Motivation | 0:00 – 0:20 | 20s | Problem hook, corpus scale, headline result |
| | **Slide 2**: 1. The Legal Retrieval Problem | 0:20 – 1:00 | 40s | Conversational vs. statutory gap, why BM25 fails |
| | **Slide 3**: 2. Dataset & Four Architectures | 1:00 – 1:45 | 45s | 4 PyTorch encoders, shared InfoNCE loss, zero leakage |
| | **Slide 4**: 3. Results & Deep Learning Insights | 1:45 – 2:40 | 55s | Primary metrics, learning curves, capacity & transfer |
| | **Slide 5**: 4. Prototype Architecture | 2:40 – 3:00 | 20s | End-to-end RAG flow, transition cue to live demo |
| **Part 2: Live Demo** | **💻 Live Streamlit Demonstration** | **3:00 – 4:30** | **90s** | **Real-time retrieval contrast, verified citation analysis** |
| **Part 3: Conclusion** | **Slide 6**: 5. Conclusions & Defense Summary | 4:30 – 5:00 | 30s | 3 core takeaways, engineering rigor, Q&A handoff |
| **Q&A Defense** | **Slides A1–A6 (Reserve Appendix)** | +5:00 | 5 min | Data-backed answers to anticipated teacher questions |

---

## Word-for-Word Spoken Script

### Slide 1: Title & Hook (⏱ 0:00 – 0:20)
> *"Good morning, Mr. Soklong and committee members. Today I am presenting my Deep Learning final project: **'Finding the Right Law, in Khmer'**.*
>
> *We investigate whether deep learning bi-encoders can retrieve the exact statutory article out of nearly two thousand Cambodian Civil and Criminal Code laws when citizens ask questions in everyday Khmer. Our best fine-tuned neural retriever outperforms standard keyword search by **+44% on MRR@10** with statistical significance ($p < 0.001$)."*

---

### Slide 2: The Legal Retrieval Problem (⏱ 0:20 – 1:00)
> *"The core challenge in legal retrieval is the **vocabulary mismatch**. Citizens ask questions in conversational language — for example:*
>
> *<span class="khmer">«តើកិច្ចសន្យាបង្កើតឡើងដោយរបៀបណា?»</span> ('How is a contract formed under the Civil Code?').*
>
> *The ground-truth answers are **Civil Code Article 336** ('Formation of contract by offer and acceptance') and **Article 311** ('Definition of contract').*
>
> *Traditional keyword matching like BM25 fails here: in fact, **57.5% of BM25's misses** are caused purely by wording differences. Furthermore, official government PDFs were typed in obsolete 1990s Limon fonts, not Unicode. We engineered a custom Limon-to-Unicode converter with finite-state vowel reordering, achieving **100% consecutive article extraction** across all 1,976 articles."*

---

### Slide 3: Dataset & Four Architectures (⏱ 1:00 – 1:45)
> *"To compare architectures fairly under course guidelines, we evaluated **four distinct PyTorch approaches** against one shared contrastive benchmark:*
>
> 1. * **A1: BiLSTM dual encoder** trained from scratch with character/word tokens (1.97M params).*
> 2. * **A2: XLM-RoBERTa linear probe**, freezing the 278M transformer backbone and training only a linear projection head (0.59M params).*
> 3. * **A3: XLM-RoBERTa fully fine-tuned**, adapting all 278M weights end-to-end — our winning model.*
> 4. * **A4: PrahokBART**, a Khmer-only pre-trained sequence-to-sequence model fine-tuned as a dual encoder (35.8M params).*
>
> *Every model was trained with the exact same **InfoNCE contrastive loss** ($\tau = 0.05$) on identical splits (seed 42). All 200 human test questions and their 261 cited articles were strictly quarantined to guarantee **zero test leakage**."*

---

### Slide 4: Results & Deep Learning Insights (⏱ 1:45 – 2:40)
> *"Looking at the empirical results on the left: **Approach A3 is the clear winner**, achieving **51.4% Recall@5** and **0.483 MRR@10**, beating BM25 by **+44.1%** ($p = 0.0002$, paired bootstrap test). Surprisingly, the 6-second frozen probe (A2) statistically ties keyword search ($p = 0.9031$) for free.*
>
> *The learning curves on the right illustrate fundamental deep learning principles:*
>
> - * **A1 (from scratch)** exhibits classic **overfitting**: training loss drops to 0.02 while validation loss stalls. Without pre-training, ~1,100 training pairs are insufficient to learn Khmer legal syntax from scratch.*
> - * **A2 (frozen)** cannot overfit, but suffers an **inductive capacity bottleneck**: the frozen generic embeddings cannot adapt to Khmer legal taxonomy.*
> - * **A3 (full fine-tuning)** achieves both low loss and high validation MRR (0.9095 in our 6-run Tesla T4 grid search), successfully bridging everyday questions to statutory law."*

---

### Slide 5: Prototype Architecture & Live Demo Cue (⏱ 2:40 – 3:00)
> *"We brought this winning A3 model into production inside a citation-grounded Khmer RAG assistant built with FastAPI, Streamlit, and DeepSeek Flash.*
>
> *The pipeline encodes the citizen's query in under 100 milliseconds, ranks the 1,976 articles via cosine similarity, validates statutory citations against retrieved text, and generates legal analysis with verified badges.*
>
> *Let us now switch to the live demonstration at `localhost:8501`."*

---

## 💻 Live Demo Step-by-Step Runbook (⏱ 3:00 – 4:30 · 90 Seconds)

### Step 1: Switch to Browser (`http://localhost:8501`) (~10s)
- **Screen**: Streamlit Web UI showing "Khmer Legal Article Retrieval Assistant".
- **Point out to teacher**:
  - Sidebar shows active retriever: **`A3: XLM-R Full Fine-Tune (278M) [Best Model]`**.
  - Total corpus metric: **1,976 articles** (1,304 Civil + 672 Criminal).

### Step 2: Test the Semantic Mismatch Query (~40s)
1. **Dropdown**: In the *"Pick an authentic benchmark question"* dropdown, select:
   > `តើកិច្ចសន្យាបង្កើតឡើងដោយរបៀបណា យោងតាមក្រមរដ្ឋប្បវេណី? (Civil Code Art. 311/336)`
2. **Explain before clicking**:
   > *"Notice how the citizen asks 'បង្កើតឡើងដោយរបៀបណា' ('in what way is it formed'). If we ran standard keyword search, the word 'របៀប' distracts BM25 and returns Article 623 ('Manner of dividing proceeds') — a complete failure.*
   > *Now let's run our fine-tuned Deep Learning retriever (Approach A3)..."*
3. **Click button**: `Run Legal Retrieval & Analysis`.
4. **Point to Retrieval Results**:
   > *"Look at the retrieved articles in under 100 milliseconds:*
   > - * **Rank 1**: **Civil Code Article 311** (<span class="khmer">និយមន័យនៃកិច្ចសន្យា</span> — 'Definition of Contract') with similarity **0.6083**.*
   > - * **Rank 2**: **Civil Code Article 336** (<span class="khmer">ការបង្កើតកិច្ចសន្យាដោយសំណើ និង ស្វីការ</span> — 'Formation of Contract by Offer and Acceptance') with similarity **0.5307**.*
   > *The neural bi-encoder grasped the underlying legal concept despite zero verbatim phrasing overlap."*

### Step 3: Show Verified Citation Analysis (~30s)
- **Point to Legal Analysis output**:
  > *"Because the context contains the true statutory articles, the LLM generates a precise legal explanation stating that under Article 336, a contract is formed when an offer and acceptance coincide.*
  > *Notice the green **[Verified Citation]** badge: our verification engine checked that Article 336 exists verbatim in the retrieved statutory chunk, preventing hallucination."*

### Step 4: Show Benchmark Tab (~10s)
- Click on **Tab 3: "DL Model Benchmark & Evaluation"**.
- Point out:
  > *"All evaluation metrics, ROC/PR curves, and training comparisons are embedded directly in the interface for reproducibility."*

---

### Slide 6: Conclusions & Defense Summary (⏱ 4:30 – 5:00)
> *(Switch back to slide deck)*
>
> *"To summarize our findings:*
>
> 1. * **Pre-training is non-negotiable** for low-resource languages like Khmer: from-scratch models fail outright at small legal scales.*
> 2. * **Full fine-tuning (A3)** is required to cross the domain gap between citizen speech and statutory law, yielding a statistically validated **+44.1% improvement**.*
> 3. * **Engineering integrity**: 100% consecutive Limon extraction, 136 passing automated unit tests, and model checkpoints hosted publicly on Hugging Face Hub.*
>
> *Thank you, Mr. Soklong and committee. I am now open for your questions."*

---

## 🛡️ Q&A Defense Strategy (Appendix Slides A1 – A6)

When the lecturer or committee asks technical questions, jump directly to the relevant appendix slide using the arrow keys or slide counter:

### Q1: "Are your results statistically significant or just random noise?"
- **Jump to**: **Slide A2 (Is the Difference Real?)**
- **Answer**:
  > *"We ran a paired bootstrap hypothesis test with 10,000 resamples across the 200 real questions. Approach A3 beats BM25 with $p = 0.0002$ on Recall@5 and $p = 0.0001$ on MRR@10. In contrast, Approach A2's difference from BM25 has $p = 0.9031$, confirming true statistical parity."*

### Q2: "Why did PrahokBART (A4) perform worse than XLM-R (A3)?"
- **Jump to**: **Slide A3 (Distribution Shift) & Table on A1**
- **Answer**:
  > *"PrahokBART was pre-trained as a sequence-to-sequence generation model on news text with 35.8M parameters (~8× smaller than XLM-R's 278M). While it scored 80.4% on formal title matching (in-distribution), it dropped to 15.2% on conversational questions because it lacks the deep semantic abstraction capacity of larger masked language models."*

### Q3: "Did you tune hyperparameters systematically?"
- **Jump to**: **Slide A5 (Hyperparameter Tuning & Reproducibility)**
- **Answer**:
  > *"Yes, every approach underwent empirical tuning. Specifically, Approach A3 was tuned via a 6-configuration grid search across learning rate $\{1\times 10^{-5}, 2\times 10^{-5}, 3\times 10^{-5}\}$ and weight decay $\{0.0, 0.01\}$ on an NVIDIA Tesla T4 GPU. The configuration `lr=2e-5, wd=0.01` achieved the highest validation MRR@10 of 0.9095 at epoch 4."*

### Q4: "How did you handle the broken Khmer PDF fonts without OCR?"
- **Jump to**: **Slide A6 (Engineering: Limon Font Converter)**
- **Answer**:
  > *"The official ODC PDFs contained a complete digital text layer, but encoded in legacy 1990s Limon ASCII keystrokes. We built a custom byte-level character mapping and a finite-state reordering machine that correctly shifts pre-base vowels and subscript consonants into Unicode logical order, verifying 1,304 consecutive Civil and 672 Criminal articles."*

### Q5: "What are the primary failure modes of your model?"
- **Jump to**: **Slide A4 (Error Analysis)**
- **Answer**:
  > *"Our error taxonomy on the 200 test set shows that 60.5% of remaining misses are multi-hop or subtle vocabulary mismatches where queries lack key legal qualifiers, 19.8% are near-misses (relevant article at rank 6–10), and 17.4% are multi-article queries where only 1 of 2 articles was retrieved in top-5."*
