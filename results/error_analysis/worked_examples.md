# Qualitative Error Analysis & Worked Examples

This document details representative retrieval cases across distinct error categories on the Primary Benchmark (T-Q: 200 human-verified questions).

### Example 1: Vocabulary Mismatch & Semantic Synonyms (Resolved by A3)
- **Question (Khmer)**: `តើកិច្ចសន្យាដែលធ្វើឡើងដោយការគំរាមកំហែងមានសុពលភាពឬទេ?`
  *(Is a contract entered into under duress/threat valid?)*
- **Expected Ground Truth**: Civil Code Art. 347 (ការគំរាមកំហែង / Duress)
- **BM25 Lexical Baseline**: **Missed (Rank > 10)**. BM25 matched on generic contract terms and missed because the statute uses 'មោឃភាព' (voidability) rather than 'សុពលភាព' (validity).
- **A3 Fine-Tuned XLM-R**: **Rank 1 (Correct)**. Semantically clustered duress ('ការគំរាមកំហែង') with contract invalidation rules.
- **Linguistic Insight**: Legal Khmer frequently contrasts conversational query phrasing (`សុពលភាព`) with technical statutory terminology (`មោឃភាព`). Dense fine-tuning resolves this vocabulary gap.

### Example 2: Civil Code vs Criminal Code Routing
- **Question (Khmer)**: `តើការក្លែងបន្លំឯកសារសាធារណៈត្រូវផ្តន្ទាទោសដូចម្តេច?`
  *(How is forgery of public documents punished?)*
- **Expected Ground Truth**: Criminal Code Art. 629 (ការក្លែងបន្លំឯកសារសាធារណៈ)
- **BM25 Lexical Baseline**: **Rank 1 (Correct)**. Exact lexical match on 'ក្លែងបន្លំឯកសារសាធារណៈ'.
- **A3 Fine-Tuned XLM-R**: **Rank 1 (Correct)**. High confidence routing to Criminal Code Book 6.
- **Linguistic Insight**: When statutory offense names are quoted verbatim, lexical keyword search remains highly effective.

### Example 3: Multi-Article Interdependent Legal Rules
- **Question (Khmer)**: `តើល័ក្ខខ័ណ្ឌអ្វីខ្លះដែលនាំឱ្យកិច្ចសន្យាត្រូវទុកជាមោឃៈ?`
  *(What conditions cause a contract to be rendered void?)*
- **Expected Ground Truth**: Civil Code Arts. 353, 354, 355, 356 (Interdependent nullity rules)
- **BM25 Lexical Baseline**: **Rank 2 (Art. 354 only)**. Missing Arts. 353, 355, 356.
- **A3 Fine-Tuned XLM-R**: **Ranks 1, 2, 4 (Arts. 353, 354, 356 all in Top-5)**.
- **Linguistic Insight**: Complex legal questions spanning multiple related sections are better captured by dense dual encoders due to shared contextual representation.

### Example 4: Near-Miss Specificity Challenge (Rank 6–10)
- **Question (Khmer)**: `តើការលក់ដូរមនុស្សក្នុងគោលដៅអាជីវកម្មផ្លូវភេទមានទោសកម្រិតណា?`
  *(What is the penalty for human trafficking for sexual exploitation?)*
- **Expected Ground Truth**: Criminal Code Art. 254 (ការនាំយកទៅដោយមានគោលដៅអាជីវកម្មផ្លូវភេទ)
- **A2 Linear Probe**: **Rank 4 (Correct)**.
- **A1 BiLSTM**: **Rank 7 (Near-Miss)**. Retrieved general human trafficking provisions in positions 1–5.
- **Linguistic Insight**: Near-misses commonly occur when a specific sub-clause article is overshadowed by a broader introductory article in the same chapter.