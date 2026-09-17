### Primary Benchmark (T-Q: 200 Human-Verified Questions)

| Approach / Model | Type | Trainable Params | Recall@1 [95% CI] | Recall@5 [95% CI] | MRR@10 [95% CI] | Hit@5 [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| BM25 Baseline | Sparse Lexical | 0 | **0.2025** [0.1525, 0.2501] | **0.4075** [0.3475, 0.4675] | **0.3350** [0.2803, 0.3860] | **0.4700** [0.4050, 0.5350] |
| Multilingual-E5 (Zero-Shot) | Pretrained Dense | 0 | **0.1625** [0.1150, 0.2076] | **0.2617** [0.2100, 0.3184] | **0.2588** [0.2047, 0.3127] | **0.3200** [0.2550, 0.3850] |
| A1: BiLSTM Dual Encoder | From Scratch Dense | 1.97 M | **0.0875** [0.0525, 0.1250] | **0.2025** [0.1574, 0.2525] | **0.1724** [0.1332, 0.2170] | **0.2450** [0.1900, 0.3050] |
| A2: XLM-R Linear Probe | Frozen + Linear Head | 0.59 M | **0.2400** [0.1875, 0.2950] | **0.4117** [0.3491, 0.4751] | **0.3591** [0.2984, 0.4152] | **0.4750** [0.4050, 0.5400] |
| A3: XLM-R Full Fine-Tuning | Full Fine-Tuning | 278.04 M | **0.3150** [0.2575, 0.3701] | **0.4942** [0.4267, 0.5567] | **0.4499** [0.3866, 0.5106] | **0.5550** [0.4850, 0.6200] |

### Secondary Benchmark (T-T: 148 Held-Out Article Titles)

| Approach / Model | Type | Recall@1 [95% CI] | Recall@5 [95% CI] | MRR@10 [95% CI] | Hit@5 [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: |
| BM25 Baseline | Sparse Lexical | **0.7973** [0.7365, 0.8581] | **0.9730** [0.9459, 0.9932] | **0.8677** [0.8229, 0.9099] | **0.9730** [0.9459, 0.9932] |
| Multilingual-E5 (Zero-Shot) | Pretrained Dense | **0.5405** [0.4595, 0.6149] | **0.7230** [0.6486, 0.7838] | **0.6235** [0.5537, 0.6864] | **0.7230** [0.6486, 0.7838] |
| A1: BiLSTM Dual Encoder | From Scratch Dense | **0.5135** [0.4324, 0.6014] | **0.7568** [0.6959, 0.8243] | **0.6242** [0.5584, 0.6925] | **0.7568** [0.6959, 0.8243] |
| A2: XLM-R Linear Probe | Frozen + Linear Head | **0.7905** [0.7230, 0.8514] | **0.9392** [0.8986, 0.9730] | **0.8563** [0.8053, 0.8995] | **0.9392** [0.8986, 0.9730] |
| A3: XLM-R Full Fine-Tuning | Full Fine-Tuning | **0.9189** [0.8716, 0.9595] | **0.9865** [0.9662, 1.0000] | **0.9467** [0.9150, 0.9736] | **0.9865** [0.9662, 1.0000] |

### Computational Efficiency & Resource Footprint

| Approach / Model | Total Params | Trainable Params | Training Time | Hardware |
| :--- | :---: | :---: | :---: | :--- |
| BM25 Baseline | 0 | 0 | 0 s | CPU |
| Multilingual-E5 (Zero-Shot) | 278.04 M | 0 | 0 s | CPU |
| A1: BiLSTM Dual Encoder | 1.97 M | 1.97 M | 2035.3 s | CPU |
| A2: XLM-R Linear Probe | 278.63 M | 0.59 M | 5.9 s | CPU |
| A3: XLM-R Full Fine-Tuning | 278.04 M | 278.04 M | 319.8 s | CPU (fp32) |