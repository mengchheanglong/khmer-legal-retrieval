### Primary Benchmark (T-Q: 200 Human-Verified Questions)

| Approach / Model | Type | Trainable Params | Recall@1 [95% CI] | Recall@5 [95% CI] | MRR@10 [95% CI] | Hit@5 [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| BM25 Baseline | Sparse Lexical | 0 | **0.2025** [0.1525, 0.2501] | **0.4075** [0.3475, 0.4675] | **0.3350** [0.2803, 0.3860] | **0.4700** [0.4050, 0.5350] |
| Multilingual-E5 (Zero-Shot) | Pretrained Dense | 0 | **0.1625** [0.1150, 0.2076] | **0.2617** [0.2100, 0.3184] | **0.2588** [0.2047, 0.3127] | **0.3200** [0.2550, 0.3850] |
| A1: BiLSTM Dual Encoder | From Scratch Dense | 1.97 M | **0.0875** [0.0525, 0.1250] | **0.2025** [0.1574, 0.2525] | **0.1724** [0.1332, 0.2170] | **0.2450** [0.1900, 0.3050] |
| A2: XLM-R Linear Probe | Frozen + Linear Head | 0.59 M | **0.2400** [0.1875, 0.2950] | **0.4117** [0.3491, 0.4751] | **0.3591** [0.2984, 0.4152] | **0.4750** [0.4050, 0.5400] |
| A3: XLM-R Full Fine-Tuning | Full Fine-Tuning | 278.04 M | **0.3375** [0.2750, 0.3975] | **0.5142** [0.4517, 0.5800] | **0.4827** [0.4216, 0.5468] | **0.5700** [0.5000, 0.6400] |
| A4: PrahokBART Dual Encoder | Khmer Pretrained (Fine-Tuning) | 35.83 M | **0.0525** [0.0250, 0.0850] | **0.1517** [0.1050, 0.2008] | **0.1035** [0.0715, 0.1395] | **0.1800** [0.1300, 0.2351] |

### Secondary Benchmark (T-T: 148 Held-Out Article Titles)

| Approach / Model | Type | Recall@1 [95% CI] | Recall@5 [95% CI] | MRR@10 [95% CI] | Hit@5 [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: |
| BM25 Baseline | Sparse Lexical | **0.7973** [0.7365, 0.8581] | **0.9730** [0.9459, 0.9932] | **0.8677** [0.8229, 0.9099] | **0.9730** [0.9459, 0.9932] |
| Multilingual-E5 (Zero-Shot) | Pretrained Dense | **0.5405** [0.4595, 0.6149] | **0.7230** [0.6486, 0.7838] | **0.6235** [0.5537, 0.6864] | **0.7230** [0.6486, 0.7838] |
| A1: BiLSTM Dual Encoder | From Scratch Dense | **0.5135** [0.4324, 0.6014] | **0.7568** [0.6959, 0.8243] | **0.6242** [0.5584, 0.6925] | **0.7568** [0.6959, 0.8243] |
| A2: XLM-R Linear Probe | Frozen + Linear Head | **0.7905** [0.7230, 0.8514] | **0.9392** [0.8986, 0.9730] | **0.8563** [0.8053, 0.8995] | **0.9392** [0.8986, 0.9730] |
| A3: XLM-R Full Fine-Tuning | Full Fine-Tuning | **0.9054** [0.8514, 0.9459] | **0.9932** [0.9730, 1.0000] | **0.9456** [0.9155, 0.9704] | **0.9932** [0.9730, 1.0000] |
| A4: PrahokBART Dual Encoder | Khmer Pretrained (Fine-Tuning) | **0.5946** [0.5133, 0.6757] | **0.8041** [0.7365, 0.8716] | **0.6733** [0.5990, 0.7422] | **0.8041** [0.7365, 0.8716] |

### Computational Efficiency & Resource Footprint

| Approach / Model | Total Params | Trainable Params | Training Time | Hardware |
| :--- | :---: | :---: | :---: | :--- |
| BM25 Baseline | 0 | 0 | 0 s | CPU |
| Multilingual-E5 (Zero-Shot) | 278.04 M | 0 | 0 s | CPU |
| A1: BiLSTM Dual Encoder | 1.97 M | 1.97 M | 2035.3 s | CPU |
| A2: XLM-R Linear Probe | 278.63 M | 0.59 M | 5.9 s | CPU |
| A3: XLM-R Full Fine-Tuning | 278.04 M | 278.04 M | 7237.5 s | CPU (fp32) |
| A4: PrahokBART Dual Encoder | 35.83 M | 35.83 M | 1602.2 s | CPU (fp32) |

### Paired Statistical Significance (T-Q Primary Benchmark, 10,000 Bootstrap Resamples)

| Comparison | Metric | Mean A | Mean B | Difference (Δ) [95% CI] | p-value | Significance | Wins / Losses / Ties |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| A3 (Fine-Tuning) vs BM25 Baseline | Recall@5 | 0.5142 | 0.4075 | **+0.1067** [+0.0500, +0.1633] | 0.0002 | *** (p < 0.001) | 41W / 13L / 146T |
| A3 (Fine-Tuning) vs BM25 Baseline | MRR@10 | 0.4827 | 0.3350 | **+0.1477** [+0.0962, +0.1993] | 0.0001 | *** (p < 0.001) | 72W / 25L / 103T |
| A3 (Fine-Tuning) vs E5 Zero-Shot | Recall@5 | 0.5142 | 0.2617 | **+0.2525** [+0.1883, +0.3175] | 0.0001 | *** (p < 0.001) | 63W / 6L / 131T |
| A3 (Fine-Tuning) vs E5 Zero-Shot | MRR@10 | 0.4827 | 0.2588 | **+0.2239** [+0.1683, +0.2804] | 0.0001 | *** (p < 0.001) | 80W / 15L / 105T |
| A3 (Fine-Tuning) vs A1 BiLSTM | Recall@5 | 0.5142 | 0.2025 | **+0.3117** [+0.2433, +0.3800] | 0.0001 | *** (p < 0.001) | 77W / 8L / 115T |
| A3 (Fine-Tuning) vs A1 BiLSTM | MRR@10 | 0.4827 | 0.1724 | **+0.3103** [+0.2485, +0.3717] | 0.0001 | *** (p < 0.001) | 104W / 13L / 83T |
| A3 (Fine-Tuning) vs A2 Linear Probe | Recall@5 | 0.5142 | 0.4117 | **+0.1025** [+0.0492, +0.1575] | 0.0001 | *** (p < 0.001) | 40W / 11L / 149T |
| A3 (Fine-Tuning) vs A2 Linear Probe | MRR@10 | 0.4827 | 0.3591 | **+0.1236** [+0.0796, +0.1697] | 0.0001 | *** (p < 0.001) | 65W / 21L / 114T |
| A2 (Linear Probe) vs BM25 Baseline | Recall@5 | 0.4117 | 0.4075 | **+0.0042** [-0.0567, +0.0625] | 0.9031 | n.s. (not sig.) | 26W / 25L / 149T |
| A2 (Linear Probe) vs BM25 Baseline | MRR@10 | 0.3591 | 0.3350 | **+0.0241** [-0.0322, +0.0793] | 0.4006 | n.s. (not sig.) | 51W / 42L / 107T |
| A3 (XLM-R) vs A4 (PrahokBART) | Recall@5 | 0.5142 | 0.1517 | **+0.3625** [+0.2925, +0.4300] | 0.0001 | *** (p < 0.001) | 83W / 5L / 112T |
| A3 (XLM-R) vs A4 (PrahokBART) | MRR@10 | 0.4827 | 0.1035 | **+0.3792** [+0.3195, +0.4396] | 0.0001 | *** (p < 0.001) | 116W / 6L / 78T |
| A4 (PrahokBART) vs BM25 Baseline | Recall@5 | 0.1517 | 0.4075 | **-0.2558** [-0.3208, -0.1925] | 0.0001 | *** (p < 0.001) | 8W / 65L / 127T |
| A4 (PrahokBART) vs BM25 Baseline | MRR@10 | 0.1035 | 0.3350 | **-0.2315** [-0.2869, -0.1775] | 0.0001 | *** (p < 0.001) | 12W / 90L / 98T |