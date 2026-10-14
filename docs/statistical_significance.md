# Paired Statistical Significance Testing (Primary Benchmark T-Q)

This document records the complete paired bootstrap hypothesis testing results across all model pairs on the primary human benchmark (**T-Q**: 200 questions, full 1,976-article corpus, no statute filter).

---

## Methodology

Because individual 95% bootstrap confidence intervals can exhibit marginal overlap, declaring statistical superiority requires paired hypothesis testing on query-by-query score differences:
$$\Delta = \text{Score}_A(q) - \text{Score}_B(q)$$

We run **10,000 paired bootstrap resamples** over the 200 evaluation queries. The empirical two-tailed $p$-value represents the proportion of bootstrap resamples where $\Delta \le 0$ (when testing if $A > B$).

---

## Complete Paired Hypothesis Testing Matrix

| Comparison ($A$ vs $B$) | Metric | Mean $A$ | Mean $B$ | Difference ($\Delta$) [95% CI] | $p$-value | Significance | Wins / Losses / Ties |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A3 (Fine-Tuning) vs BM25 Baseline** | **Recall@5** | **0.5142** | **0.4075** | **+0.1067** [+0.0500, +0.1633] | **0.0002** | **\*\*\* ($p < 0.001$)** | 41W / 13L / 146T |
| **A3 (Fine-Tuning) vs BM25 Baseline** | **MRR@10** | **0.4827** | **0.3350** | **+0.1477** [+0.0962, +0.1993] | **0.0001** | **\*\*\* ($p < 0.001$)** | 72W / 25L / 103T |
| A3 (Fine-Tuning) vs E5 Zero-Shot | Recall@5 | 0.5142 | 0.2617 | **+0.2525** [+0.1883, +0.3175] | 0.0001 | *** ($p < 0.001$) | 63W / 6L / 131T |
| A3 (Fine-Tuning) vs E5 Zero-Shot | MRR@10 | 0.4827 | 0.2588 | **+0.2239** [+0.1683, +0.2804] | 0.0001 | *** ($p < 0.001$) | 80W / 15L / 105T |
| A3 (Fine-Tuning) vs A1 BiLSTM | Recall@5 | 0.5142 | 0.2025 | **+0.3117** [+0.2433, +0.3800] | 0.0001 | *** ($p < 0.001$) | 77W / 8L / 115T |
| A3 (Fine-Tuning) vs A1 BiLSTM | MRR@10 | 0.4827 | 0.1724 | **+0.3103** [+0.2485, +0.3717] | 0.0001 | *** ($p < 0.001$) | 104W / 13L / 83T |
| A3 (Fine-Tuning) vs A2 Linear Probe | Recall@5 | 0.5142 | 0.4117 | **+0.1025** [+0.0492, +0.1575] | 0.0001 | *** ($p < 0.001$) | 40W / 11L / 149T |
| A3 (Fine-Tuning) vs A2 Linear Probe | MRR@10 | 0.4827 | 0.3591 | **+0.1236** [+0.0796, +0.1697] | 0.0001 | *** ($p < 0.001$) | 65W / 21L / 114T |
| A2 (Linear Probe) vs BM25 Baseline | Recall@5 | 0.4117 | 0.4075 | **+0.0042** [-0.0567, +0.0625] | 0.9031 | n.s. (not sig.) | 26W / 25L / 149T |
| A2 (Linear Probe) vs BM25 Baseline | MRR@10 | 0.3591 | 0.3350 | **+0.0241** [-0.0322, +0.0793] | 0.4006 | n.s. (not sig.) | 51W / 42L / 107T |
| **A3 (XLM-R) vs A4 (PrahokBART)** | **Recall@5** | **0.5142** | **0.1517** | **+0.3625** [+0.2925, +0.4300] | **0.0001** | **\*\*\* ($p < 0.001$)** | 83W / 5L / 112T |
| **A3 (XLM-R) vs A4 (PrahokBART)** | **MRR@10** | **0.4827** | **0.1035** | **+0.3792** [+0.3195, +0.4396] | **0.0001** | **\*\*\* ($p < 0.001$)** | 116W / 6L / 78T |
| A4 (PrahokBART) vs BM25 Baseline | Recall@5 | 0.1517 | 0.4075 | **-0.2558** [-0.3208, -0.1925] | 0.0001 | *** ($p < 0.001$) | 8W / 65L / 127T |
| A4 (PrahokBART) vs BM25 Baseline | MRR@10 | 0.1035 | 0.3350 | **-0.2315** [-0.2869, -0.1775] | 0.0001 | *** ($p < 0.001$) | 12W / 90L / 98T |

---

## Key Conclusions

1. **A3 vs BM25**: A3's advantage over the sparse baseline is statistically significant ($p < 0.001$) on both Recall@5 ($\Delta = +0.1067$) and MRR@10 ($\Delta = +0.1477$), with 95% bootstrap difference intervals bounded strictly above zero.
2. **A2 vs BM25**: The linear probe matches BM25 performance without a statistically significant difference ($p = 0.9031$ on Recall@5; $p = 0.4006$ on MRR@10). Adapting only the linear head is insufficient to surpass keyword matching on colloquial queries.
3. **A3 vs A4**: Full fine-tuning of XLM-R decisively outperforms PrahokBART ($\Delta = +0.3625$ Recall@5, $p < 0.001$), demonstrating that multilingual contrastive pre-training and higher model capacity provide critical advantages on complex legal queries.
