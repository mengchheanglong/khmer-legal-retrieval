"""
Unit tests for shared deep learning infrastructure modules.

Tests:
1. Seed determinism (src/dl/seed.py)
2. Khmer text normalization & tokenization (src/dl/text.py)
3. Dataset loading and collation (src/dl/data.py)
4. InfoNCE contrastive loss (src/dl/losses.py)
5. Information Retrieval metrics & bootstrap CIs (src/dl/metrics.py)
6. Atomic checkpointing & resume (src/dl/checkpoint.py)
7. Evaluation harness with a mock retriever (src/dl/evaluate.py)
"""

import math
from pathlib import Path
import random
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.dl.checkpoint import load_checkpoint, save_checkpoint
from src.dl.data import (
    CorpusDataset,
    LegalPairsDataset,
    QABenchmarkDataset,
    collate_pairs,
    generate_doc_id,
)
from src.dl.evaluate import BaseRetriever, EvaluationHarness
from src.dl.losses import InfoNCELoss
from src.dl.metrics import (
    bootstrap_ci,
    compute_hit_at_k,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_recall_at_k,
    evaluate_rankings,
)
from src.dl.seed import get_device, set_seed
from src.dl.text import detokenize_khmer, normalize_khmer_text, tokenize_khmer


# =========================================================================
# 1. Seed & Device Tests
# =========================================================================
class TestSeedAndDevice:
    """Tests for deterministic execution and device configuration."""

    def test_set_seed_reproducibility(self) -> None:
        """Verify identical random numbers generated after set_seed."""
        set_seed(42)
        r1 = [random.random() for _ in range(5)]
        n1 = np.random.rand(5)
        t1 = torch.rand(5)

        set_seed(42)
        r2 = [random.random() for _ in range(5)]
        n2 = np.random.rand(5)
        t2 = torch.rand(5)

        assert r1 == r2
        assert np.allclose(n1, n2)
        assert torch.allclose(t1, t2)

    def test_get_device_returns_torch_device(self) -> None:
        """get_device should return a valid torch.device instance."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cpu", "cuda")


# =========================================================================
# 2. Text Normalization & Tokenization Tests
# =========================================================================
class TestKhmerText:
    """Tests for Khmer text preprocessing and word segmentation."""

    def test_normalize_khmer_zero_width_chars(self) -> None:
        """Zero-width spaces (U+200B, U+200C, U+200D, U+FEFF) must be removed."""
        text = "ក្រម\u200bរដ្ឋប្បវេណី\u200cនៃ\u200dកម្ពុជា\ufeff"
        cleaned = normalize_khmer_text(text)
        assert "\u200b" not in cleaned
        assert "\u200c" not in cleaned
        assert "\u200d" not in cleaned
        assert "\ufeff" not in cleaned
        assert cleaned == "ក្រមរដ្ឋប្បវេណីនៃកម្ពុជា"

    def test_normalize_khmer_whitespace_collapse(self) -> None:
        """Redundant spaces and tabs must be collapsed to single spaces."""
        text = "  មាត្រា   ១.-    ច្បាប់  "
        cleaned = normalize_khmer_text(text)
        assert cleaned == "មាត្រា ១.- ច្បាប់"

    def test_normalize_khmer_empty_and_none(self) -> None:
        """Empty or None input must return empty string without error."""
        assert normalize_khmer_text("") == ""
        assert normalize_khmer_text(None) == ""

    def test_tokenize_khmer_returns_tokens(self) -> None:
        """tokenize_khmer should segment text into a list of word tokens."""
        text = "ក្រមរដ្ឋប្បវេណីនៃព្រះរាជាណាចក្រកម្ពុជា"
        tokens = tokenize_khmer(text)
        assert isinstance(tokens, list)
        assert len(tokens) >= 3
        # Should not include empty strings
        assert all(len(t) > 0 for t in tokens)

    def test_detokenize_khmer(self) -> None:
        """detokenize_khmer should join tokens with single spaces."""
        tokens = ["ក្រម", "រដ្ឋប្បវេណី", "នៃ", "កម្ពុជា"]
        joined = detokenize_khmer(tokens)
        assert joined == "ក្រម រដ្ឋប្បវេណី នៃ កម្ពុជា"


# =========================================================================
# 3. Dataset & Data Loading Tests
# =========================================================================
class TestDataLoaders:
    """Tests for Dataset and DataLoader loading."""

    @pytest.fixture
    def base_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    def test_legal_pairs_dataset_train(self, base_dir: Path) -> None:
        """LegalPairsDataset must load train.jsonl accurately."""
        train_path = base_dir / "data" / "05_splits" / "train.jsonl"
        ds = LegalPairsDataset(train_path)
        assert len(ds) == 1176
        item = ds[0]
        assert "query" in item and len(item["query"]) > 0
        assert "passage" in item and len(item["passage"]) > 0
        assert not item["passage"].startswith("មាត្រា")
        assert "doc_id" in item

    def test_corpus_dataset(self, base_dir: Path) -> None:
        """CorpusDataset must load all 1,976 articles."""
        corpus_path = base_dir / "data" / "05_splits" / "corpus.jsonl"
        corpus = CorpusDataset(corpus_path)
        assert len(corpus) == 1976
        assert len(corpus.doc_ids) == 1976

        doc = corpus.get_by_doc_id("civil_code_kh_art_1")
        assert doc["article_number"] == 1
        assert doc["law_name"] == "Civil Code 2007"
        assert len(doc["text"]) > 0

    def test_qa_benchmark_dataset(self, base_dir: Path) -> None:
        """QABenchmarkDataset must load all 200 benchmark questions."""
        qa_path = base_dir / "tests" / "evaluation" / "ground_truth_qa_kh.json"
        qa_ds = QABenchmarkDataset(qa_path)
        assert len(qa_ds) == 200
        assert len(qa_ds.questions) == 200
        assert len(qa_ds.ground_truth_doc_ids) == 200
        assert all(len(gt) >= 1 for gt in qa_ds.ground_truth_doc_ids)

    def test_collate_pairs(self) -> None:
        """collate_pairs must aggregate dictionaries into batch lists."""
        sample_batch = [
            {"id": "1", "query": "q1", "passage": "p1", "doc_id": "d1", "law_name": "L1", "article_number": 1},
            {"id": "2", "query": "q2", "passage": "p2", "doc_id": "d2", "law_name": "L2", "article_number": 2},
        ]
        batched = collate_pairs(sample_batch)
        assert batched["queries"] == ["q1", "q2"]
        assert batched["passages"] == ["p1", "p2"]
        assert batched["doc_ids"] == ["d1", "d2"]


# =========================================================================
# 4. InfoNCE Loss Tests
# =========================================================================
class TestInfoNCELoss:
    """Tests for contrastive InfoNCE loss."""

    def test_infonce_loss_exact_computation(self) -> None:
        """Validate InfoNCE loss value against analytical expectation."""
        loss_fn = InfoNCELoss(temperature=1.0, symmetric=False, normalize_embeddings=False)
        # Construct orthogonal queries and passages with known dot products
        # Q = [[1, 0], [0, 1]]
        # P = [[1, 0], [0, 1]]
        # Sim matrix S = [[1, 0], [0, 1]]
        # Loss q0: -log(exp(1) / (exp(1) + exp(0))) = -log(e / (e + 1))
        # Loss q1: -log(exp(1) / (exp(1) + exp(0)))
        q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        p = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        loss = loss_fn(q, p)
        expected = -math.log(math.e / (math.e + 1.0))
        assert abs(loss.item() - expected) < 1e-5

    def test_infonce_loss_temperature_scaling(self) -> None:
        """Lower temperature should sharpen probabilities and reduce loss on well-separated pairs."""
        q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        p = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        loss_t1 = InfoNCELoss(temperature=1.0)(q, p)
        loss_t01 = InfoNCELoss(temperature=0.1)(q, p)
        assert loss_t01 < loss_t1

    def test_infonce_loss_symmetry(self) -> None:
        """Symmetric loss should equal single direction when similarity is symmetric."""
        loss_sym = InfoNCELoss(temperature=0.05, symmetric=True)
        loss_asym = InfoNCELoss(temperature=0.05, symmetric=False)

        q = torch.randn(4, 16)
        # Using same vectors guarantees symmetric dot product matrix
        loss1 = loss_sym(q, q)
        loss2 = loss_asym(q, q)
        assert torch.allclose(loss1, loss2)

    def test_infonce_loss_dimension_mismatch_raises(self) -> None:
        """Mismatched query/passage shapes must raise ValueError."""
        loss_fn = InfoNCELoss()
        with pytest.raises(ValueError, match="Shape mismatch"):
            loss_fn(torch.randn(4, 16), torch.randn(5, 16))


# =========================================================================
# 5. Metrics & Bootstrap CI Tests
# =========================================================================
class TestMetricsAndCI:
    """Tests for ranking metrics and confidence interval computation."""

    def test_compute_recall_at_k(self) -> None:
        ranked = ["d1", "d2", "d3", "d4"]
        assert compute_recall_at_k(ranked, {"d1"}, k=1) == 1.0
        assert compute_recall_at_k(ranked, {"d1"}, k=5) == 1.0
        assert compute_recall_at_k(ranked, {"d2"}, k=1) == 0.0
        assert compute_recall_at_k(ranked, {"d2"}, k=2) == 1.0
        assert compute_recall_at_k(ranked, {"d1", "d3"}, k=2) == 0.5
        assert compute_recall_at_k(ranked, {"d1", "d3"}, k=3) == 1.0

    def test_compute_mrr_at_k(self) -> None:
        ranked = ["d1", "d2", "d3"]
        assert compute_mrr_at_k(ranked, {"d1"}, k=10) == 1.0
        assert compute_mrr_at_k(ranked, {"d2"}, k=10) == 0.5
        assert compute_mrr_at_k(ranked, {"d3"}, k=10) == 1.0 / 3.0
        assert compute_mrr_at_k(ranked, {"d4"}, k=10) == 0.0

    def test_compute_ndcg_at_k(self) -> None:
        # Single target at rank 1: nDCG = 1.0
        assert compute_ndcg_at_k(["d1", "d2"], {"d1"}, k=2) == 1.0
        # Single target at rank 2: DCG = 1/log2(3), IDCG = 1/log2(2) = 1.0
        expected = 1.0 / math.log2(3)
        assert abs(compute_ndcg_at_k(["d2", "d1"], {"d1"}, k=2) - expected) < 1e-5

    def test_compute_hit_at_k(self) -> None:
        ranked = ["d1", "d2", "d3"]
        assert compute_hit_at_k(ranked, {"d1"}, k=1) == 1.0
        assert compute_hit_at_k(ranked, {"d3"}, k=2) == 0.0
        assert compute_hit_at_k(ranked, {"d3"}, k=3) == 1.0

    def test_bootstrap_ci_coverage(self) -> None:
        """Bootstrap CI must bracket the sample mean."""
        scores = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.5, 0.75]
        ci = bootstrap_ci(scores, n_bootstrap=1000, seed=42)
        assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]
        assert 0.0 <= ci["ci_lower"]
        assert ci["ci_upper"] <= 1.0

    def test_evaluate_rankings(self) -> None:
        """evaluate_rankings must compute all 6 metrics with CIs."""
        rankings = [["d1", "d2"], ["d2", "d1"]]
        gt = [{"d1"}, {"d1"}]
        results = evaluate_rankings(rankings, gt, seed=42)

        for metric in ["Recall@1", "Recall@5", "Recall@10", "MRR@10", "nDCG@10", "Hit@5"]:
            assert metric in results
            assert "mean" in results[metric]
            assert "ci_lower" in results[metric]
            assert "ci_upper" in results[metric]


# =========================================================================
# 6. Checkpoint Tests
# =========================================================================
class TestCheckpointing:
    """Tests for saving and loading PyTorch checkpoints."""

    def test_save_and_load_checkpoint_roundtrip(self, tmp_path: Path) -> None:
        model1 = nn.Linear(10, 5)
        opt1 = AdamW(model1.parameters(), lr=1e-3)
        ckpt_path = tmp_path / "test_model.pt"

        # Save checkpoint
        save_checkpoint(
            ckpt_path,
            model=model1,
            optimizer=opt1,
            epoch=5,
            best_val_mrr=0.825,
            config={"model_name": "test"},
        )
        assert ckpt_path.exists()

        # Load into separate model
        model2 = nn.Linear(10, 5)
        opt2 = AdamW(model2.parameters(), lr=1e-3)
        meta = load_checkpoint(ckpt_path, model=model2, optimizer=opt2)

        assert meta["epoch"] == 5
        assert meta["best_val_mrr"] == 0.825
        assert meta["config"]["model_name"] == "test"

        # Verify weights match exactly
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)


# =========================================================================
# 7. Evaluation Harness with Mock Retriever
# =========================================================================
class MockRetriever(BaseRetriever):
    """Mock retriever returning ground-truth top document for testing harness."""

    def __init__(self, ground_truth_map: dict[str, str], default_id: str = "civil_code_kh_art_1") -> None:
        self.gt_map = ground_truth_map
        self.default_id = default_id

    def search(self, queries: list[str], top_k: int = 10) -> list[list[str]]:
        results = []
        for q in queries:
            target = self.gt_map.get(q, self.default_id)
            results.append([target] + [self.default_id] * (top_k - 1))
        return results


class TestEvaluationHarness:
    """Tests for the unified evaluation harness."""

    @pytest.fixture
    def base_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    def test_harness_evaluates_mock_retriever(self, base_dir: Path) -> None:
        harness = EvaluationHarness(
            corpus_path=base_dir / "data" / "05_splits" / "corpus.jsonl",
            qa_path=base_dir / "tests" / "evaluation" / "ground_truth_qa_kh.json",
            test_titles_path=base_dir / "data" / "05_splits" / "test_titles.jsonl",
            val_path=base_dir / "data" / "05_splits" / "val.jsonl",
        )

        # Build mock retriever that knows 50% of the benchmark questions
        gt_map = {}
        for i, r in enumerate(harness.qa.records):
            if i % 2 == 0:
                gt_map[r["question"]] = r["expected_doc_ids"][0]

        mock = MockRetriever(gt_map)
        res = harness.evaluate_qa(mock, top_k=5)

        assert res["total_questions"] == 200
        assert "overall" in res
        assert "civil_code" in res
        assert "criminal_code" in res
        # Recall@1 should be ~0.50
        assert 0.40 <= res["overall"]["Recall@1"]["mean"] <= 0.60
