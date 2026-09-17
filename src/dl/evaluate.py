"""
Shared evaluation harness for Khmer legal retrieval models.

Enforces uniform benchmark conditions across all approaches (BM25, A1 BiLSTM,
A2 frozen XLM-R probe, A3 fine-tuned XLM-R):
1. Indexes the exact same 1,976-article Khmer legal corpus (data/05_splits/corpus.jsonl).
2. Evaluates without statute filtering (models must distinguish Civil vs Criminal law).
3. Evaluates both:
   - Primary Benchmark (T-Q): 200 human-verified Khmer legal questions.
   - Secondary Benchmark (T-T): 148 held-out article title pairs.
4. Computes Recall@1/5/10, MRR@10, nDCG@10, Hit@5 with 95% bootstrap confidence intervals.
5. Produces per-code breakdowns (Civil Code 2007 vs Criminal Code 2009).
6. Exports structured JSON metrics to results/metrics/<run_name>.json.
"""

from abc import ABC, abstractmethod
import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from src.config.logging import get_logger, setup_logging
from src.dl.data import CorpusDataset, LegalPairsDataset, QABenchmarkDataset
from src.dl.metrics import evaluate_rankings
from src.dl.seed import get_device, set_seed
from src.dl.text import tokenize_khmer

setup_logging()
logger = get_logger(__name__)


class BaseRetriever(ABC):
    """Abstract base class for all retrieval approaches."""

    @abstractmethod
    def search(
        self,
        queries: Sequence[str],
        top_k: int = 10,
    ) -> list[list[str]]:
        """Retrieve top-k ranked document IDs for each query.

        Args:
            queries: List of query text strings.
            top_k: Number of highest-ranked documents to return per query.

        Returns:
            List of ranked doc_id lists, one per input query.
        """
        ...


class DenseRetriever(BaseRetriever):
    """Retriever wrapper for dual-encoder PyTorch models."""

    def __init__(
        self,
        model: torch.nn.Module,
        corpus: CorpusDataset,
        device: torch.device,
        batch_size: int = 32,
        use_context: bool = False,
    ) -> None:
        """Initialize dense retriever.

        Args:
            model: PyTorch model implementing encode_queries and encode_passages.
            corpus: Unified 1,976-article corpus dataset.
            device: Computing device (CUDA or CPU).
            batch_size: Batch size for corpus and query encoding.
            use_context: Whether to encode corpus with hierarchical context prefixes.
        """
        self.model = model.to(device)
        self.model.eval()
        self.corpus = corpus
        self.device = device
        self.batch_size = batch_size
        self.use_context = use_context

        # Pre-encoded corpus representation: (N, D)
        self.corpus_embeddings: Optional[np.ndarray] = None
        self._doc_ids = corpus.doc_ids

    def index_corpus(self) -> None:
        """Encode all 1,976 corpus articles into an embedding matrix."""
        logger.info(f"Encoding full corpus ({len(self.corpus)} articles)...")
        passages = self.corpus.get_texts(use_context=self.use_context)
        all_embeddings: list[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, len(passages), self.batch_size):
                batch_texts = passages[i : i + self.batch_size]
                if hasattr(self.model, "encode_passages"):
                    emb = self.model.encode_passages(batch_texts, device=self.device)
                else:
                    emb = self.model(batch_texts)
                if isinstance(emb, torch.Tensor):
                    emb = emb.cpu().numpy()
                all_embeddings.append(emb)

        self.corpus_embeddings = np.vstack(all_embeddings)
        # Normalize for cosine similarity
        norms = np.linalg.norm(self.corpus_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        self.corpus_embeddings /= norms
        logger.info(f"Corpus indexed. Embedding matrix shape: {self.corpus_embeddings.shape}")

    def search(
        self,
        queries: Sequence[str],
        top_k: int = 10,
    ) -> list[list[str]]:
        """Retrieve top-k documents by cosine similarity against corpus embeddings."""
        if self.corpus_embeddings is None:
            self.index_corpus()

        all_query_emb: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(queries), self.batch_size):
                batch_queries = queries[i : i + self.batch_size]
                if hasattr(self.model, "encode_queries"):
                    emb = self.model.encode_queries(batch_queries, device=self.device)
                else:
                    emb = self.model(batch_queries)
                if isinstance(emb, torch.Tensor):
                    emb = emb.cpu().numpy()
                all_query_emb.append(emb)

        query_embeddings = np.vstack(all_query_emb)
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        query_embeddings /= norms

        # Cosine similarity matrix: (Q, N)
        similarity_matrix = np.dot(query_embeddings, self.corpus_embeddings.T)

        rankings: list[list[str]] = []
        for i in range(len(queries)):
            scores = similarity_matrix[i]
            # Top-k indices in descending order
            top_indices = np.argsort(-scores)[:top_k]
            rankings.append([self._doc_ids[idx] for idx in top_indices])

        return rankings


class BM25KhmerRetriever(BaseRetriever):
    """Baseline BM25 retriever utilizing khmer-nltk word segmentation."""

    def __init__(
        self,
        corpus: CorpusDataset,
        use_context: bool = False,
    ) -> None:
        """Initialize and index the corpus with BM25.

        Args:
            corpus: CorpusDataset instance.
            use_context: Whether to index text with context headers.
        """
        from rank_bm25 import BM25Okapi

        self.corpus = corpus
        self._doc_ids = corpus.doc_ids
        passages = corpus.get_texts(use_context=use_context)

        logger.info(f"Tokenizing {len(passages)} corpus articles with khmer-nltk...")
        tokenized_corpus = [tokenize_khmer(p) for p in passages]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built successfully.")

    def search(
        self,
        queries: Sequence[str],
        top_k: int = 10,
    ) -> list[list[str]]:
        """Retrieve top-k document IDs for each query using BM25Okapi."""
        rankings: list[list[str]] = []
        for query in queries:
            tokenized_q = tokenize_khmer(query)
            scores = self.bm25.get_scores(tokenized_q)
            top_indices = np.argsort(-scores)[:top_k]
            rankings.append([self._doc_ids[idx] for idx in top_indices])
        return rankings


class MultilingualE5Retriever(BaseRetriever):
    """Zero-shot retriever wrapper for intfloat/multilingual-e5-base."""

    def __init__(
        self,
        corpus: CorpusDataset,
        model_name: str = "intfloat/multilingual-e5-base",
        device: Optional[torch.device] = None,
        batch_size: int = 16,
        max_length: int = 512,
        use_context: bool = False,
    ) -> None:
        """Initialize zero-shot Multilingual-E5 retriever.

        Args:
            corpus: CorpusDataset instance.
            model_name: Hugging Face model identifier.
            device: Computing device (CUDA or CPU).
            batch_size: Batch size for encoding passages and queries.
            max_length: Maximum sequence length.
            use_context: Whether to prepend hierarchical legal context.
        """
        from transformers import AutoModel, AutoTokenizer

        self.corpus = corpus
        self._doc_ids = corpus.doc_ids
        self.device = device or get_device()
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_context = use_context

        logger.info(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.corpus_embeddings: Optional[np.ndarray] = None

    def _encode_texts(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        """Encode texts with the required E5 prefix using mean pooling and L2 normalization."""
        prefixed = [f"{prefix}{t}" for t in texts]
        all_embeddings: list[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, len(prefixed), self.batch_size):
                batch_texts = prefixed[i : i + self.batch_size]
                batch_dict = self.tokenizer(
                    batch_texts,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**batch_dict)
                last_hidden = outputs.last_hidden_state
                mask = batch_dict["attention_mask"]

                # Masked mean pooling over valid tokens
                masked_hidden = last_hidden.masked_fill(~mask[..., None].bool(), 0.0)
                sum_hidden = masked_hidden.sum(dim=1)
                sum_mask = mask.sum(dim=1)[..., None].clamp(min=1e-9)
                pooled = sum_hidden / sum_mask

                # L2 normalize
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_embeddings.append(normalized.cpu().numpy())

        return np.vstack(all_embeddings)

    def index_corpus(self) -> None:
        """Encode full corpus with 'passage: ' prefix into embedding matrix."""
        logger.info(f"Encoding full corpus ({len(self.corpus)} articles) with multilingual-e5-base...")
        passages = self.corpus.get_texts(use_context=self.use_context)
        self.corpus_embeddings = self._encode_texts(passages, prefix="passage: ")
        logger.info(f"Corpus indexed. Embedding matrix shape: {self.corpus_embeddings.shape}")

    def search(
        self,
        queries: Sequence[str],
        top_k: int = 10,
    ) -> list[list[str]]:
        """Retrieve top-k documents by cosine similarity against corpus embeddings."""
        if self.corpus_embeddings is None:
            self.index_corpus()

        query_embeddings = self._encode_texts(queries, prefix="query: ")
        # Cosine similarity matrix: (Q, N) where both vectors are L2-normalized
        similarity_matrix = np.dot(query_embeddings, self.corpus_embeddings.T)

        rankings: list[list[str]] = []
        for i in range(len(queries)):
            scores = similarity_matrix[i]
            top_indices = np.argsort(-scores)[:top_k]
            rankings.append([self._doc_ids[idx] for idx in top_indices])

        return rankings


class EvaluationHarness:
    """The unified evaluation harness for Khmer legal retrieval."""

    def __init__(
        self,
        corpus_path: Path = Path("data/05_splits/corpus.jsonl"),
        qa_path: Path = Path("tests/evaluation/ground_truth_qa_kh.json"),
        test_titles_path: Path = Path("data/05_splits/test_titles.jsonl"),
        val_path: Optional[Path] = Path("data/05_splits/val.jsonl"),
        seed: int = 42,
    ) -> None:
        """Initialize the evaluation harness.

        Args:
            corpus_path: Path to corpus.jsonl.
            qa_path: Path to ground_truth_qa_kh.json (T-Q).
            test_titles_path: Path to test_titles.jsonl (T-T).
            val_path: Optional path to val.jsonl.
            seed: Random seed for bootstrap confidence intervals.
        """
        self.seed = seed
        set_seed(seed)

        self.corpus = CorpusDataset(corpus_path)
        self.qa = QABenchmarkDataset(qa_path)
        self.test_titles = LegalPairsDataset(test_titles_path) if test_titles_path.exists() else None
        self.val_pairs = LegalPairsDataset(val_path) if val_path and val_path.exists() else None

    def evaluate_qa(
        self,
        retriever: BaseRetriever,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Evaluate retriever on the T-Q primary benchmark (200 questions).

        Produces overall metrics and per-code breakdowns (Civil Code vs Criminal Code).

        Args:
            retriever: Retriever instance implementing search().
            top_k: Top-k ranking depth (default: 10).

        Returns:
            Dictionary containing 'overall', 'civil_code', and 'criminal_code' metric suites.
        """
        questions = self.qa.questions
        ground_truth = self.qa.ground_truth_doc_ids

        logger.info(f"Retrieving top-{top_k} documents for {len(questions)} T-Q questions...")
        rankings = retriever.search(questions, top_k=top_k)

        # 1. Overall evaluation
        overall_metrics = evaluate_rankings(rankings, ground_truth, seed=self.seed)

        # 2. Per-code slices
        civ_indices = [i for i, r in enumerate(self.qa.records) if r["expected_law"] == "Civil Code 2007"]
        crim_indices = [i for i, r in enumerate(self.qa.records) if r["expected_law"] == "Criminal Code 2009"]

        civ_rankings = [rankings[i] for i in civ_indices]
        civ_gt = [ground_truth[i] for i in civ_indices]
        civ_metrics = evaluate_rankings(civ_rankings, civ_gt, seed=self.seed)

        crim_rankings = [rankings[i] for i in crim_indices]
        crim_gt = [ground_truth[i] for i in crim_indices]
        crim_metrics = evaluate_rankings(crim_rankings, crim_gt, seed=self.seed)

        return {
            "total_questions": len(questions),
            "overall": overall_metrics,
            "civil_code": {
                "count": len(civ_indices),
                "metrics": civ_metrics,
            },
            "criminal_code": {
                "count": len(crim_indices),
                "metrics": crim_metrics,
            },
        }

    def evaluate_test_titles(
        self,
        retriever: BaseRetriever,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Evaluate retriever on the T-T secondary benchmark (148 held-out titles).

        Args:
            retriever: Retriever instance.
            top_k: Top-k ranking cutoff (default: 10).

        Returns:
            Dictionary containing evaluation metrics and per-code breakdown.
        """
        if self.test_titles is None:
            raise FileNotFoundError("test_titles.jsonl not available for evaluation")

        queries = [r["query"] for r in self.test_titles.records]
        ground_truth = [{r["doc_id"]} for r in self.test_titles.records]

        logger.info(f"Retrieving top-{top_k} documents for {len(queries)} T-T titles...")
        rankings = retriever.search(queries, top_k=top_k)

        overall_metrics = evaluate_rankings(rankings, ground_truth, seed=self.seed)

        civ_indices = [i for i, r in enumerate(self.test_titles.records) if r["law_name"] == "Civil Code 2007"]
        crim_indices = [i for i, r in enumerate(self.test_titles.records) if r["law_name"] == "Criminal Code 2009"]

        civ_metrics = evaluate_rankings(
            [rankings[i] for i in civ_indices],
            [ground_truth[i] for i in civ_indices],
            seed=self.seed,
        )
        crim_metrics = evaluate_rankings(
            [rankings[i] for i in crim_indices],
            [ground_truth[i] for i in crim_indices],
            seed=self.seed,
        )

        return {
            "total_pairs": len(queries),
            "overall": overall_metrics,
            "civil_code": {
                "count": len(civ_indices),
                "metrics": civ_metrics,
            },
            "criminal_code": {
                "count": len(crim_indices),
                "metrics": crim_metrics,
            },
        }

    def evaluate_validation(
        self,
        retriever: BaseRetriever,
        top_k: int = 10,
    ) -> dict[str, dict[str, float]]:
        """Quick validation evaluation during training loop (for model selection).

        Args:
            retriever: Retriever instance.
            top_k: Cutoff rank (default: 10).

        Returns:
            Dictionary mapping metric names to {'mean', 'ci_lower', 'ci_upper'}.
        """
        if self.val_pairs is None:
            raise FileNotFoundError("val.jsonl not available")

        queries = [r["query"] for r in self.val_pairs.records]
        ground_truth = [{r["doc_id"]} for r in self.val_pairs.records]

        rankings = retriever.search(queries, top_k=top_k)
        return evaluate_rankings(rankings, ground_truth, n_bootstrap=200, seed=self.seed)

    def run_full_evaluation(
        self,
        retriever: BaseRetriever,
        run_name: str,
        output_dir: Path = Path("results/metrics"),
    ) -> dict[str, Any]:
        """Run complete evaluation across T-Q and T-T, log results, and save JSON.

        Args:
            retriever: Retriever instance to evaluate.
            run_name: Name of the run (e.g. 'bm25_baseline', 'a1_bilstm', etc.).
            output_dir: Directory to save results JSON.

        Returns:
            Dictionary with complete evaluation results.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running full evaluation for '{run_name}'...")
        tq_results = self.evaluate_qa(retriever, top_k=10)
        tt_results = self.evaluate_test_titles(retriever, top_k=10) if self.test_titles else None

        results = {
            "run_name": run_name,
            "tq_benchmark": tq_results,
            "tt_benchmark": tt_results,
        }

        output_file = output_dir / f"{run_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved evaluation metrics -> {output_file}")

        # Print summary table
        print("\n" + "=" * 78)
        print(f"EVALUATION RESULTS: {run_name.upper()}")
        print("=" * 78)
        print("PRIMARY BENCHMARK (T-Q: 200 human-verified questions):")
        print(f"{'Metric':<12} | {'Overall (95% CI)':<25} | {'Civil Code (100)':<18} | {'Criminal Code (100)':<18}")
        print("-" * 78)
        for m in ["Recall@1", "Recall@5", "Recall@10", "MRR@10", "nDCG@10", "Hit@5"]:
            ov = tq_results["overall"][m]
            cv = tq_results["civil_code"]["metrics"][m]
            cr = tq_results["criminal_code"]["metrics"][m]
            ov_str = f"{ov['mean']:.4f} [{ov['ci_lower']:.4f}, {ov['ci_upper']:.4f}]"
            cv_str = f"{cv['mean']:.4f}"
            cr_str = f"{cr['mean']:.4f}"
            print(f"{m:<12} | {ov_str:<25} | {cv_str:<18} | {cr_str:<18}")

        if tt_results:
            print("\n" + "-" * 78)
            print("SECONDARY BENCHMARK (T-T: 148 held-out article titles):")
            print(f"{'Metric':<12} | {'Overall (95% CI)':<25} | {'Civil Code (99)':<18} | {'Criminal Code (49)':<18}")
            print("-" * 78)
            for m in ["Recall@1", "Recall@5", "Recall@10", "MRR@10", "nDCG@10", "Hit@5"]:
                ov = tt_results["overall"][m]
                cv = tt_results["civil_code"]["metrics"][m]
                cr = tt_results["criminal_code"]["metrics"][m]
                ov_str = f"{ov['mean']:.4f} [{ov['ci_lower']:.4f}, {ov['ci_upper']:.4f}]"
                cv_str = f"{cv['mean']:.4f}"
                cr_str = f"{cr['mean']:.4f}"
                print(f"{m:<12} | {ov_str:<25} | {cv_str:<18} | {cr_str:<18}")

        print("=" * 78 + "\n")
        return results


def main() -> None:
    """CLI for running evaluation harness."""
    parser = argparse.ArgumentParser(description="Evaluate retrieval models on Khmer legal benchmarks.")
    parser.add_argument(
        "--model",
        type=str,
        default="bm25",
        choices=["bm25", "e5_zero_shot", "a1"],
        help="Model to evaluate (default: bm25).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to model checkpoint (.pt) for neural models.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for the evaluation run output (defaults to model name).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/metrics"),
        help="Directory to save evaluation results.",
    )

    args = parser.parse_args()
    run_name = args.run_name or (f"{args.model}_baseline" if args.model == "bm25" else args.model)

    harness = EvaluationHarness()
    if args.model == "bm25":
        retriever = BM25KhmerRetriever(harness.corpus)
        harness.run_full_evaluation(retriever, run_name=run_name, output_dir=args.output_dir)
    elif args.model in ("e5_zero_shot", "e5"):
        device = get_device()
        retriever = MultilingualE5Retriever(harness.corpus, device=device)
        harness.run_full_evaluation(retriever, run_name=run_name, output_dir=args.output_dir)
    elif args.model == "a1":
        from src.dl.checkpoint import load_checkpoint
        from src.dl.models.bilstm import BiLSTMEncoder
        from src.dl.models.vocab import KhmerVocab

        ckpt_path = args.checkpoint or Path("results/checkpoints/a1_bilstm/best.pt")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"A1 checkpoint not found: {ckpt_path}")

        # Check for saved vocabulary
        vocab_path = Path("data/05_splits/vocab_kh.json")
        if vocab_path.exists():
            vocab = KhmerVocab.load(vocab_path)
        else:
            vocab = KhmerVocab.build_from_train_file("data/05_splits/train.jsonl")

        device = get_device()
        model = BiLSTMEncoder(vocab=vocab)
        load_checkpoint(ckpt_path, model=model, device=device)

        retriever = DenseRetriever(model, harness.corpus, device=device, batch_size=32)
        harness.run_full_evaluation(retriever, run_name=run_name, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

