"""
Hyperparameter tuning and training pipeline for Approach A1 (BiLSTM dual encoder).

Executes 2x2 grid search:
  LR in {1e-3, 3e-3} x Dropout in {0.1, 0.3}

For each configuration:
1. Trains BiLSTM from scratch with InfoNCE loss (tau=0.05).
2. Logs per-epoch train/val loss and val MRR@10 to results/logs/a1_*.csv.
3. Saves model checkpoints (best.pt and last.pt).
4. Records summary metrics to results/tuning/a1.csv.

After grid search:
5. Selects winning checkpoint on validation MRR@10.
6. Evaluates the best model once on full corpus for T-Q and T-T benchmarks.
7. Saves final evaluation metrics to results/metrics/a1_bilstm.json.
"""

import csv
import json
from pathlib import Path
import shutil
import time
from typing import Any

import pandas as pd
import torch

from src.config.logging import get_logger, setup_logging
from src.dl.checkpoint import load_checkpoint
from src.dl.data import CorpusDataset, LegalPairsDataset
from src.dl.evaluate import DenseRetriever, EvaluationHarness
from src.dl.models.bilstm import BiLSTMEncoder
from src.dl.models.vocab import KhmerVocab
from src.dl.seed import get_device, set_seed

setup_logging()
logger = get_logger(__name__)


def run_tuning(
    splits_dir: Path = Path("data/05_splits"),
    results_dir: Path = Path("results"),
    epochs: int = 15,
    patience: int = 4,
    batch_size: int = 32,
    seed: int = 42,
) -> Path:
    """Execute hyperparameter tuning grid for BiLSTM.

    Args:
        splits_dir: Directory containing split JSONL files.
        results_dir: Directory where results/logs/checkpoints are stored.
        epochs: Maximum training epochs per configuration.
        patience: Early stopping patience based on validation MRR@10.
        batch_size: Mini-batch size.
        seed: Random seed for reproducibility.

    Returns:
        Path to the best checkpoint file.
    """
    set_seed(seed)
    device = get_device()

    train_path = splits_dir / "train.jsonl"
    val_path = splits_dir / "val.jsonl"
    corpus_path = splits_dir / "corpus.jsonl"
    vocab_path = splits_dir / "vocab_kh.json"

    # 1. Build or load vocabulary from train.jsonl
    if vocab_path.exists():
        vocab = KhmerVocab.load(vocab_path)
    else:
        vocab = KhmerVocab.build_from_train_file(train_path, min_freq=1)
        vocab.save(vocab_path)

    # 2. Load datasets with pre-encoding for rapid batching
    train_dataset = LegalPairsDataset(train_path, vocab=vocab)
    val_dataset = LegalPairsDataset(val_path, vocab=vocab)
    corpus_dataset = CorpusDataset(corpus_path)

    # 3. Tuning grid: LR {1e-3, 3e-3} x Dropout {0.1, 0.3}
    grid = [
        {"lr": 1e-3, "dropout": 0.1},
        {"lr": 1e-3, "dropout": 0.3},
        {"lr": 3e-3, "dropout": 0.1},
        {"lr": 3e-3, "dropout": 0.3},
    ]

    tuning_dir = results_dir / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    tuning_csv = tuning_dir / "a1.csv"

    tuning_rows: list[dict[str, Any]] = []

    logger.info(f"Starting A1 BiLSTM hyperparameter grid search ({len(grid)} configurations)...")

    for i, params in enumerate(grid, 1):
        lr = params["lr"]
        dropout = params["dropout"]
        run_name = f"a1_lr{lr:.0e}_drop{dropout}".replace("-0", "-")

        logger.info(f"\n[{i}/{len(grid)}] Training configuration: {run_name} (LR={lr}, Dropout={dropout})")
        set_seed(seed)

        # Instantiate fresh model
        model = BiLSTMEncoder(
            vocab=vocab,
            embed_dim=300,
            hidden_dim=256,
            dropout=dropout,
            max_seq_len=256,
        )

        config = {
            "run_name": run_name,
            "approach": "A1",
            "model_type": "bilstm",
            "lr": lr,
            "dropout": dropout,
            "embed_dim": 300,
            "hidden_dim": 256,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "temperature": 0.05,
            "seed": seed,
            "logs_dir": str(results_dir / "logs"),
            "checkpoints_dir": str(results_dir / "checkpoints" / run_name),
            "metrics_dir": str(results_dir / "metrics"),
        }

        from src.dl.train import RetrieverTrainer

        trainer = RetrieverTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            corpus_dataset=corpus_dataset,
            config=config,
            device=device,
        )

        telemetry = trainer.fit()

        best_mrr = telemetry["best_val_mrr10"]
        # Find best epoch details
        best_epoch_info = max(telemetry["history"], key=lambda h: h["val_mrr10"])

        tuning_rows.append({
            "run_name": run_name,
            "lr": lr,
            "dropout": dropout,
            "trainable_params": telemetry["trainable_parameters"],
            "best_epoch": best_epoch_info["epoch"],
            "best_val_mrr10": best_mrr,
            "val_loss": round(best_epoch_info["val_loss"], 4),
            "train_loss": round(best_epoch_info["train_loss"], 4),
            "train_time_sec": telemetry["total_wall_clock_seconds"],
            "checkpoint_path": str(Path(config["checkpoints_dir"]) / "best.pt"),
        })

    # Save tuning summary table
    df_tuning = pd.DataFrame(tuning_rows)
    df_tuning.sort_values(by="best_val_mrr10", ascending=False, inplace=True)
    df_tuning.to_csv(tuning_csv, index=False)
    logger.info(f"Saved tuning summary table -> {tuning_csv}")

    print("\n" + "=" * 80)
    print("A1 BiLSTM HYPERPARAMETER TUNING SUMMARY")
    print("=" * 80)
    print(df_tuning.to_string(index=False))
    print("=" * 80 + "\n")

    # 4. Copy best checkpoint to primary A1 location
    best_config = df_tuning.iloc[0]
    best_ckpt_src = Path(best_config["checkpoint_path"])
    best_ckpt_dest = results_dir / "checkpoints" / "a1_bilstm" / "best.pt"
    best_ckpt_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_ckpt_src, best_ckpt_dest)
    logger.info(
        f"Selected winning model: '{best_config['run_name']}' "
        f"(Val MRR@10: {best_config['best_val_mrr10']:.4f}) -> {best_ckpt_dest}"
    )

    # 5. Evaluate winning model once on full corpus across T-Q and T-T
    logger.info("Evaluating selected A1 model on primary (T-Q) and secondary (T-T) benchmarks...")
    harness = EvaluationHarness(
        corpus_path=corpus_path,
        qa_path=splits_dir.parent.parent / "tests" / "evaluation" / "ground_truth_qa_kh.json",
        test_titles_path=splits_dir / "test_titles.jsonl",
        seed=seed,
    )

    best_model = BiLSTMEncoder(
        vocab=vocab,
        embed_dim=300,
        hidden_dim=256,
        dropout=float(best_config["dropout"]),
        max_seq_len=256,
    )
    load_checkpoint(best_ckpt_dest, model=best_model, device=device)

    retriever = DenseRetriever(best_model, harness.corpus, device=device, batch_size=batch_size)
    harness.run_full_evaluation(retriever, run_name="a1_bilstm", output_dir=results_dir / "metrics")

    return best_ckpt_dest


if __name__ == "__main__":
    run_tuning()
