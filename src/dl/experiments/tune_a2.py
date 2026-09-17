"""
Hyperparameter tuning and training pipeline for Approach A2 (XLM-R frozen + linear probe).

Executes 2x2 grid search:
  LR in {1e-3, 1e-2} x Weight Decay in {0.0, 0.01}

Pipeline:
1. Pre-computes and caches frozen XLM-R backbone representations for train.jsonl and val.jsonl.
2. Runs 2x2 grid search with InfoNCE loss (tau=0.05).
3. Logs per-epoch train/val loss and val MRR@10 to results/logs/a2_*.csv.
4. Saves checkpoints (best.pt and last.pt) and training telemetry.
5. Exports summary table to results/tuning/a2.csv.
6. Selects the winning model based on validation MRR@10.
7. Evaluates the winning model once on full corpus across T-Q and T-T benchmarks.
8. Saves final evaluation metrics to results/metrics/a2_linear_probe.json.
"""

import csv
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from src.config.logging import get_logger, setup_logging
from src.dl.checkpoint import load_checkpoint, save_checkpoint
from src.dl.data import CorpusDataset, LegalPairsDataset
from src.dl.evaluate import DenseRetriever, EvaluationHarness
from src.dl.losses import InfoNCELoss
from src.dl.metrics import evaluate_rankings
from src.dl.models.linear_probe import XLMRLinearProbeRetriever
from src.dl.seed import get_device, set_seed

setup_logging()
logger = get_logger(__name__)


class CachedPairEmbeddingsDataset(Dataset):
    """In-memory dataset of pre-computed query and passage backbone embeddings."""

    def __init__(
        self,
        query_emb: torch.Tensor,
        passage_emb: torch.Tensor,
        doc_ids: list[str],
    ) -> None:
        self.query_emb = query_emb
        self.passage_emb = passage_emb
        self.doc_ids = doc_ids

    def __len__(self) -> int:
        return len(self.doc_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "query_emb": self.query_emb[idx],
            "passage_emb": self.passage_emb[idx],
            "doc_id": self.doc_ids[idx],
        }


def collate_cached(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "query_emb": torch.stack([b["query_emb"] for b in batch]),
        "passage_emb": torch.stack([b["passage_emb"] for b in batch]),
        "doc_ids": [b["doc_id"] for b in batch],
    }


def get_or_create_embedding_cache(
    splits_dir: Path,
    cache_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pre-compute and cache frozen XLM-R backbone embeddings for fast training."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache_path = cache_dir / "train_backbone_emb.pt"
    val_cache_path = cache_dir / "val_backbone_emb.pt"

    train_path = splits_dir / "train.jsonl"
    val_path = splits_dir / "val.jsonl"

    train_ds = LegalPairsDataset(train_path)
    val_ds = LegalPairsDataset(val_path)

    train_doc_ids = [r["doc_id"] for r in train_ds.records]
    val_doc_ids = [r["doc_id"] for r in val_ds.records]

    if train_cache_path.exists() and val_cache_path.exists():
        logger.info(f"Loading cached backbone embeddings from {cache_dir}...")
        train_data = torch.load(train_cache_path, weights_only=True)
        val_data = torch.load(val_cache_path, weights_only=True)
        return train_data, val_data

    logger.info("Generating frozen backbone embedding cache (one-time extraction)...")
    model = XLMRLinearProbeRetriever(device=device)

    # 1. Train pairs extraction
    logger.info(f"Extracting {len(train_ds)} train query embeddings...")
    train_q = model.extract_backbone_embeddings(
        [r["query"] for r in train_ds.records], prefix="query: ", batch_size=32, device=device
    )
    logger.info(f"Extracting {len(train_ds)} train passage embeddings...")
    train_p = model.extract_backbone_embeddings(
        [r["passage"] for r in train_ds.records], prefix="passage: ", batch_size=32, device=device
    )
    train_data = {
        "query_emb": train_q,
        "passage_emb": train_p,
        "doc_ids": train_doc_ids,
    }
    torch.save(train_data, train_cache_path)
    logger.info(f"Saved train backbone cache -> {train_cache_path}")

    # 2. Val pairs extraction
    logger.info(f"Extracting {len(val_ds)} val query embeddings...")
    val_q = model.extract_backbone_embeddings(
        [r["query"] for r in val_ds.records], prefix="query: ", batch_size=32, device=device
    )
    logger.info(f"Extracting {len(val_ds)} val passage embeddings...")
    val_p = model.extract_backbone_embeddings(
        [r["passage"] for r in val_ds.records], prefix="passage: ", batch_size=32, device=device
    )
    val_data = {
        "query_emb": val_q,
        "passage_emb": val_p,
        "doc_ids": val_doc_ids,
    }
    torch.save(val_data, val_cache_path)
    logger.info(f"Saved val backbone cache -> {val_cache_path}")

    return train_data, val_data


def train_linear_probe_run(
    run_name: str,
    lr: float,
    weight_decay: float,
    train_data: dict[str, Any],
    val_data: dict[str, Any],
    results_dir: Path,
    epochs: int = 15,
    patience: int = 4,
    batch_size: int = 32,
    temperature: float = 0.05,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    """Execute training for one linear probe hyperparameter configuration."""
    set_seed(seed)

    checkpoints_dir = results_dir / "checkpoints" / run_name
    logs_dir = results_dir / "logs"
    metrics_dir = results_dir / "metrics"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    csv_log_path = logs_dir / f"{run_name}.csv"

    train_ds = CachedPairEmbeddingsDataset(
        train_data["query_emb"], train_data["passage_emb"], train_data["doc_ids"]
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_cached
    )

    val_q_all = val_data["query_emb"].to(device)
    val_p_all = val_data["passage_emb"].to(device)
    val_doc_ids = val_data["doc_ids"]

    # Linear probe: 768 -> 768
    head = nn.Linear(768, 768, bias=True).to(device)
    nn.init.eye_(head.weight)
    nn.init.zeros_(head.bias)

    loss_fn = InfoNCELoss(temperature=temperature, symmetric=True, normalize_embeddings=True)
    optimizer = AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs)

    trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)

    history: list[dict[str, Any]] = []
    best_val_mrr = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    start_time = time.time()

    logger.info(f"Training {run_name} (LR={lr}, WD={weight_decay}) for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        head.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            q_emb = batch["query_emb"].to(device)
            p_emb = batch["passage_emb"].to(device)

            optimizer.zero_grad()
            q_proj = F.normalize(head(q_emb), p=2, dim=-1)
            p_proj = F.normalize(head(p_emb), p=2, dim=-1)

            loss = loss_fn(q_proj, p_proj)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        train_loss = total_loss / max(1, num_batches)

        # Validation evaluation
        head.eval()
        with torch.no_grad():
            val_q_proj = F.normalize(head(val_q_all), p=2, dim=-1)
            val_p_proj = F.normalize(head(val_p_all), p=2, dim=-1)
            val_loss = loss_fn(val_q_proj, val_p_proj).item()

            # Rank validation passages for validation queries
            sim_matrix = torch.matmul(val_q_proj, val_p_proj.t()).cpu().numpy()
            top_k = min(10, len(val_doc_ids))
            rankings: list[list[str]] = []
            ground_truth: list[set[str]] = []

            for i in range(len(val_doc_ids)):
                scores = sim_matrix[i]
                top_indices = scores.argsort()[::-1][:top_k]
                rankings.append([val_doc_ids[idx] for idx in top_indices])
                ground_truth.append({val_doc_ids[i]})

            val_metrics = evaluate_rankings(rankings, ground_truth, n_bootstrap=100)
            val_mrr10 = val_metrics.get("MRR@10", {}).get("mean", 0.0)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_mrr10": round(val_mrr10, 4),
            "learning_rate": current_lr,
            "time_seconds": round(epoch_time, 2),
        })

        logger.info(
            f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val MRR@10: {val_mrr10:.4f} | Time: {epoch_time:.2f}s"
        )

        # Save last checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mrr10": val_mrr10,
            },
            checkpoints_dir / "last.pt",
        )

        # Save best checkpoint
        if val_mrr10 > best_val_mrr:
            best_val_mrr = val_mrr10
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mrr10": val_mrr10,
                },
                checkpoints_dir / "best.pt",
            )
            logger.info(f"New best model saved with Val MRR@10: {val_mrr10:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    total_time = time.time() - start_time

    # Write CSV history log
    with open(csv_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "val_mrr10", "learning_rate", "time_seconds"],
        )
        writer.writeheader()
        writer.writerows(history)

    # Write run telemetry JSON
    telemetry = {
        "run_name": run_name,
        "approach": "A2",
        "model_type": "xlmr_linear_probe",
        "lr": lr,
        "weight_decay": weight_decay,
        "trainable_parameters": trainable_params,
        "best_epoch": best_epoch,
        "best_val_mrr10": best_val_mrr,
        "total_wall_clock_seconds": round(total_time, 2),
        "device": str(device),
        "history": history,
    }
    with open(metrics_dir / f"{run_name}_train.json", "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    return telemetry


def run_tuning(
    splits_dir: Path = Path("data/05_splits"),
    results_dir: Path = Path("results"),
    epochs: int = 15,
    patience: int = 4,
    batch_size: int = 32,
    seed: int = 42,
) -> Path:
    """Execute hyperparameter tuning grid for Approach A2."""
    set_seed(seed)
    device = get_device()

    # 1. Obtain backbone embedding cache
    cache_dir = results_dir / "cache" / "a2_embeddings"
    train_data, val_data = get_or_create_embedding_cache(splits_dir, cache_dir, device)

    # 2. Tuning grid: LR {1e-3, 1e-2} x Weight Decay {0.0, 0.01}
    grid = [
        {"lr": 1e-3, "weight_decay": 0.0},
        {"lr": 1e-3, "weight_decay": 0.01},
        {"lr": 1e-2, "weight_decay": 0.0},
        {"lr": 1e-2, "weight_decay": 0.01},
    ]

    tuning_dir = results_dir / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    tuning_csv = tuning_dir / "a2.csv"

    tuning_rows: list[dict[str, Any]] = []

    logger.info(f"Starting A2 Linear Probe hyperparameter grid search ({len(grid)} configurations)...")

    for i, params in enumerate(grid, 1):
        lr = params["lr"]
        wd = params["weight_decay"]
        run_name = f"a2_lr{lr:.0e}_wd{wd}".replace("-0", "-")

        logger.info(f"\n[{i}/{len(grid)}] Running configuration: {run_name} (LR={lr}, WD={wd})")

        telemetry = train_linear_probe_run(
            run_name=run_name,
            lr=lr,
            weight_decay=wd,
            train_data=train_data,
            val_data=val_data,
            results_dir=results_dir,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            temperature=0.05,
            seed=seed,
            device=device,
        )

        best_epoch_info = max(telemetry["history"], key=lambda h: h["val_mrr10"])
        tuning_rows.append({
            "run_name": run_name,
            "lr": lr,
            "weight_decay": wd,
            "trainable_params": telemetry["trainable_parameters"],
            "best_epoch": best_epoch_info["epoch"],
            "best_val_mrr10": telemetry["best_val_mrr10"],
            "val_loss": round(best_epoch_info["val_loss"], 4),
            "train_loss": round(best_epoch_info["train_loss"], 4),
            "train_time_sec": telemetry["total_wall_clock_seconds"],
            "checkpoint_path": str(results_dir / "checkpoints" / run_name / "best.pt"),
        })

    # Save tuning summary
    df_tuning = pd.DataFrame(tuning_rows)
    df_tuning.sort_values(by="best_val_mrr10", ascending=False, inplace=True)
    df_tuning.to_csv(tuning_csv, index=False)
    logger.info(f"Saved tuning summary table -> {tuning_csv}")

    print("\n" + "=" * 80)
    print("A2 XLM-R LINEAR PROBE HYPERPARAMETER TUNING SUMMARY")
    print("=" * 80)
    print(df_tuning.to_string(index=False))
    print("=" * 80 + "\n")

    # 3. Instantiate full model and save winning checkpoint
    best_config = df_tuning.iloc[0]
    best_head_ckpt = Path(best_config["checkpoint_path"])
    best_ckpt_dest = results_dir / "checkpoints" / "a2_linear_probe" / "best.pt"
    best_ckpt_dest.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Selected winning model: '{best_config['run_name']}' "
        f"(Val MRR@10: {best_config['best_val_mrr10']:.4f})"
    )

    full_model = XLMRLinearProbeRetriever(device=device)
    saved_head_data = torch.load(best_head_ckpt, weights_only=True)
    full_model.proj.load_state_dict(saved_head_data["model_state_dict"])

    save_checkpoint(
        best_ckpt_dest,
        model=full_model,
        epoch=int(best_config["best_epoch"]),
        best_val_mrr=float(best_config["best_val_mrr10"]),
    )
    logger.info(f"Assembled and saved full model checkpoint -> {best_ckpt_dest}")

    # 4. Evaluate winning model once on full corpus benchmarks
    logger.info("Evaluating selected A2 model on primary (T-Q) and secondary (T-T) benchmarks...")
    harness = EvaluationHarness(
        corpus_path=splits_dir / "corpus.jsonl",
        qa_path=splits_dir.parent.parent / "tests" / "evaluation" / "ground_truth_qa_kh.json",
        test_titles_path=splits_dir / "test_titles.jsonl",
        seed=seed,
    )

    retriever = DenseRetriever(full_model, harness.corpus, device=device, batch_size=batch_size)
    harness.run_full_evaluation(retriever, run_name="a2_linear_probe", output_dir=results_dir / "metrics")

    return best_ckpt_dest


if __name__ == "__main__":
    run_tuning()
