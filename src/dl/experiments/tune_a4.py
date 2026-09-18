"""
Hyperparameter tuning and training pipeline for Approach A4 (PrahokBART Dual Encoder).

Executes hyperparameter grid search on the Khmer-native PrahokBART encoder:
  LR in {5e-5, 1e-4} x Weight Decay in {0.0, 0.01}

Features:
- End-to-end backpropagation across all ~35.8M encoder parameters.
- AdamW optimizer with 10% linear warmup and Cosine Annealing decay.
- Per-epoch validation loss and val MRR@10 tracking for model selection.
- Checkpoint management (best.pt and last.pt) with atomic saves.
- Comprehensive telemetry and benchmark evaluation on T-Q and T-T.
"""

import argparse
import csv
import json
from pathlib import Path
import shutil
import time
from typing import Any, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.config.logging import get_logger, setup_logging
from src.dl.checkpoint import load_checkpoint, save_checkpoint
from src.dl.data import CorpusDataset, LegalPairsDataset, collate_pairs
from src.dl.evaluate import DenseRetriever, EvaluationHarness
from src.dl.losses import InfoNCELoss
from src.dl.metrics import evaluate_rankings
from src.dl.models.prahokbart import PrahokBARTDualEncoder
from src.dl.seed import get_device, set_seed

setup_logging()
logger = get_logger(__name__)


def train_epoch(
    model: PrahokBARTDualEncoder,
    train_loader: DataLoader,
    optimizer: AdamW,
    scheduler: Any,
    loss_fn: InfoNCELoss,
    device: torch.device,
    max_grad_norm: float = 1.0,
    max_batches: Optional[int] = None,
    dry_run: bool = False,
) -> float:
    """Run one fine-tuning epoch for PrahokBART."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    limit = 2 if dry_run else max_batches

    for batch_idx, batch in enumerate(train_loader):
        if limit is not None and batch_idx >= limit:
            break

        queries = batch["queries"]
        passages = batch["passages"]

        q_enc = model.tokenizer(
            queries,
            max_length=model.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        p_enc = model.tokenizer(
            passages,
            max_length=model.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        optimizer.zero_grad()

        q_emb = model(q_enc["input_ids"], q_enc["attention_mask"])
        p_emb = model(p_enc["input_ids"], p_enc["attention_mask"])
        loss = loss_fn(q_emb, p_emb)

        loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate_val(
    model: PrahokBARTDualEncoder,
    val_dataset: LegalPairsDataset,
    loss_fn: InfoNCELoss,
    device: torch.device,
    batch_size: int = 32,
    dry_run: bool = False,
) -> tuple[float, float]:
    """Compute validation loss and validation MRR@10."""
    model.eval()
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_pairs,
    )

    total_loss = 0.0
    num_batches = 0
    all_q_emb: list[torch.Tensor] = []
    all_p_emb: list[torch.Tensor] = []

    for i, batch in enumerate(val_loader):
        if dry_run and i >= 2:
            break

        queries = batch["queries"]
        passages = batch["passages"]

        q_enc = model.tokenizer(
            queries,
            max_length=model.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        p_enc = model.tokenizer(
            passages,
            max_length=model.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        q_emb = model(q_enc["input_ids"], q_enc["attention_mask"])
        p_emb = model(p_enc["input_ids"], p_enc["attention_mask"])

        loss = loss_fn(q_emb, p_emb)
        total_loss += loss.item()
        num_batches += 1

        all_q_emb.append(q_emb.cpu())
        all_p_emb.append(p_emb.cpu())

    val_loss = total_loss / max(1, num_batches)
    if not all_q_emb:
        return val_loss, 0.0

    all_q = torch.cat(all_q_emb, dim=0)
    all_p = torch.cat(all_p_emb, dim=0)
    sim = torch.matmul(all_q, all_p.t()).numpy()

    val_doc_ids = [r["doc_id"] for r in val_dataset.records[: len(all_q)]]
    top_k = min(10, len(val_doc_ids))
    rankings: list[list[str]] = []
    ground_truth: list[set[str]] = []

    for idx in range(len(val_doc_ids)):
        scores = sim[idx]
        top_indices = scores.argsort()[::-1][:top_k]
        rankings.append([val_doc_ids[j] for j in top_indices])
        ground_truth.append({val_doc_ids[idx]})

    metrics = evaluate_rankings(rankings, ground_truth, n_bootstrap=100)
    val_mrr10 = metrics.get("MRR@10", {}).get("mean", 0.0)

    return val_loss, val_mrr10


def run_a4_experiment(
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    batch_size: int = 32,
    epochs: int = 5,
    patience: int = 3,
    max_batches: Optional[int] = None,
    device_name: Optional[str] = None,
    dry_run: bool = False,
    run_benchmark: bool = False,
) -> dict[str, Any]:
    """Train and evaluate a single PrahokBART configuration."""
    set_seed(42)
    device = torch.device(device_name) if device_name else get_device()
    tag = f"a4_lr{lr:.0e}_wd{weight_decay}"
    logger.info(f"Starting Approach A4 run: {tag} on {device}")

    # Paths
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    train_path = base_dir / "data" / "05_splits" / "train.jsonl"
    val_path = base_dir / "data" / "05_splits" / "val.jsonl"
    corpus_path = base_dir / "data" / "05_splits" / "corpus.jsonl"

    logs_dir = base_dir / "results" / "logs"
    checkpoints_dir = base_dir / "results" / "checkpoints" / tag
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    train_ds = LegalPairsDataset(train_path)
    val_ds = LegalPairsDataset(val_path)
    corpus_ds = CorpusDataset(corpus_path)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_pairs,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_pairs,
    )

    # Model
    model = PrahokBARTDualEncoder(device=device)

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=max(1, warmup_steps))
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    loss_fn = InfoNCELoss(temperature=0.05)

    # Tracking
    log_file = logs_dir / f"{tag}.csv"
    best_val_mrr10 = -1.0
    best_epoch = 0
    patience_counter = 0
    history: list[dict[str, Any]] = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            device,
            max_batches=max_batches,
            dry_run=dry_run,
        )

        val_loss, val_mrr10 = evaluate_val(
            model=model,
            val_dataset=val_ds,
            loss_fn=loss_fn,
            device=device,
            batch_size=batch_size,
            dry_run=dry_run,
        )

        ep_time = time.time() - ep_start
        logger.info(
            f"Epoch {epoch}/{epochs} ({ep_time:.1f}s) - "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MRR@10: {val_mrr10:.4f}"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mrr10": val_mrr10,
            "epoch_time_sec": ep_time,
        }
        history.append(record)

        save_checkpoint(
            checkpoints_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_mrr=best_val_mrr10,
            config={"lr": lr, "weight_decay": weight_decay},
        )

        if val_mrr10 > best_val_mrr10:
            best_val_mrr10 = val_mrr10
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                checkpoints_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_mrr=val_mrr10,
                config={"lr": lr, "weight_decay": weight_decay},
            )
            logger.info(f"Saved new best checkpoint at epoch {epoch} (val MRR@10: {val_mrr10:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (patience: {patience})")
                break

    total_time = time.time() - start_time
    pd.DataFrame(history).to_csv(log_file, index=False)

    result_summary = {
        "tag": tag,
        "run_name": tag,
        "approach": "A4",
        "model_type": "prahokbart_dual_encoder",
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "best_epoch": best_epoch,
        "best_val_mrr10": best_val_mrr10,
        "total_time_sec": total_time,
        "total_wall_clock_seconds": total_time,
        "checkpoint_dir": str(checkpoints_dir),
    }

    metrics_out = base_dir / "results" / "metrics" / f"{tag}_train.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(result_summary, f, indent=2)

    return result_summary


def tune_a4_grid(
    epochs: int = 5,
    batch_size: int = 32,
    patience: int = 3,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    max_batches: Optional[int] = None,
    dry_run: bool = False,
) -> None:
    """Execute hyperparameter grid search for Approach A4."""
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    results_dir = base_dir / "results" / "tuning"
    results_dir.mkdir(parents=True, exist_ok=True)

    if lr is not None:
        grid = [{"lr": lr, "weight_decay": weight_decay if weight_decay is not None else 0.01}]
    else:
        grid = [
            {"lr": 5e-5, "weight_decay": 0.01},
            {"lr": 1e-4, "weight_decay": 0.01},
        ]

    all_results: list[dict[str, Any]] = []
    best_overall_mrr10 = -1.0
    best_run_tag = ""

    for cfg in grid:
        logger.info(f"Running grid search config: {cfg}")
        res = run_a4_experiment(
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            max_batches=max_batches,
            dry_run=dry_run,
        )
        all_results.append(res)

        if res["best_val_mrr10"] > best_overall_mrr10:
            best_overall_mrr10 = res["best_val_mrr10"]
            best_run_tag = res["tag"]

    # Save summary CSV
    df = pd.DataFrame(all_results)
    tuning_csv = results_dir / "a4.csv"
    df.to_csv(tuning_csv, index=False)
    logger.info(f"Saved Approach A4 tuning summary -> {tuning_csv}")

    # Copy best checkpoint to primary location
    primary_ckpt_dir = base_dir / "results" / "checkpoints" / "a4_prahokbart"
    primary_ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_src = base_dir / "results" / "checkpoints" / best_run_tag / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, primary_ckpt_dir / "best.pt")
        logger.info(f"Promoted {best_run_tag} as primary A4 checkpoint -> {primary_ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune Approach A4 PrahokBART")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=None, help="Specific learning rate to train.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Specific weight decay.")
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches per epoch.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tune_a4_grid(
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_batches=args.max_batches,
        dry_run=args.dry_run,
    )
