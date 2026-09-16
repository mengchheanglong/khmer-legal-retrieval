"""
YAML-driven training loop and experiment runner for Khmer legal retrievers.

Handles:
1. Training loop with InfoNCE loss and in-batch negatives.
2. Per-epoch validation and MRR@10 evaluation for model selection.
3. Checkpoint management: saves last.pt and best.pt with --resume support.
4. Per-epoch metric logging to results/logs/<run_name>.csv.
5. Telemetry recording: trainable parameters, runtime, device, and peak GPU memory
   to results/metrics/<run_name>_train.json.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import yaml

from src.config.logging import get_logger, setup_logging
from src.dl.checkpoint import load_checkpoint, save_checkpoint
from src.dl.data import (
    CorpusDataset,
    LegalPairsDataset,
    create_dataloader,
)
from src.dl.losses import InfoNCELoss
from src.dl.metrics import evaluate_rankings
from src.dl.seed import get_device, set_seed

setup_logging()
logger = get_logger(__name__)


class RetrieverTrainer:
    """Trainer coordinating optimization, checkpointing, and evaluation."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: LegalPairsDataset,
        val_dataset: LegalPairsDataset,
        corpus_dataset: CorpusDataset,
        config: dict[str, Any],
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: Dual-encoder retriever PyTorch module.
            train_dataset: Training pairs dataset.
            val_dataset: Validation pairs dataset.
            corpus_dataset: Complete 1,976-article corpus dataset.
            config: Configuration dictionary loaded from YAML or dict.
            device: Computing device (CUDA or CPU).
        """
        self.config = config
        self.run_name = config.get("run_name", "retriever_experiment")
        self.epochs = config.get("epochs", 10)
        self.batch_size = config.get("batch_size", 32)
        self.lr = float(config.get("lr", 1e-3))
        self.weight_decay = float(config.get("weight_decay", 0.0))
        self.temperature = float(config.get("temperature", 0.05))
        self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
        self.patience = config.get("patience", 5)

        self.device = device or get_device()
        self.model = model.to(self.device)

        self.train_loader = create_dataloader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.val_dataset = val_dataset
        self.corpus_dataset = corpus_dataset

        self.loss_fn = InfoNCELoss(
            temperature=self.temperature,
            symmetric=True,
            normalize_embeddings=True,
        )

        # Setup optimizer (trainable parameters only)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.num_trainable_params = sum(p.numel() for p in trainable_params)
        self.optimizer = AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Setup scheduler
        total_steps = len(self.train_loader) * self.epochs
        warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))
        if warmup_steps > 0:
            warmup = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_steps)
            cosine = CosineAnnealingLR(self.optimizer, T_max=max(1, total_steps - warmup_steps))
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps],
            )
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

        # Directories
        self.logs_dir = Path(config.get("logs_dir", "results/logs"))
        self.checkpoints_dir = Path(config.get("checkpoints_dir", f"results/checkpoints/{self.run_name}"))
        self.metrics_dir = Path(config.get("metrics_dir", "results/metrics"))

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.csv_log_path = self.logs_dir / f"{self.run_name}.csv"

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch.

        Args:
            epoch: Epoch index (1-indexed).

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            queries = batch["queries"]
            passages = batch["passages"]

            self.optimizer.zero_grad()

            if hasattr(self.model, "encode_queries") and hasattr(self.model, "encode_passages"):
                q_emb = self.model.encode_queries(queries, device=self.device)
                p_emb = self.model.encode_passages(passages, device=self.device)
            else:
                q_emb, p_emb = self.model(queries, passages)

            loss = self.loss_fn(q_emb, p_emb)
            loss.backward()

            if self.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    def evaluate_val(self) -> tuple[float, float]:
        """Evaluate validation loss and validation MRR@10.

        Returns:
            Tuple of (val_loss, val_mrr10).
        """
        self.model.eval()
        val_loader = create_dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        total_loss = 0.0
        num_batches = 0
        all_q_emb: list[torch.Tensor] = []
        all_p_emb: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in val_loader:
                queries = batch["queries"]
                passages = batch["passages"]

                if hasattr(self.model, "encode_queries") and hasattr(self.model, "encode_passages"):
                    q_emb = self.model.encode_queries(queries, device=self.device)
                    p_emb = self.model.encode_passages(passages, device=self.device)
                else:
                    q_emb, p_emb = self.model(queries, passages)

                loss = self.loss_fn(q_emb, p_emb)
                total_loss += loss.item()
                num_batches += 1

                all_q_emb.append(q_emb.cpu())
                all_p_emb.append(p_emb.cpu())

        val_loss = total_loss / max(1, num_batches)

        # Compute validation MRR@10 against validation passages
        all_q = torch.cat(all_q_emb, dim=0)
        all_p = torch.cat(all_p_emb, dim=0)
        all_q = nn.functional.normalize(all_q, p=2, dim=-1)
        all_p = nn.functional.normalize(all_p, p=2, dim=-1)

        sim = torch.matmul(all_q, all_p.t())  # (V, V)
        val_doc_ids = [r["doc_id"] for r in self.val_dataset.records]

        rankings: list[list[str]] = []
        ground_truth: list[set[str]] = []
        top_k = min(10, len(val_doc_ids))

        for i in range(len(val_doc_ids)):
            scores = sim[i].numpy()
            top_indices = scores.argsort()[::-1][:top_k]
            rankings.append([val_doc_ids[idx] for idx in top_indices])
            ground_truth.append({val_doc_ids[i]})

        metrics = evaluate_rankings(rankings, ground_truth, n_bootstrap=100)
        val_mrr10 = metrics.get("MRR@10", {}).get("mean", 0.0)

        return val_loss, val_mrr10

    def fit(self, resume_path: Optional[Path] = None) -> dict[str, Any]:
        """Execute full training loop with checkpointing and logging.

        Args:
            resume_path: Optional path to checkpoint to resume from.

        Returns:
            Dictionary summarizing training history, parameters, and timings.
        """
        start_epoch = 0
        best_val_mrr = 0.0
        epochs_no_improve = 0

        if resume_path and resume_path.exists():
            checkpoint_info = load_checkpoint(
                resume_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=self.device,
            )
            start_epoch = checkpoint_info["epoch"] + 1
            best_val_mrr = checkpoint_info.get("best_val_mrr", 0.0)
            logger.info(f"Resuming training from epoch {start_epoch}")

        # Initialize CSV log file
        csv_exists = self.csv_log_path.exists()
        csv_file = open(self.csv_log_path, "a", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        if not csv_exists:
            csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_mrr10", "lr", "timestamp"])
            csv_file.flush()

        logger.info(
            f"Beginning training '{self.run_name}': {self.epochs} epochs, "
            f"{self.num_trainable_params:,} trainable parameters on {self.device}"
        )

        start_time = time.time()
        history: list[dict[str, Any]] = []

        for epoch in range(start_epoch, self.epochs):
            epoch_start = time.time()
            train_loss = self.train_epoch(epoch + 1)
            val_loss, val_mrr10 = self.evaluate_val()
            current_lr = self.optimizer.param_groups[0]["lr"]

            epoch_time = time.time() - epoch_start
            now_iso = datetime.now(timezone.utc).isoformat()

            # Record to CSV
            csv_writer.writerow([
                epoch + 1,
                round(train_loss, 5),
                round(val_loss, 5),
                round(val_mrr10, 5),
                f"{current_lr:.6e}",
                now_iso,
            ])
            csv_file.flush()

            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mrr10": val_mrr10,
                "lr": current_lr,
                "epoch_time_seconds": round(epoch_time, 2),
            })

            logger.info(
                f"Epoch {epoch + 1:2d}/{self.epochs:2d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val MRR@10: {val_mrr10:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Save last.pt
            last_path = self.checkpoints_dir / "last.pt"
            save_checkpoint(
                last_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                best_val_mrr=best_val_mrr,
                config=self.config,
            )

            # Check if best model
            if val_mrr10 > best_val_mrr:
                best_val_mrr = val_mrr10
                epochs_no_improve = 0
                best_path = self.checkpoints_dir / "best.pt"
                save_checkpoint(
                    best_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    best_val_mrr=best_val_mrr,
                    config=self.config,
                )
                logger.info(f"New best model saved with Val MRR@10: {best_val_mrr:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(f"Early stopping triggered after {self.patience} epochs without improvement.")
                    break

        csv_file.close()
        total_time = time.time() - start_time

        # Peak GPU memory
        peak_gpu_mb = 0.0
        if torch.cuda.is_available() and self.device.type == "cuda":
            peak_gpu_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

        # Telemetry metrics
        telemetry = {
            "run_name": self.run_name,
            "trainable_parameters": self.num_trainable_params,
            "total_wall_clock_seconds": round(total_time, 2),
            "device": str(self.device),
            "peak_gpu_memory_mb": round(peak_gpu_mb, 2),
            "best_val_mrr10": round(best_val_mrr, 4),
            "completed_epochs": len(history),
            "config": self.config,
            "history": history,
        }

        telemetry_file = self.metrics_dir / f"{self.run_name}_train.json"
        with open(telemetry_file, "w", encoding="utf-8") as f:
            json.dump(telemetry, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved training telemetry -> {telemetry_file}")

        return telemetry
